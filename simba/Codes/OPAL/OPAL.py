import os
import subprocess
import numpy as np
import yaml

from ...Framework_objects import (
    frameworkLattice,
    getGrids,
)
#from ...Framework_elements import dipole
from ...Modules import constants
from ...FrameworkHelperFunctions import saveFile, expand_substitution
from ...Modules import Beams as rbf
import mpi4py
mpi4py.rc.initialize = False

def update_globals(global_settings, beamlen=None, sample_interval=1):
    grids = getGrids()
    with open(
            os.path.join(os.path.dirname(__file__), "globals_Opal.yaml"), "r"
    ) as file:
        opalglobal = yaml.load(file, Loader=yaml.Loader)
    for sc in ['x', 'y', 'z']:
        if f"SC_3D_N{sc}f" in list(global_settings.keys()):
            scconv = sc.upper().replace('Z', 'T')
            global_settings.update({f"M{scconv}": global_settings[f"SC_3D_N{sc}f"]})
    for typ, vals in opalglobal.items():
        for k, v in vals.items():
            if k in global_settings.keys():
                opalglobal[typ].update({k: v})
    if beamlen:
        gridsize = grids.getGridSizes(
            (beamlen / sample_interval)
        )
        opalglobal["fieldsolver"].update({"MX": gridsize, "MY": gridsize, "MT": gridsize})
    return opalglobal

class opalLattice(frameworkLattice):
    def __init__(self, *args, **kwargs):
        super(opalLattice, self).__init__(*args, **kwargs)
        self.code = "opal"
        self.particle_definition = self.start
        self.bunch_charge = None
        self.trackBeam = True
        self.betax = None
        self.betay = None
        self.alphax = None
        self.alphay = None
        self.zstop = None
        self.grids = getGrids()
        self.commandFiles = {}
        self.opalglobal = None
        self.breakstr = "//----------------------------------------------------------------------------"

    def writeElements(self):
        fulltext = ""
        fulltext += f'{self.breakstr}\n// LATTICE\n'
        zstops = []
        for element in list(self.elements.values()):
            elemstr = element.write_Opal()
            if len(elemstr) > 0:
                if not element.subelement:
                    elemedge = self.getSValues(as_dict=True, at_entrance=True)[element.objectname]
                else:
                    elemedge = element.centre[2] - element.length / 2
                if isinstance(element, dipole):
                    elemstr += f" DESIGNENERGY = {self.global_parameters["beam"].centroids.mean_cpz.val * 1e-6}, "
                fulltext += f"{elemstr} elemedge = {elemedge};\n"
                zstops.append(elemedge + element.length)
        self.zstop = max(zstops)
        fulltext += "\n" + self.objectname + ": LINE=("
        for e, element in list(self.elements.items()):
            if len((fulltext + e).splitlines()[-1]) > 60:
                fulltext += "\n"
            fulltext += e.replace('-', '_') + ", "
        fulltext = (fulltext[:-2] + ");\n")
        return fulltext

    def writeOptions(self):
        fulltext = ""
        fulltext += f"{self.breakstr}\n// OPTIONS\n"
        for name, val in self.opalglobal["option"].items():
            fulltext += f"OPTION, {name} = {val};\n"
        return fulltext + "\n"

    def writeDistribution(self):
        fulltext = ""
        fulltext += f"{self.breakstr}\n// DISTRIBUTION\n"
        # try:
        #     # cathstatus = self.file_block["charge"]["cathode"]
        #     # emissionmodel = self.opalglobal["distribution"]["EMISSIONMODEL"]
        #     emitted = ""#f", \n\tEMITTED = {cathstatus}, \n\tEMISSIONMODEL = {emissionmodel}"
        # except:
        emitted = ""
        if "particle_definition" in list(self.file_block["input"].keys()):
            initobj = "laser" if self.file_block["input"]["particle_definition"] == "initial_distribution" else self.startObject.objectname
        else:
            initobj = self.startObject.objectname
        fulltext += f"DIST: DISTRIBUTION, \n\tTYPE = FROMFILE, \n\tFNAME = \"{initobj}.opal\"{emitted};\n"
        return fulltext + "\n"

    def writeFieldSolver(self, none=False):
        fulltext = ""
        fulltext += f"{self.breakstr}\n// FIELDSOLVER\n"
        fulltext += f"FS: FIELDSOLVER "
        if not none:
            for name, val in self.opalglobal["fieldsolver"].items():
                fulltext += f", \n\t{name} = {val}"
        else:
            fulltext += f", \n\tFSTYPE = NONE"
        return fulltext + ";\n"

    def writeBeam(self):
        bea = self.global_parameters["beam"]
        pc = np.mean(bea.cpz.val) / 1e9
        # gamma = (1 + np.mean(self.global_parameters["beam"].cpz)) / 0.511
        npart = len(self.global_parameters["beam"].x)
        charg = -1 * self.global_parameters["beam"].total_charge * 1e6
        fulltext = ""
        fulltext += f"{self.breakstr}\n// BEAM\n"
        fulltext += f"BEAM1: BEAM,\n"
        fulltext += f"\tPARTICLE = ELECTRON,\n"
        fulltext += f"\tPC = {pc},\n"
        fulltext += f"\tNPART = {npart},\n"
        fulltext += f"\tBFREQ = 1,\n"
        fulltext += f"\tBCURRENT = {charg},\n"
        fulltext += f"\tCHARGE = -1;\n"
        return fulltext + "\n"

    def writeTrack(self):
        maxsteps = self.opalglobal["track"]["MAXSTEPS"]
        dt = self.opalglobal["track"]["DT"]
        fulltext = ""
        fulltext += f"{self.breakstr}\n// TRACK\n"
        fulltext += f"TRACK, LINE = {self.objectname},\n"
        fulltext += f"\tBEAM = BEAM1,\n"
        fulltext += f"\tMAXSTEPS = {maxsteps},\n"
        fulltext += "\tDT = {" + str(dt) + "},\n"
        fulltext += "\tZSTOP = {" + str(self.zstop+1e-1) + "};\n"
        return fulltext + "\n"

    def writeRun(self):
        fulltext = ""
        fulltext += f"{self.breakstr}\n// RUN\n"
        fulltext += f"RUN, METHOD = \"PARALLEL-T\",\n"
        fulltext += f"\tBEAM = BEAM1,\n"
        fulltext += f"\tFIELDSOLVER = FS,\n"
        fulltext += f"\tDISTRIBUTION = DIST;\n"
        fulltext += "ENDTRACK;\n"
        return fulltext + "\n"

    def write(self):
        output = ''
        output += self.writeOptions()
        output += self.writeElements()
        output += self.writeDistribution()
        try:
            if self.file_block["charge"]["space_charge_mode"].upper() in ["2D", "3D"]:
                output += self.writeFieldSolver()
        except Exception as e:
            output += self.writeFieldSolver(none=True)
        output += self.writeBeam()
        output += self.writeTrack()
        output += self.writeRun()
        output += "Quit;\n"
        command_file = (
                self.global_parameters["master_subdir"] + "/" + self.objectname + ".in"
        )
        saveFile(command_file, output, "w")
        try:
            self.command_file = (
                self.global_parameters["master_subdir"] + "/" + self.objectname + ".ele"
            )
            saveFile(self.command_file, "", "w")
            for cfileid in self.commandFilesOrder:
                if cfileid in self.commandFiles:
                    cfile = self.commandFiles[cfileid]
                    saveFile(self.command_file, cfile.write(), "a")
        except:
            pass

    def preProcess(self):
        super().preProcess()
        prefix = self.get_prefix()
        fpath = self.read_input_file(prefix, self.particle_definition)
        self.hdf5_to_opal(prefix, fpath)
        beamlen = len(self.global_parameters["beam"].x)
        self.opalglobal = update_globals(self.globalSettings, beamlen=beamlen)
        self.write()

    def postProcess(self):
        for elem in self.screens_and_bpms + [self.endObject]:
            opalbeamname = f'{self.global_parameters["master_subdir"]}/{elem.objectname}_opal.h5'
            try:
                beam = rbf.beam(opalbeamname)
                rbf.hdf5.write_HDF5_beam_file(
                    beam,
                    opalbeamname.replace("_opal.h5", ".hdf5"),
                    centered=False,
                    sourcefilename=opalbeamname,
                    pos=0.0,
                    xoffset=np.mean(beam.x),
                    yoffset=np.mean(beam.y),
                    zoffset=[elem.start[2]],
                )
            except Exception as e:
                print(f"Error reading opal beam file {elem.objectname}_opal.h5", e)
        self.commandFiles = {}

    def hdf5_to_opal(self, prefix="", fpath=""):
        rbf.opal.write_opal_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + self.particle_definition + '.opal',
            subz=self.startObject.start[2],
        )

    def run(self):
        """Run the code with input 'filename'"""
        if not os.name == "nt":
            command = "bash -c '" + " ".join(self.executables[self.code] + [self.objectname + ".in"]) + "'"
            with open(
                os.path.abspath(
                    self.global_parameters["master_subdir"]
                    + "/"
                    + self.objectname
                    + ".log"
                ),
                "w",
            ) as f:
                subprocess.call(
                    command,
                    stdout=f,
                    cwd=self.global_parameters["master_subdir"],
                    env={**os.environ},
                    shell=True
                )

    def elegantCommandFile(self, *args, **kwargs):
        return elegantCommandFile(*args, **kwargs)
