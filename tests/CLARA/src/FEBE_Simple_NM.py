import os
import time
import numpy as np
from SimulationFramework.ClassFiles.Optimise_longitudinal_Elegant import (
    Optimise_Elegant,
)
from SimulationFramework.Modules.merge_two_dicts import merge_two_dicts
from SimulationFramework.Modules.Beams.Particles.mve import MVE as MVE
import argparse

parser = argparse.ArgumentParser(description="Perform an optimisation.")
parser.add_argument("subdir", default=".")
parser.add_argument("charge", default=20, type=int)
parser.add_argument("--sample", default=2, type=int)
parser.add_argument("--scale", default=1, type=float)


class FEBE(Optimise_Elegant):

    def __init__(self, argparse, charge=250, *args, **kwargs):
        self.delete_output_files = True
        super().__init__(*args, delete_output_files=self.delete_output_files, **kwargs)
        self.args = argparse
        self.scaling = 6
        self.startcharge = charge
        self.sample_interval = 2 ** (
            3 * self.args.sample
        )  # How many particles to track (1 = 262k, 2=128, etc)
        self.base_files = (
            "../../../BaseFiles/Base_" + str(self.startcharge) + "pC/"
        )  # This is where to look for the input files (in this case CLA-S02-APER-01.hdf5)
        self.clean = (
            False  # This "cleans" the output directory before running (if True)
        )
        self.doTracking = True
        self.change_to_elegant = False
        self.change_to_astra = False
        self.change_to_gpt = False
        # self.csrbins = 512
        # self.lscbins = 512
        self.deleteFolders = True
        self.MVE = MVE(beam=self.beam)
        self.verbose = False

    def run_optimisation(self):
        self.Nelder_Mead(
            best_changes="./Settings/"
            + str(self.startcharge)
            + "pC/"
            + args.subdir
            + "/nelder_mead_best_changes_Simple.yaml",
            subdir="Nelder_Mead/nelder_mead_"
            + str(self.startcharge)
            + "pC_"
            + args.subdir
            + "",
            step=[args.scale * i for i in self.steps],
            postprocess=False,
            converged=0,
        )

    def before_tracking(self):
        if not os.name == "nt":
            self.framework.defineElegantCommand(ncpu=12)
        else:
            self.framework.defineElegantCommand(ncpu=int(8 / (self.args.sample + 1)))
        csrbins = int(round(2 ** (3 * self.scaling) / self.sample_interval / 166.66, 3))
        csrbins = 8 if csrbins < 8 else csrbins
        csrbins = csrbins if csrbins % 2 == 0 else csrbins + 1
        lscbins = int(round(2 ** (3 * self.scaling) / self.sample_interval / 166.66, 3))
        lscbins = 8 if lscbins < 8 else lscbins
        lscbins = lscbins if lscbins % 2 == 0 else lscbins + 1
        # print('csrbins =', csrbins, 2**(3*self.scaling), self.sample_interval)
        # print('lscbins =', lscbins, 2**(3*self.scaling), self.sample_interval)
        # exit()
        elements = self.framework.elementObjects.values()
        for e in elements:
            e.lsc_enable = True
            e.lsc_bins = lscbins
            # e.current_bins = 0
            e.csr_bins = csrbins
            e.longitudinal_wakefield_enable = True
            e.transverse_wakefield_enable = True
            e.smoothing_half_width = 1
            e.lsc_high_frequency_cutoff_start = 0.2
            e.lsc_high_frequency_cutoff_end = 0.25
            e.smoothing = 1
        lattices = self.framework.latticeObjects.values()
        for latt in lattices:
            latt.lscDrifts = True
            latt.lsc_bins = lscbins
            latt.lsc_high_frequency_cutoff_start = 0.2
            latt.lsc_high_frequency_cutoff_end = 0.25
            latt.smoothing_half_width = 1
            latt.smoothing = 1

    def calculate_constraints(self):
        constraintsList = {}

        linac_indexes = [i for i, name in enumerate(self.linac_names) if "-L0" in name]
        linac_fields = np.array(
            [1e-6 * self.linac_fields[i] for i in linac_indexes]
        )  # 0=Linac2, 1=Linac3, (2=4HC), 3=Linac4
        fhc_indexes = [i for i, name in enumerate(self.linac_names) if "-L4H" in name]
        fhc_field = np.array(
            [1e-6 * self.linac_fields[i] for i in fhc_indexes]
        )  # 2 = 4HC

        self.beam.read_SDDS_beam_file(self.dirname + "/CLA-FEC1-SIM-FOCUS-01.sdds")
        self.beam.mve.slice_length = 0.01e-12

        t = 1e12 * (self.beam.t - np.mean(self.beam.t))
        t_grid = np.linspace(min(t), max(t), 2**8)
        bw = self.beam.rms(t) / (2**4)
        peakIPDF = self.beam.mve.PDF(t, t_grid, bandwidth=bw)
        peakICDF = self.beam.mve.CDF(t, t_grid, bandwidth=bw)
        peakIFWHM, indexes = self.beam.mve.FWHM(t_grid, peakIPDF, frac=2)

        self.beam.slice.bin_time()
        sigmap = np.std(self.beam.p)
        meanp = np.mean(self.beam.p)
        fitp = 100 * sigmap / meanp
        (
            peakI,
            peakIstd,
            peakIMomentumSpread,
            peakIEmittanceX,
            peakIEmittanceY,
            peakIMomentum,
            peakIDensity,
        ) = self.beam.slice.sliceAnalysis()

        x = 1e6 * (self.beam.x - np.mean(self.beam.x))
        sigmax = np.std(x)
        y = 1e6 * (self.beam.y - np.mean(self.beam.y))
        sigmay = np.std(y)

        self.twiss.read_elegant_twiss_files(self.dirname + "/FEBE.twi")
        ipindex = list(self.twiss.elegantData["ElementName"]).index(
            "CLA-FEC1-SIM-FOCUS-01"
        )
        constraintsListFEBE = {
            "field_max": {
                "type": "lessthan",
                "value": linac_fields,
                "limit": 32,
                "weight": 3000,
            },
            "field_max_fhc": {
                "type": "lessthan",
                "value": fhc_field,
                "limit": 50,
                "weight": 3000,
            },
            "momentum_max": {
                "type": "lessthan",
                "value": [self.twiss["cp_eV"][ipindex]],
                "limit": 265e6,
                "weight": 250,
            },
            "momentum_min": {
                "type": "greaterthan",
                "value": [self.twiss["cp_eV"][ipindex]],
                "limit": 245e6,
                "weight": 150,
            },
            "sigma_x": {
                "type": "lessthan",
                "value": float(sigmax),
                "limit": 50,
                "weight": 25,
            },
            "sigma_y": {
                "type": "lessthan",
                "value": float(sigmay),
                "limit": 50,
                "weight": 25,
            },
            "momentum_spread": {
                "type": "lessthan",
                "value": float(abs(fitp)),
                "limit": 1,
                "weight": 5,
            },
            "peakI_min": {
                "type": "greaterthan",
                "value": float(abs(peakI)),
                "limit": 4000,
                "weight": 300,
            },
            "peakI_momentum_spread": {
                "type": "lessthan",
                "value": float(abs(peakIMomentumSpread)),
                "limit": 1,
                "weight": 30,
            },
            "peakIFWHM": {
                "type": "lessthan",
                "value": float(peakIFWHM),
                "limit": 0.05,
                "weight": 50,
            },
            "peakIFraction": {
                "type": "greaterthan",
                "value": float(100 * peakICDF[indexes][-1] - peakICDF[indexes][0]),
                "limit": 90,
                "weight": 25,
            },
            "VBC": {
                "type": "lessthan",
                "value": [abs(self.vbc_angle)],
                "limit": 0.15,
                "weight": 50,
            },
            "emitx": {
                "type": "lessthan",
                "value": [1e6 * abs(self.beam.enx)],
                "limit": 5,
                "weight": 50,
            },
            "emity": {
                "type": "lessthan",
                "value": [1e6 * abs(self.beam.eny)],
                "limit": 5,
                "weight": 50,
            },
        }
        constraintsList = merge_two_dicts(constraintsList, constraintsListFEBE)

        if self.verbose:
            print(self.cons.constraintsList(constraintsList))
        return constraintsList


class FEBE_Mode_1(FEBE):

    parameter_names = [
        ["CLA-L02-LIN-CAV-01", "field_amplitude"],
        ["CLA-L02-LIN-CAV-01", "phase"],
        ["CLA-L03-LIN-CAV-01", "field_amplitude"],
        ["CLA-L03-LIN-CAV-01", "phase"],
        ["CLA-L4H-LIN-CAV-01", "field_amplitude"],
        ["CLA-L4H-LIN-CAV-01", "phase"],
        ["CLA-L04-LIN-CAV-01", "field_amplitude"],
        ["CLA-L04-LIN-CAV-01", "phase"],
        ["CLA-FEA-MAG-SEXT-01", "k2l"],
        ["CLA-FEA-MAG-SEXT-02", "k2l"],
        ["bunch_compressor", "angle"],
        ["CLA-FEH-MAG-QUAD-01", "k1l"],
        ["CLA-FEH-MAG-QUAD-02", "k1l"],
    ]

    steps = [1e6, 2, 1e6, 2, 1e6, 2, 1e6, 2, 1, 1, 0.01, 0.1, 0.1]

    def __init__(self, lattice: str = "./FEBE_2_Bunches.def", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_changes_file(
            [
                "./Settings/"
                + str(self.startcharge)
                + "pC/"
                + self.args.subdir
                + "/transverse_best_changes_upto_S07_Simple.yaml",
                "./Settings/"
                + str(self.startcharge)
                + "pC/"
                + self.args.subdir
                + "/S07_transverse_best_changes_Simple.yaml",
                "./Settings/"
                + str(self.startcharge)
                + "pC/"
                + self.args.subdir
                + "/FEBE_transverse_best_changes.yaml",
            ]
        )
        self.set_lattice_file(lattice)
        self.load_best(
            "./Settings/"
            + str(self.startcharge)
            + "pC/"
            + self.args.subdir
            + "/nelder_mead_best_changes_Simple.yaml"
        )
        self.set_start_file("FEBE")

    def before_tracking(self):
        self.framework.generator.number_of_particles = 2 ** (3 * 6)
        super().before_tracking()
        self.framework["CLA-S07-DCP-EM-01"].factor = 0

    def calculate_constraints(self):
        start = time.time()
        constraintsList = {}

        linac_indexes = [i for i, name in enumerate(self.linac_names) if "-L0" in name]
        linac_fields = np.array(
            [1e-6 * self.linac_fields[i] for i in linac_indexes]
        )  # 0=Linac2, 1=Linac3, (2=4HC), 3=Linac4
        fhc_indexes = [i for i, name in enumerate(self.linac_names) if "-L4H" in name]
        fhc_field = np.array(
            [1e-6 * self.linac_fields[i] for i in fhc_indexes]
        )  # 2 = 4HC

        self.beam.read_SDDS_beam_file(self.dirname + "/CLA-FEC1-SIM-FOCUS-01.sdds")

        # slice_length = 1e-15
        # self.beam.slice.slice_length = slice_length
        # self.beam.slice.bin_time()
        # (
        #     peakI,
        #     peakIstd,
        #     peakIMomentumSpread,
        #     peakIEmittanceX,
        #     peakIEmittanceY,
        #     peakIMomentum,
        #     peakIDensity,
        # ) = self.beam.slice.sliceAnalysis()

        t = 1e12 * (self.beam.t - np.mean(self.beam.t))
        meanp = np.mean(self.beam.cp.val)
        sigmat = np.std(t)
        sigmap = np.std(self.beam.p)
        sigmapp = 100 * sigmap / np.mean(self.beam.p)
        x = 1e6 * (self.beam.x - np.mean(self.beam.x))
        sigmax = np.std(x)
        y = 1e6 * (self.beam.y - np.mean(self.beam.y))
        sigmay = np.std(y)
        charge = self.beam.total_charge * 1e12

        density = charge / (sigmax * sigmay * sigmat)

        constraintsListFEBE = {
            "field_max": {
                "type": "lessthan",
                "value": linac_fields,
                "limit": 31,
                "weight": 3000,
            },
            "field_max_fhc": {
                "type": "lessthan",
                "value": fhc_field,
                "limit": 50,
                "weight": 3000,
            },
            "momentum_max": {
                "type": "lessthan",
                "value": [meanp],
                "limit": 255e6,
                "weight": 250,
            },
            "momentum_min": {
                "type": "greaterthan",
                "value": [meanp],
                "limit": 245e6,
                "weight": 150,
            },
            "density": {
                "type": "greaterthan",
                "value": [float(density)],
                "limit": 20,
                "weight": 25,
            },
            "energyspread": {
                "type": "lessthan",
                "value": [float(sigmapp)],
                "limit": 1,
                "weight": 5,
            },
            "VBC": {
                "type": "lessthan",
                "value": [abs(self.framework["bunch_compressor"].angle)],
                "limit": 0.15,
                "weight": 50,
            },
            # 'emitx': {'type': 'lessthan','value': [1e6 * abs(peakIEmittanceX)], 'limit': 5, 'weight': 50},
            # 'emity': {'type': 'lessthan','value': [1e6 * abs(peakIEmittanceY)], 'limit': 5, 'weight': 50},
        }
        constraintsList = merge_two_dicts(constraintsList, constraintsListFEBE)

        if True or self.verbose:
            print(constraintsList)
            print(self.cons.constraintsList(constraintsList))

        return constraintsList


class FEBE_Mode_2(FEBE_Mode_1):

    parameter_names = [
        ["CLA-L02-CAV", "field_amplitude"],
        ["CLA-L02-CAV", "phase"],
        ["CLA-L03-CAV", "field_amplitude"],
        ["CLA-L03-CAV", "phase"],
        ["CLA-L4H-CAV", "field_amplitude"],
        ["CLA-L4H-CAV", "phase"],
        ["CLA-L04-CAV", "field_amplitude"],
        ["CLA-L04-CAV", "phase"],
        ["CLA-S07-DCP-EM-01", "factor"],
        ["bunch_compressor", "angle"],
        ["CLA-FEA-MAG-SEXT-01", "k2l"],
        ["CLA-FEA-MAG-SEXT-02", "k2l"],
    ]

    steps = [1e6, 2, 1e6, 2, 1e6, 2, 1e6, 2, 0.1, 0.01, 1, 1]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_optimisation(self):
        self.Nelder_Mead(
            best_changes="./Settings/"
            + str(self.startcharge)
            + "pC/"
            + args.subdir
            + "/nelder_mead_best_changes_Simple.yaml",
            subdir="Nelder_Mead/nelder_mead_"
            + str(self.startcharge)
            + "pC_"
            + args.subdir
            + "",
            step=[args.scale * i for i in self.steps],
            postprocess=False,
        )

    def before_tracking(self):
        if not os.name == "nt":
            self.framework.defineElegantCommand(ncpu=12)
        else:
            self.framework.defineElegantCommand(ncpu=4)
        csrbins = int(round(2 ** (3 * self.scaling) / self.sample_interval / 64, 3))
        csrbins = 8 if csrbins < 8 else csrbins
        lscbins = int(round(2 ** (3 * self.scaling) / self.sample_interval / 64, 3))
        lscbins = 8 if lscbins < 8 else lscbins
        elements = self.framework.elementObjects.values()
        for e in elements:
            e.lsc_enable = True
            e.lsc_bins = lscbins
            # e.current_bins = 0
            e.csr_bins = csrbins
            e.longitudinal_wakefield_enable = True
            e.transverse_wakefield_enable = True
            e.smoothing_half_width = 1
            e.smoothing = 1
        lattices = self.framework.latticeObjects.values()
        for latt in lattices:
            latt.lscDrifts = True
            latt.lsc_bins = lscbins
            latt.smoothing_half_width = 1
            latt.smoothing = 1
        self.framework["CLA-S07-DCP-EM-01"].current_bins = lscbins

    def calculate_constraints(self):
        constraintsList = {}

        linac_indexes = [i for i, name in enumerate(self.linac_names) if "-L0" in name]
        linac_fields = np.array(
            [1e-6 * self.linac_fields[i] for i in linac_indexes]
        )  # 0=Linac2, 1=Linac3, (2=4HC), 3=Linac4
        fhc_indexes = [i for i, name in enumerate(self.linac_names) if "-L4H" in name]
        fhc_field = np.array(
            [1e-6 * self.linac_fields[i] for i in fhc_indexes]
        )  # 2 = 4HC

        self.beam.read_SDDS_beam_file(self.dirname + "/CLA-FEC1-FOCUS.sdds")
        self.beam.mve.slice_length = 0.01e-12

        t = 1e12 * (self.beam.t - np.mean(self.beam.t))
        t_grid = np.linspace(min(t), max(t), 2**8)
        bw = self.beam.rms(t) / (2**4)
        peakIPDF = self.beam.mve.PDF(t, t_grid, bandwidth=bw)
        peakICDF = self.beam.mve.CDF(t, t_grid, bandwidth=bw)
        peakIFWHM, indexes = self.beam.mve.FWHM(t_grid, peakIPDF, frac=2)

        self.beam.slice.bin_time()
        sigmap = np.std(self.beam.p)
        meanp = np.mean(self.beam.p)
        fitp = 100 * sigmap / meanp
        (
            peakI,
            _,
            peakIMomentumSpread,
            _,
            _,
            _,
            _,
        ) = self.beam.slice.sliceAnalysis()

        x = 1e6 * (self.beam.x - np.mean(self.beam.x))
        sigmax = np.std(x)
        y = 1e6 * (self.beam.y - np.mean(self.beam.y))
        sigmay = np.std(y)

        self.twiss.read_elegant_twiss_files(self.dirname + "/FEBE.twi")
        ipindex = list(self.twiss.elegantData["ElementName"]).index("CLA-FEC1-FOCUS")

        constraintsListFEBE = {
            "field_max": {
                "type": "lessthan",
                "value": linac_fields,
                "limit": 32,
                "weight": 3000,
            },
            "field_max_fhc": {
                "type": "lessthan",
                "value": fhc_field,
                "limit": 50,
                "weight": 3000,
            },
            "momentum_max": {
                "type": "lessthan",
                "value": [self.twiss["cp_eV"][ipindex]],
                "limit": 265e6,
                "weight": 250,
            },
            "momentum_min": {
                "type": "greaterthan",
                "value": [self.twiss["cp_eV"][ipindex]],
                "limit": 245e6,
                "weight": 150,
            },
            "sigma_x": {
                "type": "lessthan",
                "value": float(sigmax),
                "limit": 75,
                "weight": 25,
            },
            "sigma_y": {
                "type": "lessthan",
                "value": float(sigmay),
                "limit": 75,
                "weight": 25,
            },
            "momentum_spread": {
                "type": "lessthan",
                "value": float(abs(fitp)),
                "limit": 1.5,
                "weight": 5,
            },
            "peakI_min": {
                "type": "greaterthan",
                "value": float(abs(peakI)),
                "limit": 4000,
                "weight": 300,
            },
            "peakI_momentum_spread": {
                "type": "lessthan",
                "value": float(abs(peakIMomentumSpread)),
                "limit": 1.5,
                "weight": 30,
            },
            "peakIFWHM": {
                "type": "lessthan",
                "value": float(peakIFWHM),
                "limit": 0.1,
                "weight": 50,
            },
            "peakIFraction": {
                "type": "greaterthan",
                "value": float(100 * peakICDF[indexes][-1] - peakICDF[indexes][0]),
                "limit": 90,
                "weight": 25,
            },
            "VBC": {
                "type": "lessthan",
                "value": [abs(self.vbc_angle)],
                "limit": 0.15,
                "weight": 50,
            },
        }
        constraintsList = merge_two_dicts(constraintsList, constraintsListFEBE)

        if self.verbose:
            print(self.cons.constraintsList(constraintsList))
        return constraintsList


class FEBE_Mode_3(FEBE):

    parameter_names = [
        ["CLA-L02-LIN-CAV-01", "field_amplitude"],
        ["CLA-L02-LIN-CAV-01", "phase"],
        ["CLA-L03-LIN-CAV-01", "field_amplitude"],
        ["CLA-L03-LIN-CAV-01", "phase"],
        ["CLA-L4H-LIN-CAV-01", "field_amplitude"],
        ["CLA-L4H-LIN-CAV-01", "phase"],
        ["CLA-L04-LIN-CAV-01", "field_amplitude"],
        ["CLA-L04-LIN-CAV-01", "phase"],
        ["CLA-FEA-MAG-SEXT-01", "k2l"],
        ["CLA-FEA-MAG-SEXT-02", "k2l"],
        ["bunch_compressor", "angle"],
        ["CLA-S07-DCP-EM-01", "factor"],
    ]

    steps = [1e6, 1, 1e6, 1, 1e6, 1, 1e6, 1, 5, 5, 0.00025, 0.1]

    def __init__(self, lattice: str = "./FEBE_2_Bunches.def", *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("hello from FEBE_Mode_3")
        self.set_changes_file(
            [
                "./Settings/"
                + str(self.startcharge)
                + "pC/"
                + self.args.subdir
                + "/transverse_best_changes_upto_S07_Simple.yaml",
                "./Settings/"
                + str(self.startcharge)
                + "pC/"
                + self.args.subdir
                + "/S07_transverse_best_changes_Simple.yaml",
                "./Settings/"
                + str(250)
                + "pC/"
                + str(1)
                + "/FEBE_transverse_best_changes.yaml",
            ]
        )
        self.set_lattice_file(lattice)
        self.set_start_file("FEBE")
        # self.load_best('./Settings/'+str(250)+'pC/'+str(1)+'/nelder_mead_best_changes_Simple.yaml')
        self.load_best(
            "./Settings/"
            + str(self.startcharge)
            + "pC/"
            + self.args.subdir
            + "/nelder_mead_best_changes_Simple.yaml"
        )

    def before_tracking(self):
        super().before_tracking()
        # self.framework['CLA-S07-DCP-EM-01'].bins = 32

    def run_optimisation(self):
        self.Nelder_Mead(
            best_changes="./Settings/"
            + str(self.startcharge)
            + "pC/"
            + args.subdir
            + "/nelder_mead_best_changes_Simple.yaml",
            subdir="Nelder_Mead/nelder_mead_"
            + str(self.startcharge)
            + "pC_"
            + args.subdir
            + "",
            step=[args.scale * i for i in self.steps],
            postprocess=False,
        )

    def calculate_constraints(self):
        constraintsList = {}

        linac_indexes = [i for i, name in enumerate(self.linac_names) if "-L0" in name]
        linac_fields = np.array(
            [1e-6 * self.linac_fields[i] for i in linac_indexes]
        )  # 0=Linac2, 1=Linac3, (2=4HC), 3=Linac4
        fhc_indexes = [i for i, name in enumerate(self.linac_names) if "-L4H" in name]
        fhc_field = np.array(
            [1e-6 * self.linac_fields[i] for i in fhc_indexes]
        )  # 2 = 4HC

        self.beam.read_SDDS_beam_file(self.dirname + "/CLA-FEC1-SIM-FOCUS-01.sdds")

        self.beam.mve.slice_length = 0.01e-12

        self.beam.slice.bin_time()
        sigmap = float(np.std(self.beam.cp))
        meanp = float(np.mean(self.beam.cp))
        sigmapp = float(100 * sigmap / meanp)
        (
            peakI,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self.beam.slice.sliceAnalysis()

        constraintsListFEBE = {
            "field_max": {
                "type": "lessthan",
                "value": linac_fields,
                "limit": 31,
                "weight": 3000,
            },
            "field_max_fhc": {
                "type": "lessthan",
                "value": fhc_field,
                "limit": 50,
                "weight": 3000,
            },
            "momentum_max": {
                "type": "lessthan",
                "value": [meanp],
                "limit": 255e6,
                "weight": 250,
            },
            "momentum_min": {
                "type": "greaterthan",
                "value": [meanp],
                "limit": 245e6,
                "weight": 150,
            },
            # 'density': {'type': 'greaterthan', 'value': [float(density)], 'limit': 2, 'weight': 25},
            "peakI": {
                "type": "greaterthan",
                "value": [1e-3 * float(peakI)],
                "limit": 10,
                "weight": 30,
            },
            "energyspread": {
                "type": "lessthan",
                "value": [float(sigmapp)],
                "limit": 3.0,
                "weight": 5,
            },
            "VBC": {
                "type": "lessthan",
                "value": [abs(self.vbc_angle)],
                "limit": 0.15,
                "weight": 50,
            },
        }
        constraintsList = merge_two_dicts(constraintsList, constraintsListFEBE)

        if self.verbose:
            print(self.cons.constraintsList(constraintsList))
        return constraintsList


class FEBE_Mode_3a(FEBE_Mode_3):

    parameter_names = [
        ["CLA-L4H-CAV", "field_amplitude"],
        ["CLA-L4H-CAV", "phase"],
        ["CLA-L04-CAV", "field_amplitude"],
        ["CLA-L04-CAV", "phase"],
        ["CLA-FEA-MAG-SEXT-01", "k2l"],
        ["CLA-FEA-MAG-SEXT-02", "k2l"],
    ]

    steps = [1e6, 1, 1e6, 1, 5, 5]

    def __init__(self, lattice: str = "./FEBE_2_Bunches.def", *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("hello from FEBE_Mode_3a")
        self.set_changes_file(
            [
                "./Settings/"
                + str(self.startcharge)
                + "pC/"
                + self.args.subdir
                + "/transverse_best_changes_upto_S07_Simple.yaml",
                "./Settings/"
                + str(self.startcharge)
                + "pC/"
                + self.args.subdir
                + "/S07_transverse_best_changes_Simple.yaml",
                "./Settings/"
                + str(250)
                + "pC/"
                + str(1)
                + "/FEBE_transverse_best_changes.yaml",
                "./Settings/"
                + str(250)
                + "pC/"
                + str(1)
                + "/nelder_mead_best_changes_Simple.yaml",
            ]
        )
        self.set_lattice_file(lattice)
        self.set_start_file("FEBE")
        # self.load_best('./Settings/'+str(250)+'pC/'+str(1)+'/nelder_mead_best_changes_Simple.yaml')
        self.load_best(
            "./Settings/"
            + str(self.startcharge)
            + "pC/"
            + self.args.subdir
            + "/nelder_mead_best_changes_Simple.yaml"
        )


class FEBE_Mode_3b(FEBE_Mode_3):
    steps = [3e6, 3, 3e6, 3, 3e6, 3, 3e6, 3, 1, 1, 0.001, 0.3]


class FEBE_Mode_4(FEBE_Mode_1):

    parameter_names = [
        ["CLA-L02-CAV", "field_amplitude"],
        ["CLA-L03-CAV", "field_amplitude"],
        ["CLA-L4H-CAV", "field_amplitude"],
        ["CLA-L04-CAV", "field_amplitude"],
        ["CLA-L4H-CAV", "phase"],
    ]

    steps = [10e6, 10e6, 10e6, 10e6, 30]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def before_tracking(self):
        self.framework["CLA-L02-CAV"].phase = 0
        self.framework["CLA-L03-CAV"].phase = 0
        self.framework["CLA-L04-CAV"].phase = 0
        super().before_tracking()


class FEBE_Mode_5(FEBE):

    parameter_names = [
        ["CLA-S07-MAG-QUAD-01", "k1l"],
        ["CLA-S07-MAG-QUAD-02", "k1l"],
        ["CLA-S07-MAG-QUAD-03", "k1l"],
        ["CLA-S07-MAG-QUAD-04", "k1l"],
        ["CLA-S07-MAG-QUAD-05", "k1l"],
        ["CLA-S07-MAG-QUAD-06", "k1l"],
        ["CLA-S07-MAG-QUAD-07", "k1l"],
        ["CLA-S07-MAG-QUAD-08", "k1l"],
        ["CLA-S07-MAG-QUAD-09", "k1l"],
        ["CLA-S07-MAG-QUAD-10", "k1l"],
    ]

    steps = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

    def __init__(self, lattice: str = "./FEBE_2_Bunches.def", *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("hello from FEBE_Mode_5")
        self.set_changes_file(
            [
                "./Settings/"
                + str(250)
                + "pC/"
                + str(1)
                + "/FEBE_transverse_best_changes.yaml",
                "./Settings/"
                + str(self.startcharge)
                + "pC/"
                + self.args.subdir
                + "/transverse_best_changes_upto_S07_Simple.yaml",
                "./Settings/"
                + str(self.startcharge)
                + "pC/"
                + self.args.subdir
                + "/nelder_mead_best_changes_Simple.yaml",
            ]
        )
        self.set_lattice_file(lattice)
        self.set_start_file("FEBE")
        # self.load_best('./Settings/'+str(250)+'pC/'+str(1)+'/nelder_mead_best_changes_Simple.yaml')
        self.load_best(
            "./Settings/"
            + str(self.startcharge)
            + "pC/"
            + self.args.subdir
            + "/S07_transverse_best_changes_Simple.yaml"
        )

    def run_optimisation(self):
        self.Nelder_Mead(
            best_changes="./Settings/"
            + str(self.startcharge)
            + "pC/"
            + args.subdir
            + "/S07_transverse_best_changes_Simple.yaml",
            subdir="Nelder_Mead/nelder_mead_"
            + str(self.startcharge)
            + "pC_"
            + args.subdir
            + "",
            step=[args.scale * i for i in self.steps],
        )

    def calculate_constraints(self):
        constraintsList = {}

        linac_indexes = [i for i, name in enumerate(self.linac_names) if "-L0" in name]
        linac_fields = np.array(
            [1e-6 * self.linac_fields[i] for i in linac_indexes]
        )  # 0=Linac2, 1=Linac3, (2=4HC), 3=Linac4
        fhc_indexes = [i for i, name in enumerate(self.linac_names) if "-L4H" in name]
        fhc_field = np.array(
            [1e-6 * self.linac_fields[i] for i in fhc_indexes]
        )  # 2 = 4HC

        self.beam.read_HDF5_beam_file(self.dirname + "/CLA-FEC1-FOCUS.hdf5")
        t = 1e12 * (self.beam.t - np.mean(self.beam.t))
        meanp = np.mean(self.beam.cp)
        sigmat = np.std(t)
        sigmap = np.std(self.beam.p)
        sigmapp = 100 * sigmap / np.mean(self.beam.p)
        x = 1e6 * (self.beam.x - np.mean(self.beam.x))
        sigmax = np.std(x)
        y = 1e6 * (self.beam.y - np.mean(self.beam.y))
        sigmay = np.std(y)
        charge = self.beam.total_charge * 1e12

        density = charge / (sigmax * sigmay * sigmat)

        constraintsListFEBE = {
            "field_max": {
                "type": "lessthan",
                "value": linac_fields,
                "limit": 31,
                "weight": 3000,
            },
            "field_max_fhc": {
                "type": "lessthan",
                "value": fhc_field,
                "limit": 50,
                "weight": 3000,
            },
            "momentum_max": {
                "type": "lessthan",
                "value": [meanp],
                "limit": 255e6,
                "weight": 250,
            },
            "momentum_min": {
                "type": "greaterthan",
                "value": [meanp],
                "limit": 245e6,
                "weight": 150,
            },
            "density": {
                "type": "equalto",
                "value": [float(density)],
                "limit": 4.22,
                "weight": 25,
            },
            "energyspread": {
                "type": "lessthan",
                "value": [float(sigmapp)],
                "limit": 1,
                "weight": 5,
            },
            "VBC": {
                "type": "lessthan",
                "value": [abs(self.vbc_angle)],
                "limit": 0.15,
                "weight": 50,
            },
        }
        constraintsList = merge_two_dicts(constraintsList, constraintsListFEBE)

        if self.verbose:
            print(self.cons.constraintsList(constraintsList))
        return constraintsList


class FEBE_Mode_11(FEBE_Mode_3):
    """This is mode 11: Lowest energy spread (5pC)
    We modify the density to be Q/(Sx * Sy * Sp)
    using Sp instead of St!
    """

    def calculate_constraints(self):
        constraintsList = {}

        linac_indexes = [i for i, name in enumerate(self.linac_names) if "-L0" in name]
        linac_fields = np.array(
            [1e-6 * self.linac_fields[i] for i in linac_indexes]
        )  # 0=Linac2, 1=Linac3, (2=4HC), 3=Linac4
        fhc_indexes = [i for i, name in enumerate(self.linac_names) if "-L4H" in name]
        fhc_field = np.array(
            [1e-6 * self.linac_fields[i] for i in fhc_indexes]
        )  # 2 = 4HC

        self.beam.read_HDF5_beam_file(self.dirname + "/CLA-FEC1-FOCUS.hdf5")
        meanp = np.mean(self.beam.cp)
        sigmap = np.std(self.beam.p)
        sigmapp = sigmap / np.mean(self.beam.p)
        x = 1e6 * (self.beam.x - np.mean(self.beam.x))
        sigmax = np.std(x)
        y = 1e6 * (self.beam.y - np.mean(self.beam.y))
        sigmay = np.std(y)
        charge = self.beam.total_charge * 1e12

        density = charge / (sigmax * sigmay * sigmapp)

        constraintsListFEBE = {
            "field_max": {
                "type": "lessthan",
                "value": linac_fields,
                "limit": 31,
                "weight": 3000,
            },
            "field_max_fhc": {
                "type": "lessthan",
                "value": fhc_field,
                "limit": 50,
                "weight": 3000,
            },
            "momentum_max": {
                "type": "lessthan",
                "value": [meanp],
                "limit": 255e6,
                "weight": 250,
            },
            "momentum_min": {
                "type": "greaterthan",
                "value": [meanp],
                "limit": 245e6,
                "weight": 150,
            },
            "density": {
                "type": "greaterthan",
                "value": [float(density)],
                "limit": 500,
                "weight": 15,
            },
            "energyspread": {
                "type": "lessthan",
                "value": [float(100 * sigmapp)],
                "limit": 0.01,
                "weight": 5,
            },
            "VBC": {
                "type": "lessthan",
                "value": [abs(self.vbc_angle)],
                "limit": 0.15,
                "weight": 50,
            },
        }
        constraintsList = merge_two_dicts(constraintsList, constraintsListFEBE)

        if self.verbose:
            print(self.cons.constraintsList(constraintsList))
        return constraintsList


class FEBE_Mode_99(FEBE_Mode_1):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FEBE_Mode_THz(FEBE_Mode_1):

    parameter_names = [
        ["CLA-L02-LIN-CAV-01", "field_amplitude"],
        ["CLA-L02-LIN-CAV-01", "phase"],
        ["CLA-L03-LIN-CAV-01", "field_amplitude"],
        ["CLA-L03-LIN-CAV-01", "phase"],
        ["CLA-L4H-LIN-CAV-01", "field_amplitude"],
        ["CLA-L4H-LIN-CAV-01", "phase"],
        ["CLA-L04-LIN-CAV-01", "field_amplitude"],
        ["CLA-L04-LIN-CAV-01", "phase"],
        ["bunch_compressor", "angle"],
        ["CLA-S07-DIA-BPM-01", "phase"],
        ["ARC_R56", "r56"],
    ]

    steps = [1e6, 2, 1e6, 2, 1e6, 2, 1e6, 3, 0.01, 10, 0.005]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, lattice="clara400_v13_H3Beams.def", **kwargs)
        self.lattice_file = "clara400_v13_H3Beams.def"

    def before_tracking(self, *args, **kwargs):
        super().before_tracking(*args, **kwargs)
        self.framework["CLA-S07-DIA-BPM-01"].field_amplitude = 10e6
        # self.framework['CLA-S07-DIA-BPM-01'].phase = 275 + 5
        # self.framework['ARC_R56'].r56 = 0.02

    def calculate_constraints(self):
        constraintsList = {}

        linac_fields = np.array(
            [1e-6 * self.linac_fields[i] for i in [0, 1, 3]]
        )  # 0=Linac2, 1=Linac3, (2=4HC), 3=Linac4
        fhc_field = np.array([1e-6 * self.linac_fields[i] for i in [2]])  # 2 = 4HC

        self.beam.read_SDDS_beam_file(self.dirname + "/CLA-S07-SIM-START-01.sdds")

        cp = 1e-6 * self.beam.cp
        cp_range = float(max(cp) - min(cp))
        meanp = float(np.mean(cp))

        self.beam.read_SDDS_beam_file(self.dirname + "/CLA-FEC1-SIM-FOCUS-01.sdds")

        slice_length = (max(self.beam.t) - min(self.beam.t)) / 64
        self.beam.slice.slice_length = slice_length
        self.beam.slice.bin_time()
        (
            peakI,
            peakIstd,
            peakIMomentumSpread,
            peakIEmittanceX,
            peakIEmittanceY,
            peakIMomentum,
            peakIDensity,
        ) = self.beam.slice.sliceAnalysis()

        constraintsListFEBE = {
            "field_max": {
                "type": "lessthan",
                "value": linac_fields,
                "limit": 32,
                "weight": 3000,
            },
            "field_max_fhc": {
                "type": "lessthan",
                "value": fhc_field,
                "limit": 50,
                "weight": 3000,
            },
            "momentum_max": {
                "type": "lessthan",
                "value": meanp,
                "limit": 250,
                "weight": 250,
            },
            "momentum_min": {
                "type": "greaterthan",
                "value": meanp,
                "limit": 200,
                "weight": 150,
            },
            "momentum_spread": {
                "type": "lessthan",
                "value": float(abs(cp_range)),
                "limit": np.sqrt(self.args.charge / 20.0) * 0.25,
                "weight": 10,
            },
            # 'time_spread': {'type': 'equalto', 'value': float(abs(t_range)), 'limit': 2, 'weight': 15},
            "VBC": {
                "type": "lessthan",
                "value": self.vbc_angle,
                "limit": 0.2,
                "weight": 50,
            },
            "peakI_min": {
                "type": "greaterthan",
                "value": float(abs(peakI)),
                "limit": np.sqrt(self.args.charge / 20.0) * 1000,
                "weight": 300,
            },
        }
        print(constraintsListFEBE)
        constraintsList = merge_two_dicts(constraintsList, constraintsListFEBE)

        if self.verbose:
            print(self.cons.constraintsList(constraintsList))
        return constraintsList


class FEBE_Mode_VHEE(FEBE_Mode_THz):

    parameter_names = [
        ["CLA-L02-LIN-CAV-01", "field_amplitude"],
        ["CLA-L02-LIN-CAV-01", "phase"],
        ["CLA-L03-LIN-CAV-01", "field_amplitude"],
        ["CLA-L03-LIN-CAV-01", "phase"],
        ["CLA-L4H-LIN-CAV-01", "field_amplitude"],
        ["CLA-L4H-LIN-CAV-01", "phase"],
        ["CLA-L04-LIN-CAV-01", "field_amplitude"],
        ["CLA-L04-LIN-CAV-01", "phase"],
        ["bunch_compressor", "angle"],
    ]

    def before_tracking(self):
        self.framework.generator.number_of_particles = 2 ** (3 * 6)
        super().before_tracking()
        self.framework["CLA-S07-DCP-EM-01"].factor = 0

    def calculate_constraints(self):
        constraintsList = {}

        linac_fields = np.array(
            [1e-6 * self.linac_fields[i] for i in [0, 1, 3]]
        )  # 0=Linac2, 1=Linac3, (2=4HC), 3=Linac4
        fhc_field = np.array([1e-6 * self.linac_fields[i] for i in [2]])  # 2 = 4HC

        self.beam.read_SDDS_beam_file(self.dirname + "/CLA-FEC1-SIM-FOCUS-01.sdds")

        slice_length = (max(self.beam.t) - min(self.beam.t)) / 64
        self.beam.slice.slice_length = slice_length
        self.beam.slice.bin_time()
        (
            peakI,
            peakIstd,
            peakIMomentumSpread,
            peakIEmittanceX,
            peakIEmittanceY,
            peakIMomentum,
            peakIDensity,
        ) = self.beam.slice.sliceAnalysis()

        t = 1e12 * (self.beam.t - np.mean(self.beam.t))
        t_range = float(max(t) - min(t))

        cp = 1e-6 * self.beam.cp
        cp_range = float(max(cp) - min(cp))
        meanp = float(np.mean(cp))

        constraintsListFEBE = {
            "field_max": {
                "type": "lessthan",
                "value": linac_fields,
                "limit": 32,
                "weight": 3000,
            },
            "field_max_fhc": {
                "type": "lessthan",
                "value": fhc_field,
                "limit": 50,
                "weight": 3000,
            },
            "momentum_max": {
                "type": "lessthan",
                "value": meanp,
                "limit": 255,
                "weight": 150,
            },
            "momentum_min": {
                "type": "greaterthan",
                "value": meanp,
                "limit": 245,
                "weight": 150,
            },
            "momentum_spread": {
                "type": "lessthan",
                "value": float(abs(cp_range)),
                "limit": 0.25,
                "weight": 10,
            },
            "time_spread": {
                "type": "lessthan",
                "value": float(abs(t_range)),
                "limit": 5,
                "weight": 15,
            },
            "VBC": {
                "type": "lessthan",
                "value": self.vbc_angle,
                "limit": 0.15,
                "weight": 50,
            },
            "emitx": {
                "type": "lessthan",
                "value": [1e6 * abs(peakIEmittanceX)],
                "limit": 1,
                "weight": 50,
            },
            "emity": {
                "type": "lessthan",
                "value": [1e6 * abs(peakIEmittanceY)],
                "limit": 1,
                "weight": 50,
            },
        }
        constraintsList = merge_two_dicts(constraintsList, constraintsListFEBE)

        if self.verbose:
            print(self.cons.constraintsList(constraintsList))
        return constraintsList


opt_names = {
    "1": FEBE_Mode_1,
    "1b": FEBE_Mode_1,
    "2": FEBE_Mode_2,
    "3": FEBE_Mode_3,
    "3a": FEBE_Mode_3a,
    "3b": FEBE_Mode_3b,
    "4": FEBE_Mode_4,
    "5": FEBE_Mode_5,
    "10": FEBE_Mode_3,
    "11": FEBE_Mode_11,
    "12": FEBE_Mode_3,
    "test": FEBE_Mode_1,
    "THz": FEBE_Mode_THz,
    "THzLong": FEBE_Mode_THz,
    "VHEE": FEBE_Mode_VHEE,
}

if __name__ == "__main__":
    args = parser.parse_args()
    if args.subdir in opt_names:
        opt = opt_names[args.subdir](argparse=args, charge=int(args.charge))
    else:
        print("subdir", args.subdir, "not found")
        opt = FEBE(argparse=args, charge=int(args.charge))
    print("Performing a Nelder-Mead Optimisation...")
    opt.run_optimisation()
