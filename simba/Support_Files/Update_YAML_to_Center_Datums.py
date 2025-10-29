import time, os, subprocess, re, sys
import yaml

sys.path.append("../..")
from SimulationFramework.Framework import *

# from SimulationFramework.FrameworkHelperFunctions import convert_numpy_types
from collections import OrderedDict
from munch import Munch, unmunchify
import csv
from difflib import get_close_matches


class noflow_list(list):
    pass


def noflow_list_rep(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)


class flow_list(list):
    pass


def flow_list_rep(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(noflow_list, noflow_list_rep)
yaml.add_representer(flow_list, flow_list_rep)


def convert_numpy_types(v):
    if isinstance(v, (np.ndarray, list, tuple)):
        return flow_list([convert_numpy_types(l) for l in v])
    elif isinstance(v, (np.float64, np.float32, np.float16, np.float_)):
        return float(v)
    elif isinstance(
        v,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(v)
    else:
        return v


class Converter(Framework):

    def __init__(self, master_lattice=None):
        super(Converter, self).__init__(
            directory="",
            master_lattice=master_lattice,
            overwrite=None,
            runname="",
            clean=False,
            verbose=True,
        )
        self.master_lattice_location = self.global_parameters["master_lattice_location"]
        # self.mariadb_connection = mariadb.connect(host='astecnas2', user='root', password='control123', database='master_lattice')
        # self.cursor = self.mariadb_connection.cursor(buffered=True)
        self.datums = {
            k: v
            for k, v in self.load_datums().items()
            if not "-a" == k[-2:] and not "-b" == k[-2:]
        }
        self.all_data = OrderedDict()

    def load_datums(self):
        datums = {}
        with open(
            r"\\fed.cclrc.ac.uk\Org\NLab\ASTeC-TDL\Projects\tdl-1168 CLARA\CLARA-ASTeC Folder\Layouts\CLARA V12 layout\CLA_V12_layout.csv"
        ) as csvfile:
            datumreader = csv.reader(csvfile, delimiter=",", quotechar="|")
            for row in datumreader:
                elemname = row[1].lower()
                elemname = (
                    elemname[:-2]
                    if (elemname[-2:] == "-k" or elemname[-2:] == "-w")
                    else elemname
                )
                baseelemname = elemname
                i = 1
                if not elemname[-2:].isdigit():
                    elemname = baseelemname + "-" + str(i).zfill(2)
                while elemname in datums:
                    elemname = baseelemname + "-" + str(i).zfill(2)
                    i += 1
                datums[elemname] = [float(row[5]), 0, float(row[4])]
        return datums

    def loadSettings(self, filename="short_240.settings"):
        """Load Lattice Settings from file"""
        global master_run_no
        self.settingsFilename = filename
        # print 'self.settingsFilename = ', self.settingsFilename
        if os.path.exists(filename):
            stream = open(filename, "r")
        else:
            stream = open(self.master_lattice_location + filename, "r")
        self.settings = yaml.load(stream, Loader=yaml.UnsafeLoader)
        self.globalSettings = self.settings["global"]
        master_run_no = (
            self.globalSettings["run_no"] if "run_no" in self.globalSettings else 1
        )
        self.fileSettings = self.settings["files"]
        elements = self.settings["elements"]
        self.groups = (
            self.settings["groups"]
            if "groups" in self.settings and self.settings["groups"] is not None
            else {}
        )
        stream.close()

        # for name, elem in list(self.groups.items()):
        #     group = globals()[elem['type']](name, self.elementObjects, **elem)
        #     self.groupObjects[name] = group

        for name, elem in list(elements.items()):
            self.read_Element(name, elem)

    def read_Lattice(self, name, lattice):
        print(name)

    def add_Element(self, name=None, type=None, **kwargs):
        if name == None:
            if not "name" in kwargs:
                raise NameError("Element does not have a name")
            else:
                name = kwargs["name"]
        # try:
        element = globals()[type](name, type, **kwargs)
        self.elementObjects[name] = element
        melement = Munch(name=name, type=type, **kwargs)
        self.insert_element(melement)

    def default_value(self, element, key, default=0, index=None):
        if key in element:
            if index is not None and isinstance(element[key], (list, tuple)):
                return element[key][index]
            else:
                return element[key]
        else:
            return default

    def read_Element(self, name, element, subelement=None):
        if name == "filename":
            self.load_Elements_File(element)
        else:
            if subelement is not None:
                self.add_Element(name, subelement=subelement, **element)
            else:
                self.add_Element(name, subelement="", **element)
            if "sub_elements" in element:
                for subname, subelem in list(element["sub_elements"].items()):
                    self.read_Element(subname, subelem, subelement=name)

    def load_Elements_File(self, input):
        if isinstance(input, (list, tuple)):
            filename = input
        else:
            filename = [input]
        for f in filename:
            print("#####", f, "#####")
            self.currentfile = f
            self.all_data[self.currentfile] = OrderedDict()
            if os.path.isfile(f):
                with open(f, "r") as stream:
                    elements = yaml.load(stream, Loader=yaml.UnsafeLoader)["elements"]
            else:
                with open(self.master_lattice_location + f, "r") as stream:
                    elements = yaml.load(stream, Loader=yaml.UnsafeLoader)["elements"]
            for name, elem in list(elements.items()):
                self.read_Element(name, elem)
            with open(self.currentfile.replace("YAML/", "newYAML/"), "w") as outfile:
                yaml.dump(
                    {"elements": self.all_data[self.currentfile]},
                    outfile,
                    default_flow_style=False,
                )

    def insert_element(self, element):
        element["centre"] = [
            round(elem, 6) for elem in self.elementObjects[element["name"]].middle
        ]
        lname = element["name"].lower()
        # for r in (("fea", "feb"), ("feh", "feb"), ("fed", "feb")):
        #     lname = lname.replace(*r)
        if "dip" in lname:
            lname = lname + "-d"
        match = self.closeMatches(self.datums, lname)
        finalmatch = None
        mdiff = 100
        if not ("-fea-" in lname or "-feh-" in lname or "-fed-" in lname):
            if not match == []:
                for m in match:
                    dx, dy, dz = self.datums[m.lower()]
                    ex, ey, ez = element["position_end"]
                    diff = ((dx - ex) ** 2 + (dz - ez) ** 2) ** 0.5
                    if diff < 0.001:
                        finalmatch = m
                        mdiff = diff
                    elif diff < mdiff:
                        finalmatch = m
                        mdiff = diff
                if mdiff > 0.001:
                    print(
                        "Match found but diff big",
                        mdiff,
                        ez,
                        dz,
                        ex,
                        dx,
                        lname,
                        finalmatch,
                    )
        element["start"] = [
            round(elem, 7) for elem in self.elementObjects[element["name"]].start
        ]
        element["end"] = [
            round(elem, 7) for elem in self.elementObjects[element["name"]].end
        ]
        if "-fea-" in lname or "-feh-" in lname or "-fed-" in lname:
            element["datum"] = [
                round(elem, 7) for elem in self.elementObjects[element["name"]].middle
            ]
        else:
            element["datum"] = [
                round(elem, 7) for elem in self.elementObjects[element["name"]].end
            ]
        if not "global_rotation" in element:
            element["global_rotation"] = [0, 0, 0]
        # if element['type'] == 'dipole' and ('-fea-' in lname or '-feh-' in lname or '-fed-' in lname):
        #     element['arc_centre'] = [ round(elem, 6) for elem in self.elementObjects[element['name']].arc_middle ]
        #     element['line_centre'] = [ round(elem, 6) for elem in self.elementObjects[element['name']].line_middle ]
        #     element['TD_centre'] = [ round(elem, 6) for elem in self.elementObjects[element['name']].TD_middle ]
        if (
            element["type"] == "screen"
            and ("-fea-" in lname or "-feh-" in lname or "-fed-" in lname)
            and not "mask" in lname
            and not "ctr" in lname
            and not "fed-dia-scr-02-wide" in lname
        ):
            element["datum"] = [
                round(elem, 7)
                for elem in self.elementObjects[
                    element["name"]
                ].relative_position_from_centre([0, 0, -0.0167])
            ]
        if element["type"] == "beam_position_monitor" and (
            "-fea-" in lname or "-feh-" in lname or "-fed-" in lname
        ):
            if "inside" in lname:
                element["datum"] = [
                    round(elem, 7)
                    for elem in self.elementObjects[
                        element["name"]
                    ].relative_position_from_start([0, 0, 0.0353])
                ]
            else:
                element["datum"] = [
                    round(elem, 7)
                    for elem in self.elementObjects[
                        element["name"]
                    ].relative_position_from_centre([0, 0, -0.0697])
                ]
        # if finalmatch is None or mdiff > 0.001 or True:
        #     # print('No Match found  ', lname, finalmatch)
        #     element['old_datum'] = [ round(elem, 6) for elem in element['position_end']]
        # else:
        #     element['old_datum'] = [ round(elem, 6) for elem in self.datums[finalmatch.lower()]]
        del element["start"]
        del element["end"]

        subelem = element["subelement"]
        newelement = dict()
        [
            newelement.update({k: convert_numpy_types(element[k])})
            for k in element.keys()
            if not "position" in k
            and not "buffer" in k
            and not "subelement" in k
            and not "Online_Model_Name" in k
            and not "Controller_Name" in k
            and not "name" in k
        ]
        name = element["name"]
        element = newelement
        if not subelem == "":
            print("found subelement ", subelem, name)
            self.all_data[self.currentfile][subelem]["sub_elements"][name] = element
        else:
            self.all_data[self.currentfile][name] = element

    def closeMatches(self, patterns, word):
        return get_close_matches(word, patterns, 3, 0.3)

    def insert_element_type(self, element):
        type = element["type"]
        try:
            cols = self.get_columns(type)
            valuesstring = [[c, element[c]] for c in cols if c in element]
            cols, valuesstring = zip(*valuesstring)
            colstring = ", ".join(cols)
            valuesstring = [
                1 if e is True else 0 if e is False else e for e in valuesstring
            ]
            valuestring = ", ".join(["%s" for c in cols])
            # self.cursor.execute("""INSERT IGNORE INTO """+type+""" ("""+colstring+""") VALUES ("""+valuestring+""")""", valuesstring)
        except Exception as e:
            print("#####ERRROR#####", type, e)

    def get_columns(self, table):
        # self.cursor.execute("SHOW COLUMNS from "+table+" from master_lattice")
        # print (c)
        return [c[0] for c in self.cursor]


fw = Converter(
    master_lattice="C:/Users/jkj62/Documents/GitHub/20240205 masterlattice/MasterLattice"
)
# fw.loadSettings('Lattices/clara400_v12_FEBE.def')
#
# fw.read_Element('filename', ['YAML/Injector400.yaml', 'YAML/S02.yaml','YAML/L02.yaml',
#              'YAML/S03.yaml', 'YAML/L03.yaml', 'YAML/S04.yaml', 'YAML/L4H.yaml',
#              'YAML/S05.yaml', 'YAML/VBC.yaml', 'YAML/S06.yaml', 'YAML/L04.yaml',
#              'YAML/S07_FEBE.yaml', 'YAML/FEBE_ARC.yaml', 'YAML/FEBE5_long_laser_input.yaml',
#              'YAML/S07_FEBE_STRAIGHT_ON.yaml'])
#
# # fw.read_Element('filename', ['YAML/VBC.yaml'])
#
# exit()

import glob

basedir = "../../../20240205 masterlattice/MasterLattice/"
print(os.path.abspath(basedir))
print(glob.glob(basedir + "YAML/*.yaml"))
filenames = [
    a.replace(basedir, "").replace("\\", "/")
    for a in glob.glob(basedir + "YAML/*.yaml")
    if not "FRS" in a
]
for f in filenames:
    try:
        fw.read_Element("filename", f)
    except Exception as e:
        print("Error - ", f, e)
