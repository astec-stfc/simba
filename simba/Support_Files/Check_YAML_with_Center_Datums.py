import time, os, subprocess, re, sys
from ruamel.yaml import YAML

sys.path.append("../..")
from SimulationFramework.Framework_centre import *
from collections import OrderedDict
from munch import Munch, unmunchify
import mysql.connector as mariadb
import csv
from difflib import get_close_matches

_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG


def dict_representer(dumper, data):
    return dumper.represent_dict(iter(list(data.items())))


def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))


yaml.add_representer(OrderedDict, dict_representer)
yaml.add_constructor(_mapping_tag, dict_constructor)


class Converter(Framework):

    def __init__(self):
        super(Converter, self).__init__(
            directory="",
            master_lattice=None,
            overwrite=None,
            runname="",
            clean=False,
            verbose=True,
        )
        global master_lattice_location
        master_lattice_location = self.master_lattice_location
        self.mariadb_connection = mariadb.connect(
            host="astecnas2",
            user="root",
            password="control123",
            database="master_lattice",
        )
        self.cursor = self.mariadb_connection.cursor(buffered=True)
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
            stream = open(master_lattice_location + filename, "r")
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
                with open(master_lattice_location + f, "r") as stream:
                    elements = yaml.load(stream, Loader=yaml.UnsafeLoader)["elements"]
            for name, elem in list(elements.items()):
                self.read_Element(name, elem)
            with open(
                self.currentfile.replace("newYAML/", "testYAML/"), "w"
            ) as outfile:
                yaml.dump({"elements": self.all_data[self.currentfile]}, outfile)

    def FSlist(self, l):  # concert list into flow-style (default is block style)
        from ruamel.yaml.comments import CommentedSeq

        cs = CommentedSeq(l)
        cs.fa.set_block_style()
        return cs

    def insert_element(self, element):
        element["centre"] = [
            round(elem, 6) for elem in self.elementObjects[element["name"]].middle
        ]
        lname = element["name"].lower()
        for r in (("fea", "feb"), ("feh", "feb"), ("fed", "feb")):
            lname = lname.replace(*r)
        if "dip" in lname:
            lname = lname + "-d"
        match = self.closeMatches(self.datums, lname)
        finalmatch = None
        mdiff = 100
        if not "-feb-" in lname:
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
        if finalmatch is None or mdiff > 0.001:
            # print('No Match found  ', lname, finalmatch)
            if "-feb-" in lname:
                element["start"] = [
                    round(elem, 6)
                    for elem in self.elementObjects[element["name"]].start
                ]
                element["end"] = [
                    round(elem, 6) for elem in self.elementObjects[element["name"]].end
                ]
                # element['datum'] = [ round(elem, 6) for elem in self.elementObjects[element['name']].middle ]
                # if element['type'] == 'screen' and '-feb-' in lname and not 'mask' in lname and not 'ctr' in lname:
                #     element['datum'] = [ round(elem, 6) for elem in self.elementObjects[element['name']].relative_position_from_centre([0,0,-0.0167]) ]
                # if element['type'] == 'beam_position_monitor' and '-feb-' in lname:
                #     element['datum'] = [ round(elem, 6) for elem in self.elementObjects[element['name']].relative_position_from_centre([0,0,-0.0697]) ]
            else:
                element["old_datum"] = [
                    round(elem, 6) for elem in element["position_end"]
                ]
        else:
            element["old_datum"] = [
                round(elem, 6) for elem in self.datums[finalmatch.lower()]
            ]

        subelem = element["subelement"]
        newelement = OrderedDict()
        [
            newelement.update({k: self.convert_numpy_types(element[k])})
            for k in element.keys()
            if not "position" in k
            and not "buffer" in k
            and not "subelement" in k
            and not "Online_Model_Name" in k
            and not "Controller_Name" in k
        ]
        element = newelement
        if not subelem == "":
            print("found subelement ", subelem, element["name"])
            self.all_data[self.currentfile][subelem]["sub_elements"][
                element["name"]
            ] = element
        else:
            self.all_data[self.currentfile][element["name"]] = element

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
            self.cursor.execute(
                """INSERT IGNORE INTO """
                + type
                + """ ("""
                + colstring
                + """) VALUES ("""
                + valuestring
                + """)""",
                valuesstring,
            )
        except Exception as e:
            print("#####ERRROR#####", type, e)

    def get_columns(self, table):
        self.cursor.execute("SHOW COLUMNS from " + table + " from master_lattice")
        # print (c)
        return [c[0] for c in self.cursor]


fw = Converter()
# fw.loadSettings('Lattices/clara400_v12_FEBE.def')

fw.read_Element("filename", ["newYAML/FEBE_arc.yaml"])

exit()

import glob

dir = "../../MasterLattice/YAML/"
filenames = [
    a.replace("../../MasterLattice/", "").replace("\\", "/")
    for a in glob.glob(dir + "*.yaml")
]
fw.read_Element("filename", filenames)
