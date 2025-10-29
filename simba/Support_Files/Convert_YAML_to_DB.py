import time, os, subprocess, re, sys
from pyyaml import yaml

sys.path.append("../..")
from SimulationFramework.Framework import *
from collections import OrderedDict
from munch import Munch, unmunchify
import mysql.connector as mariadb

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

        # for name, lattice in list(self.fileSettings.items()):
        #     self.read_Lattice(name, lattice)

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
        # print element
        # self.elementObjects[name] = element
        self.insert_element(melement)
        # return element

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

    # def element_exists(self, element):

    def insert_element(self, element):
        self.insert_element_type(element)
        # print((element['name'], element['type'], element['subelement'], self.default_value(element,'length',0),
        # element['position_start'][0], element['position_start'][1], element['position_start'][2],
        # element['position_end'][0], element['position_end'][1], element['position_end'][2],
        # self.default_value(element, 'global_rotation',default=0, index=0), self.default_value(element, 'global_rotation',default=0, index=1), self.default_value(element, 'global_rotation',default=0, index=2),))
        middle = self.elementObjects[element["name"]].middle
        # print (self.elementObjects[element['name']].middle)
        # exit()
        self.cursor.execute(
            """INSERT IGNORE INTO element (name, type, parent, length, center_x, center_y, center_z, datum_x, datum_y, datum_z, phi, psi, theta)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                element["name"],
                element["type"],
                element["subelement"],
                self.default_value(element, "length", 0),
                middle[0],
                middle[1],
                middle[2],
                element["position_end"][0],
                element["position_end"][1],
                element["position_end"][2],
                self.default_value(element, "global_rotation", default=0, index=0),
                self.default_value(element, "global_rotation", default=0, index=1),
                self.default_value(element, "global_rotation", default=0, index=2),
            ),
        )

        # print (element['name'], element['type'], self.default_value(element,'length',0),
        # element['position_start'][0], element['position_start'][1], element['position_start'][2],
        # element['position_end'][0], element['position_end'][1], element['position_end'][2],
        # element['global_rotation'][0], element['global_rotation'][1], element['global_rotation'][2],)
        self.mariadb_connection.commit()

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
fw.loadSettings("Lattices/clara400_v12_v3.def")
# fw.get_columns('cavity')
