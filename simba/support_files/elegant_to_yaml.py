from simba._compat import DeprecatedMethodAliases
import time, os, subprocess, re, sys
from ruamel.yaml import YAML

sys.path.append("../..")
from SimulationFramework.framework import *
import SimulationFramework.modules.read_twiss_file as rtf
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

with open(
    os.path.dirname(os.path.abspath(__file__))
    + "/../keyword_conversion_rules_elegant.yaml",
    "r",
) as infile:
    keyword_conversion_rules_elegant = yaml.load(infile, Loader=yaml.UnsafeLoader)

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/../type_conversion_rules.yaml", "r"
) as infile:
    type_conversion_rules = yaml.load(infile, Loader=yaml.UnsafeLoader)
    type_conversion_rules_elegant = type_conversion_rules["elegant"]
    type_conversion_rules_names = type_conversion_rules["name"]


def find_key(dict, value):
    try:
        return list(dict.keys())[list(dict.values()).index(value)]
    except:
        return None


def find_key_string(dict, name):
    for substring, type in dict.items():
        if substring.lower() in name:
            return type
    return None


class ElegantInterpret(DeprecatedMethodAliases, object):

    _DEPRECATED_METHOD_ALIASES = {
        "readElegantFile": "read_elegant_file",
        "readLine": "read_line",
    }

    def __init__(self, parent=None):
        self.elements = {}
        self.lines = {}
        # self.lattice = {'elements': self.elements, 'lines': self.lines}

    def read_elegant_file(self, file):
        alllines = []
        with open(file, "r") as f:
            str = ""
            for line in f:
                str += line
                if not "&" in line:
                    alllines.append(str)
                    str = ""
        for l in alllines:
            self.read_line(l)
        # print(self.elements)
        # print(self.lines)
        # exit()
        self.lattice = [
            [k, OrderedDict([[e, self.elements[e]] for e in v])]
            for k, v in self.lines.items()
        ]
        return OrderedDict(self.lattice)

    def read_line(self, element):
        string = element.replace("&", "").replace("\n", "").replace(";", "")
        if "!" in string:
            pos = string.index("!")
            string = string[:pos].strip()
        if not "line=" in string.lower():
            # try:
            pos = string.index(":")
            name = string[:pos].strip().lower()
            keywords = string[(pos + 1) :].split(",")
            commandtype = keywords[0].strip().lower()
            if commandtype in elements_elegant:
                kwargs = {}
                kwargs["name"] = name.strip().lower()
                name = kwargs["name"]
                kwargs["type"] = commandtype.strip().lower()
                key = find_key(type_conversion_rules_elegant, kwargs["type"])
                if key is not None:
                    # print('Found type ', kwargs['type'], ' which is now', key)
                    kwargs["type"] = key
                key = find_key_string(type_conversion_rules_names, name)
                if key is not None:
                    # print('Found name type ', name, ' which is now', key)
                    kwargs["type"] = key
                self.keyword_conversion_rules_elegant = (
                    keyword_conversion_rules_elegant["general"]
                )
                if kwargs["type"] in keyword_conversion_rules_elegant:
                    # print('found ', kwargs['type'], ' keywords!')
                    self.keyword_conversion_rules_elegant = merge_two_dicts(
                        self.keyword_conversion_rules_elegant,
                        keyword_conversion_rules_elegant[kwargs["type"]],
                    )
                for kw in keywords[1:]:
                    kwname = kw.split("=")[0].strip()
                    kwvalue = kw.split("=")[1].strip()
                    if kwname.lower() in elements_elegant[commandtype]:
                        key = find_key(
                            self.keyword_conversion_rules_elegant, kwname.lower()
                        )
                        if key is not None:
                            # print('found ', kwname, ' in rules -> ', key)
                            kwname = key
                        kwargs[kwname.lower()] = kwvalue
                self.elements[name] = kwargs
            else:
                print("Commandtype not found = ", commandtype)
        # except:
        #     pass
        else:
            try:
                pos = string.index(":")
                name = string[:pos]
                pos1 = string.index("(")
                pos2 = string.index(")")
                lines = [
                    x.strip().lower() for x in string[(pos1 + 1) : pos2].split(",")
                ]
                # print 'lines = ', lines
                name = name.lower()
                self.lines[name] = lines
                # print( 'added line ', name.lower())
            except:
                pass
        return element


class Elegant2Yaml(Framework):

    def __init__(self):
        super(Elegant2Yaml, self).__init__(
            directory="",
            master_lattice=None,
            overwrite=None,
            runname="",
            clean=False,
            verbose=True,
        )
        global master_lattice
        master_lattice = self.master_lattice

    def load_elegant_file(self, filename, floor_filename=None):
        interpret = ElegantInterpret()
        lattice = interpret.read_elegant_file(filename)
        twiss = rtf.Twiss()
        if floor_filename is None:
            dot_index = filename.rfind(".")
            floor_filename = filename[:dot_index] + ".flr"
        xyz = twiss.read_elegant_floor_file(floor_filename)
        # print(xyz)
        # exit()
        if len(lattice.keys()) == 1:
            latt = lattice[list(lattice.keys())[0]]
            for n, start, end, rot in xyz:
                if n.lower() in latt:
                    latt[n.lower()]["position_start"] = start
                    latt[n.lower()]["position_end"] = end
                    latt[n.lower()]["global_rotation"] = rot
                    print(n.lower(), start, rot)
        # print(latt)


fw = Elegant2Yaml()
fw.load_elegant_file("..\Examples\example\VBC.lte")

exit()

import glob

dir = "../../MasterLattice/YAML/"
filenames = [
    a.replace("../../MasterLattice/", "").replace("\\", "/")
    for a in glob.glob(dir + "*.yaml")
]
fw.read_element("filename", filenames)


# ---------------------------------------------------------------------------
# Backwards compatibility: names renamed for PEP 8. Served lazily with a
# FutureWarning so downstream consumers keep working.
# ---------------------------------------------------------------------------
from simba._compat import deprecated_aliases  # noqa: E402

__getattr__ = deprecated_aliases(
    __name__,
    globals(),
    {
        "elegant2YAML": "Elegant2Yaml",
        "elegantInterpret": "ElegantInterpret",
        "type_conversion_rules_Elegant": "type_conversion_rules_elegant",
        "type_conversion_rules_Names": "type_conversion_rules_names",
    },
)
