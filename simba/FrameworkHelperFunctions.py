import os
import re
from shutil import copyfile
import numpy as np
from .Modules.Fields import field
from pydantic import BaseModel
from deepdiff import DeepDiff

from nala.models.element import Element

def readFile(fname):
    with open(fname) as f:
        content = f.readlines()
    return content


def saveFile(filename, lines=[], mode="w"):
    stream = open(filename, mode)
    for line in lines:
        stream.write(line)
    stream.close()


def findSetting(setting, value, dictionary={}):
    """Looks for a 'value' in 'setting' in dict 'dictionary'"""
    settings = []
    for li, e in dictionary.items():
        if isinstance(e, (dict)) and setting in e.keys() and value == e[setting]:
            settings.append([li, e])
    return settings


def findSettingValue(setting, dictionary={}):
    """Finds the value of a setting in dict 'dictionary'"""
    return [k[setting] for k in findSetting(setting, "", dictionary)]


def lineReplaceFunction(line, findString, replaceString, i=None):
    """Searches for, and replaces, the string 'findString' with 'replaceString' in 'line'"""
    global lineIterator
    if findString in line:
        if i is not None:
            lineIterator += 1
            return line.replace("$" + findString + "$", str(replaceString[i]))
        else:
            return line.replace("$" + findString + "$", str(replaceString))
    else:
        return line


def replaceString(lines=[], findString=None, replaceString=None):
    """Iterates over lines and replaces 'findString' with 'replaceString' which can be a list"""
    global lineIterator
    if isinstance(replaceString, list):
        lineIterator = 0
        return [
            lineReplaceFunction(line, findString, replaceString, lineIterator)
            for line in lines
        ]
    else:
        return [lineReplaceFunction(line, findString, replaceString) for line in lines]


def chop(expr, delta=1e-8):
    """Performs a chop on small numbers"""
    if isinstance(expr, (int, float, complex)):
        return 0 if -delta <= expr <= delta else expr
    else:
        return [chop(x, delta) for x in expr]


def chunks(li, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(li), n):
        yield li[i : i + n]


def dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def sortByPositionFunction(element):
    """Sort function for element positions"""
    return float(element[1]["position_start"][2])


def rotationMatrix(theta):
    """Simple 3D rotation matrix"""
    c, s = np.cos(theta), np.sin(theta)
    return np.matrix([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def getParameter(dicts, param, default=0):
    """Returns the values of 'param' in dict 'dict' if it exists, else returns default value. dict can be a list, the most important last."""
    param = param.lower()
    if isinstance(dicts, list) or isinstance(dicts, tuple):
        val = default
        for d in dicts:
            if isinstance(d, dict) or isinstance(d, dict):
                dset = {k.lower(): v for k, v in d.items()}
                if param in dset:
                    val = dset[param]
        return val
    elif isinstance(dicts, dict) or isinstance(dicts, dict):
        dset = {k.lower(): v for k, v in dicts.items()}
        # val = dset[param] if param in dset else default
        if param in dset:
            return dset[param]
        else:
            # print 'not here! returning ', default
            return default
    else:
        # print 'not here! returning ', default
        return default


def formatOptionalString(parameter, string, n=None):
    """String for optional parameters"""
    if n is None:
        return " " + string + "=" + parameter + "\n" if parameter != "None" else ""
    else:
        return (
            " " + string + "(" + str(n) + ")=" + parameter + "\n"
            if parameter != "None"
            else ""
        )


def createOptionalString(paramaterdict, parameter, n=None):
    """Formats ASTRA strings for optional ASTRA parameters"""
    val = str(getParameter(paramaterdict, parameter, default=None))
    return formatOptionalString(val, parameter, n)


def _rotation_matrix(theta):
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-1 * np.sin(theta), 0, np.cos(theta)],
        ]
    )


def isevaluable(self, s):
    try:
        eval(s)
        return True
    except Exception:
        return False


def path_function(a, b):
    # a_drive, a_tail = os.path.splitdrive(os.path.abspath(a))
    # b_drive, b_tail = os.path.splitdrive(os.path.abspath(b))
    # if (a_drive == b_drive):
    #     return os.path.relpath(a, b)
    # else:
    return os.path.abspath(a)


def expand_substitution(self, param, subs={}, elements={}, absolute=False):
    # print(param)
    if isinstance(param, (str)):
        subs["master_lattice_location"] = (
            path_function(
                self.global_parameters["master_lattice_location"],
                self.global_parameters["master_subdir"],
            )
            + "/"
        )
        subs["master_subdir"] = "./"
        regex = re.compile(r"\$(.*)\$")
        s = re.search(regex, param)
        if s:
            if isevaluable(self, s.group(1)) is True:
                replaced_str = str(eval(re.sub(regex, str(eval(s.group(1))), param)))
            else:
                replaced_str = re.sub(regex, s.group(1), param)
            for key in subs:
                replaced_str = replaced_str.replace(key, subs[key])
            if os.path.exists(replaced_str):
                replaced_str = path_function(
                    replaced_str, self.global_parameters["master_subdir"]
                ).replace("\\", "/")
                # print('\tpath exists', replaced_str)
            for e in elements.keys():
                if e in replaced_str:
                    print("Element is in string!", e, replaced_str)
            return replaced_str
        else:
            return param
    else:
        return param


def checkValue(self, d, default=None):
    if isinstance(d, dict):
        if "type" in d and d["type"] == "list":
            if "default" in d:
                return [
                    a if a is not None else b for a, b in zip(d["value"], d["default"])
                ]
            else:
                if isinstance(d["value"], list):
                    return [val if val is not None else default for val in d["value"]]
                else:
                    return None
        else:
            d["value"] = expand_substitution(self, d["value"])
            return (
                d["value"]
                if d["value"] is not None
                else d["default"] if "default" in d else default
            )
    elif isinstance(d, str):
        return (
            getattr(self, d)
            if hasattr(self, d) and getattr(self, d) is not None
            else default
        )


def clean_directory(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print("clean_directory error:", e)


def list_add(list1, list2):
    return [l1 + l2 for l1, l2 in zip(list1, list2)]


def symlink(source, link_name):
    os_symlink = getattr(os, "symlink", None)
    if callable(os_symlink):
        try:
            os_symlink(source, link_name)
        except FileExistsError:
            pass
    else:
        import ctypes

        csl = ctypes.windll.kernel32.CreateSymbolicLinkW
        csl.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
        csl.restype = ctypes.c_ubyte
        flags = 1 if os.path.isdir(source) else 0
        if csl(link_name, source, flags) == 0:
            raise ctypes.WinError()


def copylink(source, destination):
    try:
        copyfile(source, destination)
    except Exception as e:
        print("copylink error!", e)
        pass


def convert_numpy_types(v):
    if isinstance(v, dict):
        return {key: convert_numpy_types(item) for key, item in v.items()}
    elif isinstance(v, (np.ndarray, list, tuple)):
        try:
            return [convert_numpy_types(li) for li in v]
        except TypeError:
            return float(v)
    elif isinstance(v, (np.float64, np.float32, np.float16)):
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
    elif isinstance(v, field):
        return convert_numpy_types(v.model_dump())
    else:
        return v

def pydantic_basemodel_dump_computed_fields(self, *args, **kwargs):
    # Only include computed fields
    computed_keys = {
        f for f in self.__pydantic_decorators__.computed_fields.keys()
    }
    full_dump = BaseModel().model_dump(*args, **kwargs)
    return {k: v for k, v in full_dump.items() if k in computed_keys}

def normalize(obj):
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # np.float64, np.int64, etc.
        return obj.item()
    else:
        return obj

def deepdiff_to_nested(diff: dict) -> dict:
    """
    Convert a DeepDiff result (values_changed only)
    into a nested dictionary structure.
    """
    nested = {}

    if 'values_changed' not in diff:
        return nested

    for path, change in diff['values_changed'].items():
        # Strip the "root" prefix and split the path into keys
        parts = path.replace("root", "").strip(".")
        keys = []
        current = ""
        in_brackets = False

        # Parse keys like ['a']['b'][0]['c'] → ['a','b',0,'c']
        for char in parts:
            if char == "[":
                in_brackets = True
                current = ""
            elif char == "]":
                in_brackets = False
                key = current.strip("'\"")
                keys.append(int(key) if key.isdigit() else key)
            elif in_brackets:
                current += char

        # Build nested dicts
        d = nested
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = {
            "old": change["old_value"],
            "new": change["new_value"],
        }

    return nested

def compare_multiple_models(model_pairs: list[tuple[Element, Element]]) -> dict:
    """
    Given a list of (old_model, new_model) pairs,
    return a nested dictionary of all changes.
    """
    all_changes = {}
    for old, new in model_pairs:
        old_dump = normalize(old.model_dump())
        new_dump = normalize(new.model_dump())
        diff = DeepDiff(old_dump, new_dump, ignore_order=True, significant_digits=10)
        nested_diff = deepdiff_to_nested(diff.to_dict())
        all_changes[old.name] = nested_diff
    return all_changes

def set_deep_attr(obj, dotted_path, value):
    """Set nested attribute using a dotted path like 'a.b.c.d'."""
    attrs = dotted_path.split('.')
    target = obj
    for attr in attrs[:-1]:
        target = getattr(target, attr)
    setattr(target, attrs[-1], value)

def flatten_changes_dict(d, parent_key=""):
    """
    Flattens nested dict keys into dotted paths.
    Returns a list of (dotted_path, value).
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict) and not ("old" in v and "new" in v):
            items.extend(flatten_changes_dict(v, new_key))
        elif isinstance(v, dict) and "new" in v:
            # This node contains the actual value diff
            items.append((new_key, v["new"]))
        else:
            # Simple leaf
            items.append((new_key, v))
    return items