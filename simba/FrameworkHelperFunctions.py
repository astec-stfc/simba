import multiprocessing
import os
import re
from shutil import copyfile
import numpy as np
from deepdiff import DeepDiff

from laura.models.element import Element
from laura.translator.utils.fields import FieldMap

def saveFile(filename, lines=[], mode="w"):
    with open(filename, mode) as stream:
        for line in lines:
            stream.write(line)


def isevaluable(self, s):
    try:
        eval(s)
        return True
    except Exception:
        return False


def path_function(a, b):
    return os.path.abspath(a)


def expand_substitution(self, param, subs={}, elements={}, absolute=False):
    if isinstance(param, (str)):
        subs["master_lattice"] = (
            path_function(
                self.global_parameters["master_lattice"],
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
            for e in elements:
                if e in replaced_str:
                    print("Element is in string!", e, replaced_str)
            return replaced_str
        else:
            return param
    else:
        return param


def clean_directory(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print("clean_directory error:", e)


def copylink(source, destination):
    try:
        copyfile(source, destination)
    except Exception as e:
        print("copylink error!", e)


_CONVERSION = None
"""``(convert, items)`` for the forked workers of :func:`convert_outputs`."""


def _convert_one(index):
    convert, items = _CONVERSION
    convert(items[index])


def convert_outputs(convert, items, workers: int = 1) -> None:
    """
    Call ``convert(item)`` for each item, in ``workers`` forked processes if more than 1.

    The workers inherit ``convert`` and ``items`` through the fork, so neither is pickled,
    but changes they make to Python objects are lost; ``convert`` should only write files.
    Without ``fork`` (Windows) the items are converted one by one.

    Parameters
    ----------
    convert: callable
        Converts one item.
    items: list
        Items to convert.
    workers: int
        Number of processes.
    """
    global _CONVERSION
    items = list(items)
    workers = min(workers, len(items))
    if workers <= 1 or "fork" not in multiprocessing.get_all_start_methods():
        for item in items:
            convert(item)
        return
    _CONVERSION = (convert, items)
    try:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            pool.map(_convert_one, range(len(items)), chunksize=max(1, len(items) // (4 * workers)))
    finally:
        _CONVERSION = None


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
    elif isinstance(v, FieldMap):
        return convert_numpy_types(v.model_dump())
    else:
        return v

def normalize(obj):
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # np.float64, np.int64, etc.
        return obj.item()
    elif isinstance(obj, (int, float)):
        return float(obj)  # Normalize int to float
    else:
        return obj

def deepdiff_to_nested(diff: dict) -> dict:
    """Convert the ``values_changed`` part of a DeepDiff result into nested dicts."""
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

        d = nested
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = {
            "old": change["old_value"],
            "new": change["new_value"],
        }

    return nested

def compare_multiple_models(model_pairs: list[tuple[Element, Element]]) -> dict:
    """Return a nested dict of changes, by element name, for (old, new) model pairs."""
    all_changes = {}
    for old, new in model_pairs:
        old_dump = normalize(old.model_dump())
        new_dump = normalize(new.model_dump())
        if old_dump == new_dump:
            all_changes[old.name] = {}
            continue

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
    """Flatten nested dict keys into a list of (dotted_path, value)."""
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