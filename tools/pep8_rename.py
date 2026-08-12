"""
AST-aware renaming for the PEP 8 naming migration.

Usage
-----
    python tools/pep8_rename.py --map tools/renames/<name>.toml --check
    python tools/pep8_rename.py --map tools/renames/<name>.toml --apply
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import tomllib
from pathlib import Path

try:
    import libcst as cst
except ImportError:  # pragma: no cover
    sys.exit("libcst is required: pip install libcst")


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_map(path: Path) -> tuple[dict[str, str], set[str]]:
    """
    Flatten the TOML sections into one old -> new mapping.
    """
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    merged: dict[str, str] = {}
    for section in ("classes", "functions", "globals"):
        for old, new in raw.get(section, {}).items():
            if old in merged and merged[old] != new:
                raise ValueError(
                    f"{path}: '{old}' mapped to both '{merged[old]}' and '{new}'"
                )
            merged[old] = new

    for old, new in merged.items():
        if old == new:
            raise ValueError(f"{path}: '{old}' maps to itself")

    keep_attrs = set(raw.get("options", {}).get("no_attribute_rename", []))
    unknown = keep_attrs - set(merged)
    if unknown:
        raise ValueError(f"{path}: no_attribute_rename names not in the map: {unknown}")
    return merged, keep_attrs


def _dotted_name(node) -> str:
    """Flatten an Attribute/Name chain into 'a.b.c'."""
    parts: list[str] = []
    while isinstance(node, cst.Attribute):
        parts.append(node.attr.value)
        node = node.value
    if isinstance(node, cst.Name):
        parts.append(node.value)
    return ".".join(reversed(parts))


def _is_internal_import(node) -> bool:
    """
    Whether an import statement refers to simba's own modules.
    """
    if not isinstance(node, cst.ImportFrom):
        return False
    if node.relative:
        return True
    if node.module is None:
        return False
    return _dotted_name(node.module).split(".")[0] == "simba"


class _ProtectedNameCollector(cst.CSTVisitor):
    """
    Records Name nodes that must not be renamed even if they match the map.
    """

    METADATA_DEPENDENCIES = (cst.metadata.ScopeProvider,)

    def __init__(self, targets: set[str], keep_attrs: set[str] | None = None) -> None:
        self.targets = targets
        self.keep_attrs = keep_attrs or set()
        self.protected: set[int] = set()

    def visit_Attribute(self, node: cst.Attribute) -> None:
        """
        Protect ``obj.Name`` where Name is also a schema field.
        """
        if node.attr.value in self.keep_attrs:
            self.protected.add(id(node.attr))

    def visit_Arg(self, node: cst.Arg) -> None:
        if node.keyword is not None:
            self.protected.add(id(node.keyword))

    def visit_Param(self, node: cst.Param) -> None:
        self.protected.add(id(node.name))

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        """
        Protect names imported from outside simba.
        """
        if _is_internal_import(node) or isinstance(node.names, cst.ImportStar):
            return
        for alias in node.names:
            if isinstance(alias.name, cst.Name):
                self.protected.add(id(alias.name))

    def visit_Name(self, node: cst.Name) -> None:
        if node.value not in self.targets:
            return
        try:
            scope = self.get_metadata(cst.metadata.ScopeProvider, node)
        except KeyError:
            return
        if scope is None:
            return

        assignments = scope[node.value]
        if not assignments:
            return

        for assignment in assignments:
            if isinstance(assignment, cst.metadata.BuiltinAssignment):
                self.protected.add(id(node))
                return
            if isinstance(assignment, cst.metadata.ImportAssignment):
                if _is_internal_import(assignment.node):
                    continue
                self.protected.add(id(node))
                return
            bound_in = getattr(assignment, "scope", None)
            if bound_in is not None and not isinstance(
                bound_in, (cst.metadata.GlobalScope, cst.metadata.ClassScope)
            ):
                self.protected.add(id(node))
                return


_ROLE_RE = re.compile(
    r"(:(?:class|func|meth|attr|obj|exc|data|mod):`~?)([A-Za-z_][\w.]*)(`)"
)


class _Renamer(cst.CSTTransformer):
    def __init__(self, renames: dict[str, str], protected: set[int], docstrings: bool):
        self.renames = renames
        self.protected = protected
        self.docstrings = docstrings
        self.count = 0

    def leave_Name(
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> cst.BaseExpression:
        if id(original_node) in self.protected:
            return updated_node
        new = self.renames.get(original_node.value)
        if new is None:
            return updated_node
        self.count += 1
        return updated_node.with_changes(value=new)

    def leave_SimpleString(
        self, original_node: cst.SimpleString, updated_node: cst.SimpleString
    ) -> cst.BaseExpression:
        """
        Rewrite Sphinx cross-reference roles inside docstrings.
        """
        if not self.docstrings:
            return updated_node

        raw = updated_node.value

        def _sub(m: re.Match) -> str:
            head, target, tail = m.groups()
            parts = target.split(".")
            if parts[-1] not in self.renames:
                return m.group(0)
            self.count += 1
            parts[-1] = self.renames[parts[-1]]
            return f"{head}{'.'.join(parts)}{tail}"

        rewritten = _ROLE_RE.sub(_sub, raw)
        if rewritten == raw:
            return updated_node
        return updated_node.with_changes(value=rewritten)


def rename_source(
    source: str,
    renames: dict[str, str],
    docstrings: bool,
    keep_attrs: set[str] | None = None,
) -> tuple[str, int]:
    wrapper = cst.metadata.MetadataWrapper(cst.parse_module(source), unsafe_skip_copy=True)
    collector = _ProtectedNameCollector(set(renames), keep_attrs)
    wrapper.visit(collector)
    transformer = _Renamer(renames, collector.protected, docstrings)
    return wrapper.module.visit(transformer).code, transformer.count


def iter_python_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            files.append(p)
        elif p.is_dir():
            files.extend(
                f
                for f in sorted(p.rglob("*.py"))
                if "__pycache__" not in f.parts and "build" not in f.parts
            )
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", required=True, type=Path, help="TOML rename map")
    ap.add_argument(
        "--paths",
        nargs="*",
        type=lambda s: Path(s).resolve(),
        default=[REPO_ROOT / "simba", REPO_ROOT / "unit_tests"],
        help="files/dirs to rewrite (default: simba/ and unit_tests/)",
    )
    ap.add_argument(
        "--docstrings",
        action="store_true",
        help="also rewrite Sphinx :class:/:func: roles inside docstrings",
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="show a diff, write nothing")
    mode.add_argument("--apply", action="store_true", help="rewrite files in place")
    args = ap.parse_args()

    renames, keep_attrs = load_map(args.map)
    print(f"{len(renames)} renames from {args.map}\n", file=sys.stderr)

    total_files = total_edits = 0
    for path in iter_python_files(args.paths):
        source = path.read_text(encoding="utf-8")
        if not any(old in source for old in renames):
            continue
        try:
            new_source, count = rename_source(source, renames, args.docstrings, keep_attrs)
        except cst.ParserSyntaxError as exc:
            print(f"SKIP {path}: {exc}", file=sys.stderr)
            continue
        if not count or new_source == source:
            continue

        total_files += 1
        total_edits += count
        rel = path.relative_to(REPO_ROOT)
        if args.apply:
            path.write_text(new_source, encoding="utf-8")
            print(f"{rel}: {count} edits")
        else:
            sys.stdout.writelines(
                difflib.unified_diff(
                    source.splitlines(keepends=True),
                    new_source.splitlines(keepends=True),
                    fromfile=f"a/{rel}",
                    tofile=f"b/{rel}",
                )
            )

    verb = "applied" if args.apply else "would apply"
    print(f"\n{verb} {total_edits} edits across {total_files} files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
