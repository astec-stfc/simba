"""
Rewrite every import referring to a module/package path renamed for PEP 8
compliance, using simba/_legacy.py's LEGACY_MODULES as the single source of
truth (old dotted path -> new dotted path).

Usage
-----

    python tools/rewrite_legacy_imports.py --check
    python tools/rewrite_legacy_imports.py --apply
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simba._legacy import LEGACY_MODULES as _OLD_TO_NEW  # noqa: E402

SEGMENT_RENAMES: dict[str, str] = {}
for _old, _new in _OLD_TO_NEW.items():
    _old_parts, _new_parts = _old.split("."), _new.split(".")
    if len(_old_parts) != len(_new_parts):
        continue  # not expected to happen; skip rather than mis-map
    for _o, _n in zip(_old_parts, _new_parts):
        if _o == _n:
            continue
        if _o in SEGMENT_RENAMES and SEGMENT_RENAMES[_o] != _n:
            raise ValueError(f"ambiguous segment rename: {_o!r} -> "
                              f"both {SEGMENT_RENAMES[_o]!r} and {_n!r}")
        SEGMENT_RENAMES[_o] = _n


def _rename_dotted(text: str) -> str | None:
    """Rename each dot-separated segment of *text*; None if nothing changed."""
    parts = text.split(".")
    new_parts = [SEGMENT_RENAMES.get(p, p) for p in parts]
    if new_parts == parts:
        return None
    return ".".join(new_parts)


def package_of(path: Path) -> str:
    """Dotted package containing *path* (``simba/codes/astra/astra.py`` -> simba.codes.astra)."""
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    parts.pop()  # drop the leaf filename (module or __init__)
    return ".".join(parts)


def _resolved_dir(node_module: str | None, level: int, pkg: str) -> Path | None:
    """
    Filesystem directory an ``ImportFrom`` targets, in the *current* (already
    renamed) tree, or None if it can't be resolved to a real directory.
    """
    if level == 0:
        # absolute import: resolve purely from node_module, ignoring the
        # current file's own package entirely
        parts = []
    else:
        parts = pkg.split(".") if pkg else []
        if level > 1:
            parts = parts[: len(parts) - (level - 1)]
    if node_module:
        parts += _rename_dotted(node_module).split(".") if _rename_dotted(node_module) \
            else node_module.split(".")
    d = ROOT
    for p in parts:
        d = d / p
    return d if d.is_dir() else None


def rewrite_imports(src: str, path: Path) -> tuple[str, int]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, 0

    pkg = package_of(path)
    edits: list[tuple[int, str, str, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                new_module = _rename_dotted(node.module)
                if new_module is not None:
                    edits.append((node.lineno, "module", node.module, new_module))
            target_dir = _resolved_dir(node.module, node.level, pkg)
            for alias in node.names:
                new_name = SEGMENT_RENAMES.get(alias.name)
                if new_name is None:
                    continue
                if target_dir is None:
                    continue
                if not ((target_dir / f"{new_name}.py").exists()
                        or (target_dir / new_name / "__init__.py").exists()):
                    continue
                edits.append((node.lineno, "name", alias.name, new_name))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                new_module = _rename_dotted(alias.name)
                if new_module is not None:
                    edits.append((node.lineno, "module", alias.name, new_module))

    if not edits:
        return src, 0

    lines = src.splitlines(keepends=True)
    count = 0
    for lineno, kind, old, new in edits:
        i = lineno - 1
        if i >= len(lines) or old not in lines[i]:
            continue
        if kind == "module":
            pat = rf"((?:from|import)\s+\.*){re.escape(old)}(?![\w])"
            new_line, n = re.subn(pat, lambda m: m.group(1) + new, lines[i], count=1)
        else:
            pat = rf"((?:import|,)\s+){re.escape(old)}(?![\w])"
            new_line, n = re.subn(pat, lambda m: m.group(1) + new, lines[i], count=1)
        if n:
            lines[i] = new_line
            count += 1
    return "".join(lines), count


def iter_sources() -> list[Path]:
    out: list[Path] = []
    for base in ("simba", "unit_tests", "tests", "examples"):
        d = ROOT / base
        if not d.is_dir():
            continue
        out += [p for p in sorted(d.rglob("*.py")) if "__pycache__" not in p.parts]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    total = 0
    for p in iter_sources():
        s = p.read_text()
        new_s, n = rewrite_imports(s, p)
        if n:
            total += n
            print(f"  {p.relative_to(ROOT)}: {n} import edits")
            if args.apply:
                p.write_text(new_s)

    print(f"\n{'applied' if args.apply else 'would apply'}: {total} import edits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
