"""
Deprecated-name aliasing for the PEP 8 naming migration.

Every renamed public name in ``simba`` keeps working under its old spelling,
via a module ``__getattr__`` (:func:`deprecated_aliases`) or, for methods, the
:class:`DeprecatedMethodAliases` mixin.

Warnings are ``FutureWarning``, not ``DeprecationWarning``, matching the
convention already used by ``laura`` (a sibling package in this ecosystem) so
scripts and notebooks that don't opt into ``DeprecationWarning`` still see
them by default.
"""

from __future__ import annotations

import warnings
from typing import Callable, ClassVar

__all__ = ["deprecated_aliases", "DeprecatedMethodAliases", "SIMBA_RENAMES"]

SIMBA_RENAMES: dict[str, dict[str, str]] = {}


def deprecated_aliases(
    module_name: str,
    module_globals: dict,
    aliases: dict[str, str],
) -> Callable[[str], object]:
    """
    Build a module ``__getattr__`` serving *aliases* with a ``FutureWarning``.

    Parameters
    ----------
    module_name:
        The importing module's ``__name__``, used in the warning text.
    module_globals:
        The importing module's ``globals()``. Resolved lazily on each lookup, so
        the renamed object does not have to exist yet at call time.
    aliases:
        Mapping of legacy name -> current name.

    Returns
    -------
        A function suitable for assignment to a module's ``__getattr__``.
    """
    SIMBA_RENAMES[module_name] = dict(aliases)

    def __getattr__(name: str) -> object:
        current = aliases.get(name)
        if current is None:
            raise AttributeError(
                f"module {module_name!r} has no attribute {name!r}"
            )
        warnings.warn(
            f"{module_name}.{name} was renamed to {current} for PEP 8 "
            f"compliance. The old name still works but will be removed in a "
            f"future release; update to {module_name}.{current}.",
            FutureWarning,
            stacklevel=2,
        )
        try:
            return module_globals[current]
        except KeyError:  # pragma: no cover -- signals a broken alias map
            raise AttributeError(
                f"{module_name}.{name} is aliased to {current!r}, which does "
                f"not exist in that module. The alias map is out of date."
            ) from None

    return __getattr__


class DeprecatedMethodAliases:
    """
    Mixin serving renamed *methods* under their old names.
    Done *before* pydantic's ``BaseModel`` so this ``__getattr__`` wins, and
    set the mapping on each class::

        class FrameworkLattice(DeprecatedMethodAliases, BaseModel):
            _DEPRECATED_METHOD_ALIASES = {"preProcess": "pre_process"}

    ``__getattr__`` only runs after normal lookup fails, so the renamed method
    is unaffected and this costs nothing until a legacy name is used.
    """

    _DEPRECATED_METHOD_ALIASES: ClassVar[dict[str, str]] = {}

    @classmethod
    def _merged_aliases(cls) -> dict[str, str]:
        """
        Alias tables from the whole MRO, most-derived winning.
        """
        merged: dict[str, str] = {}
        for klass in reversed(cls.__mro__):
            merged.update(vars(klass).get("_DEPRECATED_METHOD_ALIASES", {}))
        return merged

    def __init_subclass__(cls, **kwargs) -> None:
        """
        Warn when a subclass defines a method under its *old* name.
        """
        super().__init_subclass__(**kwargs)
        inherited: dict[str, str] = {}
        for klass in reversed(cls.__mro__[1:]):
            inherited.update(vars(klass).get("_DEPRECATED_METHOD_ALIASES", {}))
        for legacy, current in inherited.items():
            if legacy in vars(cls):
                warnings.warn(
                    f"{cls.__module__}.{cls.__qualname__} defines {legacy!r}, "
                    f"which simba renamed to {current!r}. simba now calls "
                    f"{current!r}, so this override will never run. Rename it "
                    f"to {current!r}.",
                    FutureWarning,
                    stacklevel=2,
                )

    def __getattr__(self, name: str):
        current = type(self)._merged_aliases().get(name)
        if current is not None:
            warnings.warn(
                f"{type(self).__name__}.{name} was renamed to {current} for "
                f"PEP 8 compliance. The old name still works but will be "
                f"removed in a future release.",
                FutureWarning,
                stacklevel=2,
            )
            return getattr(self, current)
        # Defer to pydantic's BaseModel.__getattr__ (private attrs, extras).
        parent = getattr(super(), "__getattr__", None)
        if parent is not None:
            return parent(name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )
