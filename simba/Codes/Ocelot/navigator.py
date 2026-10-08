"""
Ocelot's ``Navigator``, without the per-step costs that grow with the lattice.

Ocelot's ``MagneticLattice.totalLen`` is a property that sums every element's
length each time it is read, and the ``Navigator`` reads it two or three times a
step; between physics processes it also sums every length up to the current
element, to find where that element ends.
"""

from contextlib import contextmanager

import numpy as np
from ocelot.cpbd.navi import Navigator, _logger_navi


class _PassLattice:
    """Mixed into a lattice's class for the length of a pass; see
    :func:`lattice_pass`."""

    @property
    def totalLen(self):
        return self._pass_length

    def __deepcopy__(self, memo):
        return self


_pass_classes = {}


@contextmanager
def lattice_pass(lattice):
    """
    Within the block, `lattice` keeps its ``totalLen`` from entry, and a deep copy
    of anything holding it shares it rather than copying it. Nothing may change an
    element's length inside. Re-entrant.
    """
    if isinstance(lattice, _PassLattice):
        yield lattice
        return
    original = type(lattice)
    if original not in _pass_classes:
        _pass_classes[original] = type(
            f"Pass{original.__name__}", (_PassLattice, original), {}
        )
    lattice._pass_length = lattice.totalLen
    lattice.__class__ = _pass_classes[original]
    try:
        yield lattice
    finally:
        lattice.__class__ = original
        del lattice._pass_length


class PassNavigator(Navigator):
    """
    A ``Navigator`` for one pass of an unchanging lattice; see the module.
    """

    def __init__(self, lattice, unit_step=1):
        with lattice_pass(lattice):
            super().__init__(lattice, unit_step=unit_step)
        self._element_ends = None

    def add_physics_proc(self, physics_proc, elem1, elem2) -> None:
        with lattice_pass(self.lat):
            super().add_physics_proc(physics_proc, elem1, elem2)

    def add_physics_processes(self, processes, elem1s, elem2s) -> None:
        with lattice_pass(self.lat):
            super().add_physics_processes(processes, elem1s, elem2s)

    def reset_position(self):
        with lattice_pass(self.lat):
            super().reset_position()
        self._element_ends = None

    def element_end(self, n_elem: int) -> float:
        """Where element `n_elem` ends, summed as Ocelot's ``get_next`` sums it."""
        if self._element_ends is None:
            lengths = np.array([elem.l for elem in self.lat.sequence])
            self._element_ends = [np.sum(lengths[: n + 1]) for n in range(len(lengths))]
        return self._element_ends[n_elem]

    def get_next(self):
        # Ocelot 25.6's Navigator.get_next, with only the element's end changed
        proc_list = self.get_proc_list()

        if len(proc_list) > 0:
            counters = np.array([p.counter for p in proc_list])
            step = counters.min()

            inxs = np.where(counters == step)

            processes = [proc_list[i] for i in inxs[0]]

            phys_steps = np.array([p.step for p in processes]) * self.unit_step

            for p in proc_list:
                p.counter -= step
                if p.counter == 0:
                    p.counter = p.step

            dz = np.min(phys_steps)
        else:
            processes = proc_list
            n_elems = len(self.lat.sequence)
            if n_elems >= self.n_elem + 1:
                L = self.element_end(self.n_elem)
            else:
                L = self.lat.totalLen
            dz = L - self.z0
            phys_steps = np.array([])
        # check if dz overjumps the stop element
        dz, processes, phys_steps = self.check_overjump(dz, processes, phys_steps)
        processes, phys_steps = self.check_proc_bounds(dz, proc_list, phys_steps, processes)

        _logger_navi.debug(
            " Navigator.get_next: process: "
            + " ".join([proc.__class__.__name__ for proc in processes])
        )
        _logger_navi.debug(
            " Navigator.get_next: navi.z0=" + str(self.z0) + " navi.n_elem="
            + str(self.n_elem) + " navi.sum_lengths=" + str(self.sum_lengths)
            + " dz=" + str(dz)
        )
        _logger_navi.debug(
            " Navigator.get_next: element type="
            + self.lat.sequence[self.n_elem].__class__.__name__ + " element name="
            + str(self.lat.sequence[self.n_elem].id)
        )

        self.remove_used_processes(processes)

        return dz, processes, phys_steps
