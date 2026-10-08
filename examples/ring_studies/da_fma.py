"""Dynamic aperture and frequency map of a small ring, through SIMBA.

A ten-cell FODO ring with a sextupole beside each quadrupole is built in LAURA,
then each code asked for scans the same grid of starting amplitudes:

* ``run_dynamic_aperture()`` gives ``(x, y, turns_survived)`` per grid point;
* ``run_frequency_map()`` gives ``(x, y, Qx, Qy, D)`` per surviving point, where
  ``D = log10|dQ|`` is the tune drift between the two halves of the run;
* ``dynamic_aperture_boundary()`` gives the largest surviving ``x`` at each ``y``.

The results are drawn with :mod:`simba.Modules.plotting.ring`. Run it as::

    python da_fma.py                 # Ocelot
    python da_fma.py ocelot xsuite elegant madx

Before trusting a scan of your own machine:

* the ring needs apertures, or a loss limit.
* install ``nafflib`` (it is in ``requirements.txt``). Without it the tunes come
  from an FFT, and its ~1e-4 error buries the regular part of ``D``.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simba.Framework as fw
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine
from laura.models.element import Marker, Quadrupole, Sextupole
from simba.Codes.Generators import frameworkGenerator
from simba.Modules.plotting.ring import (
    plot_amplitude_map,
    plot_dynamic_aperture,
    plot_frequency_map,
)

HERE = Path(__file__).parent.resolve()
CELLS, CELL_LENGTH = 10, 4.0
TRACKING = {
    "turns": 512,
    # 20 x 10 starting amplitudes, from x_max/nx to x_max (and the same in y)
    "dynamic_aperture": {"nx": 20, "ny": 10, "x_max": 0.02, "y_max": 0.01},
}


def element(cls, name, z, length=0.0, **magnetic):
    physical = {"length": length, "middle": {"x": 0.0, "y": 0.0, "z": z}}
    if magnetic:
        magnetic["length"] = length
        return cls(name=name, machine_area="RING", magnetic=magnetic, physical=physical)
    return cls(name=name, machine_area="RING", hardware_class="Marker", physical=physical)


def ring_machine(directory):
    """The ring, exported where SIMBA will read it. Drifts are filled in by SIMBA."""
    elements = [element(Marker, "START", 0.0)]
    for n in range(CELLS):
        z = n * CELL_LENGTH
        elements += [
            # tunes 2.42 / 1.77; the sextupoles put the aperture near 11 mm in x
            element(Quadrupole, f"QF{n}", z + 0.15, 0.3, k1l=0.7),
            element(Sextupole, f"SF{n}", z + 0.45, 0.2, k2l=8.0),
            element(Quadrupole, f"QD{n}", z + 2.15, 0.3, k1l=-0.6),
            element(Sextupole, f"SD{n}", z + 2.45, 0.2, k2l=-8.0),
        ]
    elements.append(element(Marker, "END", CELLS * CELL_LENGTH))
    section = {
        "sections": {
            "RING": {
                "elements": [e.name for e in elements],
                "geometry": "closed",  # makes it a ring: periodic optics, turns
                "reference_energy": 1e9,  # eV; the design momentum, so no beam is needed
            }
        }
    }
    machine = LAURA(
        element_list=elements,
        layout={"default_layout": "ring", "layouts": {"ring": ["RING"]}},
        section=section,
    )
    export_machine(path=str(directory / "lattice"), machine=machine, overwrite=True)
    return machine, section


def scan(code, directory):
    machine, section = ring_machine(directory)
    settings = fw.FrameworkSettings()
    settings.files = {
        "RING": {
            "code": code,
            "output": {"start_element": "START", "end_element": "END"},
            "tracking": TRACKING,
        }
    }
    settings.layout = machine.layout
    settings.section = section
    settings.element_list = str(directory / "lattice")
    framework = fw.Framework(machine=machine, directory=str(directory), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    # the scan tracks its own grid, but setting a line up still reads a beam
    frameworkGenerator(
        global_parameters={"master_subdir": framework.subdirectory},
        filename="START.openpmd.hdf5", initial_momentum=1e9, number_of_particles=16,
        sigma_x=1e-5, sigma_px=1e3, sigma_y=1e-5, sigma_py=1e3,
        sigma_z=1e-4, sigma_pz=1e3, charge=1e-15,
    ).write()

    ring = framework["RING"]
    ring.preProcess()
    ring.write()
    # a scan that fails warns and returns [] -- read the warnings
    aperture = ring.run_dynamic_aperture()
    footprint = ring.run_frequency_map()
    return aperture, footprint, ring.dynamic_aperture_boundary(aperture)


def main(codes):
    turns = TRACKING["turns"]
    fig, axes = plt.subplots(len(codes), 3, figsize=(17, 4.6 * len(codes)), squeeze=False)
    for row, code in zip(axes, codes):
        aperture, footprint, boundary = scan(code, HERE / "runs" / code)
        x_max = max(x for _, x in boundary) if boundary else 0.0
        print(f"{code}: {len(aperture)} points, {len(footprint)} tuned, "
              f"largest surviving x {x_max * 1e3:.1f} mm")
        plot_dynamic_aperture(aperture, turns, axes=row[0], title=f"{code}: dynamic aperture, {turns} turns")
        plot_frequency_map(footprint, axes=row[1], title=f"{code}: frequency map")
        plot_amplitude_map(footprint, axes=row[2], title=f"{code}: diffusion by amplitude")
    fig.tight_layout()
    fig.savefig(HERE / "da_fma.png", dpi=120)
    print("wrote", HERE / "da_fma.png")


if __name__ == "__main__":
    main(sys.argv[1:] or ["ocelot"])
