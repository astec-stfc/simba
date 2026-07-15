"""Tests for RF-Track's registration into SIMBA's code framework, and (when
RF_Track is installed) a full build/track/postProcess integration test.

RF-Track is not on PyPI; whether it's importable depends on the environment.
The registration tests below don't need it. ``TestRealRFTrack`` needs both
RF_Track and a working ``simba`` install (all of simba's own heavy deps —
xsuite/ocelot/cheetah/etc.) and is skipped if either is unavailable.
"""
import os
import shutil

import pytest

import simba.Framework_lattices as frameworkLattices
from simba.Codes.RFTrack.RFTrack import rftrackLattice
from simba.Framework_objects import frameworkLattice


def test_rftrack_lattice_is_a_frameworklattice():
    assert issubclass(rftrackLattice, frameworkLattice)


def test_rftrack_code_default():
    assert rftrackLattice.model_fields["code"].default == "rftrack"


def test_rftrack_discoverable_via_introspection():
    """Mirrors Framework.py:159's own derivation of `supported_codes`."""
    supported_codes = [
        code.split("Lattice")[0].lower()
        for code in dir(frameworkLattices)
        if "lattice" in code.lower()
    ]
    assert "rftrack" in supported_codes


def test_rftrack_resolved_by_class_name_lookup():
    """Mirrors Framework.py:774's `getattr(frameworkLattices, code.lower() + "Lattice")`."""
    assert getattr(frameworkLattices, "rftrack" + "Lattice") is rftrackLattice


def _build_drift_framework(tmp_path, charge, initial_momentum=5e6, tracking_type=None):
    """Build a real `simba.Framework` for a minimal single-drift section (two
    markers 0.2 m apart) with the given ``charge`` settings block, generate a
    beam, and track it. Returns the tracked framework. Drift-only geometry keeps
    transmission guaranteed regardless of energy, so these tests exercise the
    Volume/Bunch6dT machinery (not focusing physics). Shared by the cathode and
    tracking-switch tests below."""
    import simba.Framework as fw
    from simba.Codes.Generators import frameworkGenerator
    from laura.models.element import Marker
    from laura import LAURA
    from laura.Exporters.YAML import export_machine

    outdir = str(tmp_path)
    m1 = Marker(name="M1", machine_area="INJ", hardware_class="Marker",
                physical={"middle": {"x": 0.0, "y": 0.0, "z": 0.0}})
    m2 = Marker(name="M2", machine_area="INJ", hardware_class="Marker",
                physical={"middle": {"x": 0.0, "y": 0.0, "z": 0.2}})
    sections = {"sections": {"INJ": ["M1", "M2"]}}
    layouts = {"default_layout": "line1", "layouts": {"line1": ["INJ"]}}
    machine = LAURA(element_list=[m1, m2], layout=layouts, section=sections)
    export_machine(path=f"{outdir}/lattice", machine=machine, overwrite=True)

    gen = frameworkGenerator(
        global_parameters={"master_subdir": outdir},
        filename="M1.openpmd.hdf5",
        initial_momentum=initial_momentum,
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3,
        sigma_z=1e-3, sigma_pz=1e3,
        gaussian_cutoff_x=3, gaussian_cutoff_y=3, gaussian_cutoff_z=3,
        gaussian_cutoff_px=3, gaussian_cutoff_py=3, gaussian_cutoff_pz=3,
        charge=100e-12, number_of_particles=200,
    )
    gen.write()

    settings = fw.FrameworkSettings()
    settings.files = {
        "INJ": {
            "code": "rftrack",
            "charge": charge,
            "output": {"start_element": "M1", "end_element": "M2"},
        }
    }
    settings.layout = machine.layout
    settings.section = {"sections": {name: e.names for name, e in machine.sections.items()}}
    settings.element_list = os.path.join(outdir, "lattice")

    framework = fw.Framework(
        machine=machine, directory=os.path.join(outdir, "rftrack"), clean=True,
    )
    framework.loadSettings(settings=settings)
    framework["INJ"].lsc_enable = False
    framework["INJ"].csr_enable = False
    if tracking_type is not None:
        framework["INJ"].tracking_type = tracking_type
    framework.set_lattice_prefix("INJ", outdir + "/")
    framework.track()
    # Beam/twiss files land in the lattice's master_subdir, not `outdir`.
    return framework, framework["INJ"].global_parameters["master_subdir"]


class TestRealRFTrack:
    def test_import_rf_track(self):
        pytest.importorskip("RF_Track")

    def test_cathode_section_tracks_as_volume(self, tmp_path):
        """A cathode section (charge.cathode=True) must auto-select Volume /
        Bunch6dT tracking so mirror-charge space charge is modelled (§5.1.1,
        §7.5), track successfully, and write its end beam + twiss."""
        pytest.importorskip("RF_Track")
        framework, outdir = _build_drift_framework(
            tmp_path, charge={"cathode": True, "space_charge_mode": "3D"}
        )
        latt = framework["INJ"]
        assert latt._tracking_type() == "volume"
        assert type(latt.pin).__name__ == "Bunch6dT"
        assert type(latt.lat_obj).__name__ == "Volume"
        assert type(latt.pout).__name__ == "Bunch6dT"
        assert latt.pout.get_info().transmission > 0
        assert os.path.isfile(os.path.join(outdir, "M2.openpmd.hdf5"))
        # Volume transport table (Table 4.2 identifiers) was read + saved.
        assert os.path.isfile(os.path.join(outdir, "INJ_twiss.rftrack.hdf5"))
        import simba.Modules.Twiss as rtf
        twiss = rtf.twiss()
        twiss.read_rftrack_twiss_files(os.path.join(outdir, "INJ_twiss.rftrack.hdf5"))
        assert len(twiss.s.val) > 0

    def test_volume_output_beam_recovered(self, tmp_path):
        """The Bunch6dT end beam converts back to a sane SIMBA beam. Uses a
        no-space-charge Volume (forced via the switch) through a pure drift, so
        all particles survive and momentum is exactly preserved -- isolating the
        beam_to_bunch6dt / bunch6dt_to_beam round-trip from lossy SC physics."""
        pytest.importorskip("RF_Track")
        framework, outdir = _build_drift_framework(
            tmp_path, charge={"space_charge_mode": "False", "tracking": "volume"}
        )
        import simba.Modules.Beams as rbf
        beam = rbf.beam()
        rbf.openpmd.read_openpmd_beam_file(beam, os.path.join(outdir, "M2.openpmd.hdf5"))
        assert len(beam.x.val) == 200
        assert abs(float(beam.cpz.val.mean()) - 5e6) / 5e6 < 0.05

    def test_tracking_switch_forces_volume_and_lattice(self, tmp_path):
        """The explicit tracking switch overrides the auto default both ways:
        a non-cathode section forced to Volume tracks as Bunch6dT; a cathode
        section forced to Lattice tracks as Bunch6d."""
        pytest.importorskip("RF_Track")
        # Non-cathode, forced Volume via the charge-block `tracking` key.
        fwk, _ = _build_drift_framework(
            tmp_path / "vol", charge={"space_charge_mode": "False", "tracking": "volume"}
        )
        assert fwk["INJ"]._tracking_type() == "volume"
        assert type(fwk["INJ"].pin).__name__ == "Bunch6dT"
        assert fwk["INJ"].pout.get_info().transmission > 0

        # Cathode, forced back to Lattice via the attribute switch.
        fwk2, _ = _build_drift_framework(
            tmp_path / "lat", charge={"cathode": True, "space_charge_mode": "3D"},
            tracking_type="lattice",
        )
        assert fwk2["INJ"]._tracking_type() == "lattice"
        assert type(fwk2["INJ"].pin).__name__ == "Bunch6d"
        assert type(fwk2["INJ"].lat_obj).__name__ == "Lattice"

    def test_full_fodo_build_track_postprocess(self, tmp_path):
        """End-to-end: a real `simba.Framework` builds an RF-Track Lattice from
        a LAURA FODO, tracks a real generated beam through it, and converts the
        result back — exercising write()/preProcess()/run()/postProcess() and
        the Modules/Beams/rftrack.py + Modules/Twiss/rftrack.py conversions
        together. This is the integration test that was originally deferred
        as "not meaningful to write blind"; now verified for real."""
        pytest.importorskip("RF_Track")
        import simba.Framework as fw
        from simba.Codes.Generators import frameworkGenerator
        from laura.models.element import Quadrupole, Marker
        from laura import LAURA
        from laura.Exporters.YAML import export_machine

        outdir = str(tmp_path)
        m1 = Marker(
            name="M1", machine_area="FODO", hardware_class="Marker",
            physical={"middle": {"x": 0.0, "y": 0.0, "z": 0.0}},
        )
        q1f = Quadrupole(
            name="QUAD1F", machine_area="FODO",
            magnetic={"length": 1.0, "k1l": -1},
            physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 0.75}},
        )
        q1d = Quadrupole(
            name="QUAD1D", machine_area="FODO",
            magnetic={"length": 1.0, "k1l": 1.0},
            physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 3.25}},
        )
        m3 = Marker(
            name="M3", machine_area="FODO", hardware_class="Marker",
            physical={"middle": {"x": 0.0, "y": 0.0, "z": 4.0}},
        )
        sections = {"sections": {"FODO": ["M1", "QUAD1F", "QUAD1D", "M3"]}}
        layouts = {"default_layout": "line1", "layouts": {"line1": ["FODO"]}}
        machine = LAURA(element_list=[m1, q1f, q1d, m3], layout=layouts, section=sections)
        export_machine(path=f"{outdir}/lattice", machine=machine, overwrite=True)

        gen = frameworkGenerator(
            global_parameters={"master_subdir": outdir},
            filename="M1.openpmd.hdf5",
            initial_momentum=100e6,
            sigma_x=1e-4, sigma_px=1e3,
            sigma_y=1e-4, sigma_py=1e3,
            sigma_z=1e-3, sigma_pz=1e3,
            gaussian_cutoff_x=3, gaussian_cutoff_y=3, gaussian_cutoff_z=3,
            gaussian_cutoff_px=3, gaussian_cutoff_py=3, gaussian_cutoff_pz=3,
            charge=100e-12,
        )
        gen.write()

        settings = fw.FrameworkSettings()
        files = {}
        for sec, elems in machine.sections.items():
            files[sec] = {
                "code": "rftrack",
                "charge": {"space_charge_mode": "False"},
                "input": {
                    "twiss": {
                        "beta_x": 3.2844606, "alpha_x": 2.48956886, "nemit_x": 1e-6,
                        "beta_y": 3.2846606, "alpha_y": -2.48956886, "nemit_y": 1e-6,
                    }
                },
                "output": {"start_element": elems[0].name, "end_element": elems[-1].name},
            }
        settings.files = files
        settings.layout = machine.layout
        settings.section = {"sections": {name: e.names for name, e in machine.sections.items()}}
        settings.element_list = os.path.join(outdir, "lattice")

        framework = fw.Framework(
            machine=machine, directory=os.path.join(outdir, "rftrack"), clean=True,
        )
        framework.loadSettings(settings=settings)
        framework["FODO"].lsc_enable = False
        framework["FODO"].csr_enable = False
        framework.set_lattice_prefix("FODO", outdir + "/")
        framework.track()

        lattice = framework["FODO"]
        assert lattice.lat_obj is not None
        assert lattice.pout is not None
        assert lattice.pout.get_info().transmission > 0

        # Opt-in Twiss reading path (mirrors how a user would call
        # `simba.Modules.Twiss.twiss().read_astra_twiss_files(...)` post-hoc;
        # `Framework` itself has no `.twiss` attribute -- only the separate
        # `frameworkDirectory` load-from-disk helper does).
        import simba.Modules.Twiss as rtf

        twiss = rtf.twiss()
        twiss.read_rftrack_transport_table(lattice.lat_obj, lattice.objectname)
        assert len(twiss.s.val) > 0
