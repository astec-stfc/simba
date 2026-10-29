from types import SimpleNamespace as NS

from simba.Codes.Genesis.Genesis import (
    genesis_alter_beam_command,
    genesis_setup_command,
    genesis_track_command,
    genesisLattice,
)


def _el(name, start, end):
    return NS(name=name, physical=NS(start=NS(z=start), end=NS(z=end)), magnetic=NS(length=end - start))


def test_chicane_uses_physical_positions():
    group = NS(elementObjects=[_el("D1", 0, 0.2), _el("D2", 1.2, 1.4), _el("D4", 3.0, 3.2)], r56=1e-4)
    fake = NS(
        chicanes="CH", groupObjects={"CH": group}, split_element=None,
        section=NS(to_genesis=lambda split_element, chicanes: chicanes),
    )
    ch = genesisLattice.writeElements(fake)["CH"]
    assert ch["drift_length"] == 1.0 and ch["length"] == 3.2


def test_lambda_is_written():
    assert "lambda = 2e-09" in genesis_alter_beam_command(**{"lambda": 2e-9}).write_Genesis()


def test_setup_seed_per_instance_and_bunchharm_one():
    kw = {"rootname": "a", "lattice": "b", "beamline": "c", "gamma0": 1, "lambda0": 1, "delz": 1, "npart": 1, "nbins": 1}
    seeds = {genesis_setup_command(**kw).seed for _ in range(5)}
    assert len(seeds) > 1
    assert "seed = 7" in genesis_setup_command(seed=7, **kw).write_Genesis()
    assert genesis_track_command(bunchharm=1).bunchharm == 1
