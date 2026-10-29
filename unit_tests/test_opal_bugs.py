from simba.Codes.OPAL.OPAL import update_globals


def test_global_settings_override_defaults():
    g = update_globals({"NBIN": 3, "AUTOPHASE": 0})
    assert g["distribution"]["NBIN"] == 3
    assert g["option"]["AUTOPHASE"] == 0
