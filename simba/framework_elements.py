from simba.framework_objects import (
    Chicane,
    SChicane,
    R56Group,
    ElementGroup,
)  # noqa F401

GROUP_CLASSES = {
    "chicane": Chicane,
    "s_chicane": SChicane,
    "r56_group": R56Group,
    "element_group": ElementGroup,
}

disallowed_keywords = [
    "allowedkeywords",
    "conversion_rules",
    "objectdefaults",
    "global_parameters",
    "objectname",
    "beam",
]
