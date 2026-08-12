from typing import Dict

def merge_two_dicts(y: Dict, x: Dict) -> Dict:
    """
    Combine two dictionaries: first dictionary overwrites keys in the second dictionary

    Parameters
    ----------
    y: Dict
        First dictionary (overwrites second dictionary if keys match)
    x: Dict
        Second dictionary

    Returns
    -------
    Dict
        Merged single dictionary
    """
    if not isinstance(x, (dict, dict)) and not isinstance(y, (dict, dict)):
        return dict()
    elif not isinstance(x, (dict, dict)):
        return y
    elif not isinstance(y, (dict, dict)):
        return x
    else:
        z = x.copy()  # start with x's keys and values
        z.update(y)  # modifies z with y's keys and values & returns None
        return z


def merge_dicts(*dicts: Dict) -> Dict:
    """
    Combine multiple dictionaries: first dictionary overwrites keys in the second dictionary and so on

    Parameters
    ----------
    dicts: Dict of dicts
        Dictionary, ordered by priority

    Returns
    -------
    Dict
        Merged single dictionary
    """
    final_dict = dicts[-1]
    for dic in list(reversed(dicts))[1:]:
        final_dict = merge_two_dicts(dic, final_dict)
    return final_dict
