import numpy as np


class constraintsClass:

    def lessthan(self, type, value, limit, weight=1):
        if abs(limit) > 0:
            weight = float(weight) / float(limit)
        if hasattr(value, "__iter__"):
            return np.sum(
                [(weight * np.abs(x - limit)) ** 2 if x > limit else 0 for x in value]
            )
        elif value > limit:
            return (weight * np.abs(value - limit)) ** 2
        else:
            return 0

    def greaterthan(self, type, value, limit, weight=1):
        if abs(limit) > 0:
            weight = float(weight) / float(limit)
        if hasattr(value, "__iter__"):
            return np.sum(
                [(weight * np.abs(x - limit)) ** 2 if x < limit else 0 for x in value]
            )
        elif value < limit:
            return (weight * np.abs(value - limit)) ** 2
        else:
            return 0

    def equalto(self, type, value, limit, weight=1):
        if abs(limit) > 0:
            weight = float(weight) / float(limit)
        if hasattr(value, "__iter__"):
            return np.sum([(weight * np.abs(x - limit)) ** 2 for x in value])
        else:
            return (weight * np.abs(value - limit)) ** 2

    def constraints(self, constraints={}):
        ans = 0
        if isinstance(constraints, dict):
            for k, v in list(constraints.items()):
                if hasattr(self, v["type"].lower()):
                    ans += getattr(self, v["type"].lower())(**v)
        return np.sqrt(ans)

    def formatDict(self, d, tab=0):
        s = ["{\n"]
        for k, v in list(d.items()):
            if isinstance(v, dict):
                v = self.formatDict(v, tab + 1)
            else:
                v = repr(v)

            s.append("%s%r: %s,\n" % ("  " * tab, k, str(v)))
        s.append("%s}" % ("  " * tab))
        return "".join(s)

    def constraintsList(self, constraints={}):
        ans = {}
        if isinstance(constraints, dict):
            for k, v in list(constraints.items()):
                if hasattr(self, v["type"].lower()):
                    ans[k] = {}
                    ans[k]["value"] = v["value"]
                    ans[k]["limit"] = v["limit"]
                    ans[k]["error"] = np.sqrt(getattr(self, v["type"].lower())(**v))
        return self.formatDict(ans)


if __name__ == "__main__":

    cons = constraintsClass()

    constraintsList = {
        "1": {"type": "lessThan", "value": 23, "limit": 0, "weight": 1},
        "2": {"type": "greaterThan", "value": 0, "limit": 12, "weight": 1},
    }

    print(cons.constraints(constraintsList))
