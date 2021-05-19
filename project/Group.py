# A group of variables
class Group:
    def __init__(self, varnames, isMinimal=True):
        if isinstance(varnames, str):
            self.vars = set([varnames])

        elif isinstance(varnames, list):
            self.vars = set(varnames)

        if len(self.vars) > 0:
            v = next(iter(self.vars))
            self.type = v[:1]

        self.minimal = isMinimal

    def __eq__(self, other): 
        if not isinstance(other, Group):
            return NotImplemented
        return self.vars == other.vars

    # The set of variables in any minimalGroup should be unique
    def __hash__(self):
        s = "".join(sorted(list(self.vars)))
        return hash(s)

    # Union with another Group
    def union(self, L):
        self.vars = self.vars.union(L.vars)

    def __len__(self):
        return len(self.vars)

    def isLatent(self):
        v = next(iter(self.vars))
        if v[:1] == "L":
            return True
        else:
            return False

    def isMinimal(self):
        return self.minimal

    def takeOne(self):
        return next(iter(self.vars))

    def __str__(self):
        return ",".join(list(self.vars))

    def __repr__(self):
        return str(self)

    def isSubset(self, B):
        return self.vars <= B.vars

    def intersection(self, B):
        return self.vars.intersection(B.vars)

