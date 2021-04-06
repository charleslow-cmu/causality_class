# A minimalGroup is an atomic group of variables
# Any X is a minimalGroup by itself
# Any minimal Group with length > 1 must be latent
class MinimalGroup:
    def __init__(self, varnames):
        if isinstance(varnames, str):
            self.vars = set([varnames])

        elif isinstance(varnames, list):
            self.vars = set(varnames)

        if len(self.vars) > 0:
            v = next(iter(self.vars))
            self.type = v[:1]

    def __eq__(self, other): 
        if not isinstance(other, MinimalGroup):
            return NotImplemented
        return self.vars == other.vars

    # The set of variables in any minimalGroup should be unique
    def __hash__(self):
        s = "".join(sorted(list(self.vars)))
        return hash(s)

    # Union with another MinimalGroup
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

