import numpy as np
from numpy.linalg import matrix_rank
from math import sqrt, pow
from itertools import combinations
from copy import deepcopy
from pdb import set_trace

#!!
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
        if self.type == "L":
            return True
        else:
            return False

    def takeOne(self):
        return next(iter(self.vars))

    def __str__(self):
        return ",".join(list(self.vars))

    def __repr__(self):
        return str(self)

#####################################################################
# !!misc
#####################################################################

def setLength(varset):
    assert not isinstance(varset, str), "Cannot be string."
    n = 0
    for vset in varset:
        if isinstance(vset, MinimalGroup):
            n += len(vset)
        else:
            assert False, "Should be MinimalGroup."
    return n


# Take difference between sets of MinimalGroups
def setDifference(A, B):
    diff = A - B # first remove any common elements
    #newset = set()
    #while len(diff) > 0:
    #    a = diff.pop()
    #    newset.add(a)
    #    for b in B:
    #        if a <= b:
    #            newset.remove(a)
    return diff



# generateSubset: Generate set of MinimalGroups of variables 
# vset: set of MinimalGroup of variables
# Returns: list of sets of MinimalGroups, each set has setLength = k
def generateSubset(vset, k=2):

    def recursiveSearch(d, gap, currSubset=set()):
        thread = f"currSubset: {currSubset}, d: {d}, gap is {gap}"
        d = deepcopy(d)
        currSubset = deepcopy(currSubset)
        setlist = []

        # Terminate if empty list
        if len(d) == 0:
            return setlist

        # Pop MinimalGroups larger than current gap, we cannot take
        # any of them in
        maxDim = max(d)
        while maxDim > gap:
            d.pop(maxDim)
            maxDim = max(d)

        # Pop one MinimalGroup
        v = d[maxDim].pop()
        if len(d[maxDim]) == 0:
            d.pop(maxDim)

        # Branch to consider all cases
        # Continue current search without this element
        if len(d) > 0:
            setlist.extend(recursiveSearch(d, gap, currSubset))

        # Terminate branch if newGroup overlaps with currSubset
        if groupInLatentSet(v, currSubset):
            return setlist

        gap -= maxDim
        currSubset.add(v)

        # Continue search if gap not met
        if gap > 0 and len(d) > 0:
            setlist.extend(recursiveSearch(d, gap, currSubset))

        # End of search tree
        if gap == 0:
            setlist.append(currSubset)

        return setlist

    if k == 0:
        return [set()]

    # Create dictionary where key is dimension size and v is a list 
    # of frozensets of variables
    d = {}
    for v in vset:
        assert isinstance(v, MinimalGroup), "Should be MinimalGroup."
        n = len(v)
        d[n] = d.get(n, set()).union([v])

    # Run recursive search
    result = recursiveSearch(d, k)
    if len(result) == 0:
        return [set()]
    else:
        return result


# Check if new group of latent vars exists in a current
# list of latent vars
def groupInLatentSet(v: MinimalGroup, currSubset: set):
    return set([v]) in currSubset


# Print latent Dict
def pprint(d):

    def fsetToText(fset):
        l = [x for x in iter(fset)]
        return ",".join(l)

    for P, v in d.items():
        subgroups = v["subgroups"]
        Cs = v["children"]
        Ctext = ",".join([str(C) for C in Cs])
        Ptext = str(P)

        text = f"{Ptext} : {Ctext}"
        if len(subgroups) > 0:
            text += " | "
            for subgroup in subgroups:
                text += f"[{str(subgroup)}]"
        print(text)

#####################################################################
# !!graph
#####################################################################
class GaussianGraph:
    def __init__(self):
        self.vars = []
        self.xvars = []
        self.lvars = []
        self.L = None
        self.phi = []

    def random_variance(self):
        return np.random.normal()

    def random_coef(self):
        coef = 0
        while abs(coef) < 0.5:
            coef = np.random.uniform(low=-5, high=5)
        return coef

    def getParentIndex(self, parents):
        if isinstance(parents, str):
            return self.vars.index(parents)
        else:
            return [self.vars.index(parent) for parent in parents]

    def expandL(self, parents):
        n = len(self.vars)
        newL = np.zeros((n, n))
        newL[:(n-1), :(n-1)] = self.L  # Padded
        for parent in parents:
            pindex = self.getParentIndex(parent)
            newL[pindex, n-1] = self.random_coef()
        return newL


    def add_variable(self, name, parents=None):
        if isinstance(parents, str):
            parents = [parents]
        if parents is None:
            parents = []

        # Sanity checks
        if len(parents) > 0:
            variables = set(self.lvars) | set(self.xvars)
            valid_parents = len(set(parents)-variables) == 0
            assert valid_parents, "Parents not found in graph!"

        # Keep track of variables
        self.vars.append(name)
        if "X" in name:
            self.xvars.append(name)
        else:
            self.lvars.append(name)

        # Add exogeneous noise
        self.phi.append(self.random_variance())  # Add noise

        # Adding first variable
        if len(self.vars) == 1:
            self.L = np.zeros((1, 1))
        else:
            self.L = self.expandL(parents)

    def covariance(self):
        n = len(self.vars)
        lamb = np.identity(n) - self.L
        lamb_inv = np.linalg.inv(lamb)
        phi = np.diag(self.phi)
        return lamb_inv.T @ phi @ lamb_inv

    def subcovariance(self, rowvars, colvars):
        cov = self.covariance()
        rowindex = self.getParentIndex(rowvars)
        colindex = self.getParentIndex(colvars)
        return cov[np.ix_(rowindex, colindex)]

    # The main workhorse method
    def rankTest(self, A, B):
        A = sorted(A)
        B = sorted(B)
        cov = g.subcovariance(A, B)
        matrix_rank(cov)
        return matrix_rank(cov)

    def ranktests(self, v=2):
        results = {}
        xset = set(self.xvars)
        for A in combinations(self.xvars, v):
            B = sorted(xset.difference(A))
            A = sorted(A)
            cov = g.subcovariance(A, B)
            ranktest = matrix_rank(cov)
            results[ranktest] = results.get(ranktest, []) + [A]

        for r in results:
            print(f"{'='*20} Rank {r} {'='*20}")
            for A in results[r]:
                print(f"Test {A}: {r}")


#!!
# Class to store discovered latent groups
class LatentGroups():
    def __init__(self, X):
        self.i = 1
        self.X = set([MinimalGroup(x) for x in X])
        self.latentSet = set()
        self.activeSet = set([MinimalGroup(x) for x in X])
        self.latentDict = {}
        self.latentDictTemp = {}

    # Create a new Minimal Latent Group
    # Ls: Existing set of MinimalGroups of latent variables
    # As: Set of MinimalGroups to add as children
    # If Ls is empty, we simply create new latent variables
    def addToLatentSet(self, Ls, As):

        # Create new Latent Variables
        sizeL = setLength(As) - 1
        newLlist = []
        for _ in range(sizeL):
            newLlist.append(f"L{self.i}")
            self.i += 1
        newParents = MinimalGroup(newLlist)

        # Append existing Ls to newParents
        for L in Ls:
            newParents.union(L)

        # Create new entry
        self.latentDictTemp[newParents] = {
                "children": As,
                "subgroups": Ls
                }

        # Remove from Active Set
        self.activeSet = setDifference(self.activeSet, As)


    # Merge overlapping groups in dTemp
    def mergeTempGroups(self):
        if len(self.latentDictTemp) == 0:
            return

        # Ps for parent set, Cs for children set
        # Each p and c are frozensets of variables
        # Values of inv_list becomes a list of frozensets
        inv_list = {}
        for P, values in self.latentDictTemp.items():
            Cs = values["children"]
            for C in Cs:
                inv_list[C] = inv_list.get(C, set()).union([P])

        # Merge Ps with overlapping elements in the same group
        foundGroups = []
        for Ps in inv_list.values():
            if len(Ps) > 1:
                if len(foundGroups) == 0:
                    foundGroups.append(Ps)
                    continue
    
                for i, group in enumerate(foundGroups):
                    if len(Ps.intersection(group)) > 0:
                        foundGroups[i] = foundGroups[i].union(Ps)
                    else:
                        foundGroups.append(Ps)

        # foundGroups is now a list of parentSets with overlapping
        # children
        # Need to do a pairwise merge
        for group in foundGroups:
            mergeMap = self.getMergeMap(group)
            for oldp, p in mergeMap.items():
                values = self.latentDictTemp.pop(oldp)
                if not p in self.latentDictTemp:
                    self.latentDictTemp[p] = {}
                    self.latentDictTemp[p]["children"] = set()
                    self.latentDictTemp[p]["subgroups"] = set()
                self.latentDictTemp[p]["children"].update(values["children"])
                self.latentDictTemp[p]["subgroups"].update(values["subgroups"])

    # Return a 1-1 mapping from original group to new group
    # group: set of MinimalGroups of latent vars
    def getMergeMap(self, group):
        mergeMap = {}

        # Find lowest keys
        k = len(next(iter(group)))
        nums = []
        for A in group:
            A1, A2 = self.findMergeableVars(A)
            Anums = [int(x[1:]) for x in A2]
            nums.extend(Anums)
        lowestKeys = sorted(nums)[0:k]
        lowestKeys = [f"L{x}" for x in lowestKeys]
        print(lowestKeys)

        # Create new keys
        for A in group:
            A1, A2 = self.findMergeableVars(A)
            newkey = A1.union(lowestKeys)
            mergeMap[A] = MinimalGroup(list(newkey))
        return mergeMap

    # Given a set of MinimalGroups of latent vars, find
    # subgroups which are not minimal groups.
    def findMergeableVars(self, A):
        A1 = set() # Non-mergeable
        A2 = set() # Mergeable
        for P in A.vars:

            # Not mergeable if already confirmed in latentDict
            if P in self.latentDict:
                A1.update([P])
            else:
                A2.update([P])
        return A1, A2


    # Move elements in latentDictTemp to latentDict
    def confirmTempGroups(self):
        while len(self.latentDictTemp) > 0:
            p, values = self.latentDictTemp.popitem()
            self.latentDict[p] = values

            # Update V by grouping variables
            self.activeSet = setDifference(self.activeSet, 
                                    values["children"])
            self.latentSet.add(p)


    # Recursive search for one X per latent var in minimal group L
    def pickRepresentativeMeasures(self, L):
        assert isinstance(L, MinimalGroup), "L is not a MinimalGroup."
        A = set()
        values = self.latentDict[L]

        # Add one X per L from each subgroup
        if len(values["subgroups"]) > 0:
            for subL in values["subgroups"]:
                A.update(self.pickRepresentativeMeasures(subL))
        else:
            availableXs = values["children"]
            X = next(iter(availableXs))
            A.add(X)
        return A

    # As opposed to pickRepresentativeMeasures, pickAllMeasures 
    # recursively picks all measured variables that are in the subgroups
    # of the provided MinimalGroup.
    def pickAllMeasures(self, L):
        assert isinstance(L, MinimalGroup), "L is not a MinimalGroup."

        if not L.isLatent():
            return set([L])

        A = set()
        values = self.latentDict[L]

        if len(values["subgroups"]) > 0:
            for subL in values["subgroups"]:
                A.update(self.pickAllMeasures(subL))
        else:
            A.update(values["children"])
        return A


    # Return a set of all other measures that are not in the groups of 
    # the provided Vs.
    # Each v is a MinimalGroup
    def getAllOtherMeasuresFromGroups(self, Vs=set()):

        # Measures that we should not include
        Vmeasures = set()
        for V in Vs:
            Vmeasures.update(self.pickAllMeasures(V))
        return self.X - Vmeasures
            
    def getAllOtherMeasuresFromXs(self, Xs=set()):
        return self.X - Xs

# !!
class StructureFinder:

    def __init__(self, g):
        self.g = g     # graph object
        self.l = None  # LatentGroups object

    # Extract all measured vars in a list from a set of MinimalGroups
    # A: a set of MinimalGroups
    def getMeasuredVarList(self, A):
        measuredVars = []
        for a in A:
            if a.isLatent():
                assert False, "A is not a measured var set"
            if len(a) > 1:
                assert False, "Measured Vars should not have more than 1"
            measuredVars.append(a.takeOne())
        return measuredVars


    # Test if A forms a group by seeing if it drops rank against all
    # other vars
    # A and B are sets of MinimalGroups
    def structuralRankTest(self, As):
        Bs = self.getMeasuredVarList(self.l.getAllOtherMeasuresFromGroups(As))
        As = self.getMeasuredVarList(As)

        if len(Bs) < len(As):
            return -1
        else:
            return self.g.rankTest(As, Bs)


    def runStructuralRankTest(self):
        print("Starting structuralRankTest...")
        r = 1
        sufficientActiveVars = True
        while sufficientActiveVars:
            for As in generateSubset(self.l.activeSet, r+1):
                if not (As <= self.l.activeSet):
                    continue

                elif setLength(As) < r+1:
                    sufficientActiveVars = False
                    break

                rankResult = self.structuralRankTest(As)
                if rankResult < 0:
                    sufficientActiveVars = False
                    break

                if rankResult == r:
                    self.l.addToLatentSet(set(), As)
                    print(f"Found cluster {As}.")

            r += 1
            self.l.mergeTempGroups()
        self.l.confirmTempGroups()
        pprint(self.l.latentDict)


    # Test if V is children of Ls
    # V: a MinimalGroup
    def parentSetTest(self, Ls, V):
        r = setLength(Ls)
        k = len(V)
        As = set()
        for L in Ls:
            As.update(self.l.pickRepresentativeMeasures(L))

        if V.isLatent():
            As.update(self.l.pickRepresentativeMeasures(V))
        else:
            assert k == 1, "Measured Var of length > 1."
            As.add(V)

        assert setLength(As) == r+k, "Wrong length"

        # Construct Bs set
        Bs = self.getMeasuredVarList(self.l.getAllOtherMeasuresFromXs(As))
        As = self.getMeasuredVarList(As)

        if len(Bs) < len(As):
            return -1
        else:
            return self.g.rankTest(As, Bs)

    def runParentSetTest(self):
        print("Starting parentSetTest...")
        for V in self.l.activeSet:
            k = 1 # Number of MinimalGroup Latents to test as Parent
            sufficientVars = True
            while sufficientVars:
                for Ls in combinations(self.l.latentSet, k):
                    Ls = set(Ls)

                    if len(Ls) == 0:
                        sufficientVars = False
                        break

                    rankResult = self.parentSetTest(Ls, V)
                    if rankResult < 0:
                        sufficientVars = False
                        break

                    if rankResult == setLength(Ls):
                        print(f"Found that {V} belongs to {Ls}.")
                        self.l.addToLatentSet(Ls, set([V]))

                self.l.mergeTempGroups()
                k += 1
                if k >= len(self.l.latentSet):
                    sufficientVars = False
                    break

        self.l.confirmTempGroups()
        pprint(self.l.latentDict)

    # Test if A forms a new group with L by adding a latent var
    def adjacentParentTest(self, L, A):
        pass

    # Test if Ls d-separates As and Bs
    def relationshipTest(self, A, B, L):
        pass

    # Algorithm to find the latent structure
    def findLatentStructure(self):
        self.l = LatentGroups(self.g.xvars)
        self.runStructuralRankTest()
        self.runParentSetTest()


if __name__ == "__main__":
    g = GaussianGraph()
    g.add_variable("L3", None)
    g.add_variable("L1", "L3")
    g.add_variable("L2", "L1")
    g.add_variable("X6", "L3")
    g.add_variable("X7", "L3")
    g.add_variable("X8", ["L2", "L3"])
    g.add_variable("X1", "L1")
    g.add_variable("X2", "L1")
    g.add_variable("X3", ["L1", "L2"])
    g.add_variable("X4", "L2")
    g.add_variable("X5", "L2")

    model = StructureFinder(g)
    model.findLatentStructure()


