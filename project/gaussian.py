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

    def isSubset(self, B):
        return self.vars <= B.vars


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
def setDifference(As, Bs):
    diff = As - Bs # first remove any common elements
    newset = set()
    while len(diff) > 0:
        A = diff.pop()
        newset.add(A)
        for B in Bs:
            if A.isSubset(B):
                newset.remove(A)
    return newset


def vprint(s, verbose=False):
    if verbose:
        print(s)

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
def groupInLatentSet(V: MinimalGroup, currSubset: set):
    for group in currSubset:
        if len(V.vars.intersection(group.vars)) > 0:
            return True
    return False

# Print latent Dict
def pprint(d, verbose=False):

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
        if verbose:
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
        cov = self.subcovariance(A, B)
        matrix_rank(cov)
        return matrix_rank(cov)


    def scenario1(self):
        self.add_variable("L3", None)
        self.add_variable("L1", "L3")
        self.add_variable("L2", "L1")
        self.add_variable("X6", "L3")
        self.add_variable("X7", "L3")
        self.add_variable("X8", ["L2", "L3"])
        self.add_variable("X1", "L1")
        self.add_variable("X2", "L1")
        self.add_variable("X3", ["L1", "L2"])
        self.add_variable("X4", "L2")
        self.add_variable("X5", "L2")


    def scenario2(self):
        self.add_variable("L1", None)
        self.add_variable("L2", "L1")
        self.add_variable("L3", "L1")
        self.add_variable("L4", "L1")
        self.add_variable("L5", "L1")
        self.add_variable("L6", "L2")
        self.add_variable("L7", ["L2", "L3"])
        self.add_variable("L8", ["L2", "L3"])
        self.add_variable("L9", ["L2", "L3"])
        self.add_variable("L10", "L3")
        self.add_variable("L11", "L4")
        self.add_variable("L12", ["L4", "L5"])
        self.add_variable("L13", ["L4", "L5"])
        self.add_variable("L14", ["L4", "L5"])
        self.add_variable("L15", "L5")
        self.add_variable("X1", "L6")
        self.add_variable("X2", "L6")
        self.add_variable("X3", "L7")
        self.add_variable("X4", ["L7", "L8"])
        self.add_variable("X5", ["L7", "L8"])
        self.add_variable("X6", "L9")
        self.add_variable("X7", "L9")
        self.add_variable("X8", ["L9", "L10"])
        self.add_variable("X9", "L10")
        self.add_variable("X10", "L11")
        self.add_variable("X11", "L11")
        self.add_variable("X12", "L12")
        self.add_variable("X13", ["L12", "L13"])
        self.add_variable("X14", ["L12", "L13"])
        self.add_variable("X15", "L14")
        self.add_variable("X16", "L14")
        self.add_variable("X17", ["L14", "L15"])
        self.add_variable("X18", "L15")

    def scenario3(self):
        self.add_variable("L1", None)
        self.add_variable("L2", "L1")
        self.add_variable("L3", "L1")
        self.add_variable("L5", "L2")
        self.add_variable("L6", ["L2","L5"])
        self.add_variable("L7", ["L2","L6"])

        self.add_variable("X1", "L1")
        self.add_variable("X6", "L1")
        self.add_variable("X2", "L2")
        self.add_variable("X3", "L3")
        self.add_variable("X4", "L3")
        self.add_variable("X5", "L3")
        self.add_variable("X8", "L5")
        self.add_variable("X9", "L5")
        self.add_variable("X10", "L6")
        self.add_variable("X11", "L6")
        self.add_variable("X12", ["L6","L7"])
        self.add_variable("X13", "L7")


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
    def addToLatentSet(self, Ls, As, rankDeficiency=1):

        # Create new Latent Variables
        sizeL = setLength(As) - rankDeficiency
        newLlist = []
        for _ in range(sizeL):
            newLlist.append(f"L{self.i}")
            self.i += 1
        newParents = MinimalGroup(newLlist)

        # Append existing Ls to newParents
        for L in Ls:
            newParents.union(L)

        # Create subgroups
        subgroups = Ls
        for A in As:
            if A.isLatent():
                subgroups.add(A)

        # Create new entry
        self.latentDictTemp[newParents] = {
                "children": As,
                "subgroups": subgroups
                }

        # Remove from Active Set
        # self.activeSet = setDifference(self.activeSet, As)


    # Merge overlapping groups in dTemp
    def mergeTempGroups(self, run=1):
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
    def confirmTempGroups(self, run=1):
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

        if not L.isLatent():
            return set([L])

        A = set()
        values = self.latentDict[L]
        n = len(L)

        # Add one X per L from each subgroup
        if len(values["subgroups"]) > 0:
            for subL in values["subgroups"]:
                A.update(self.pickRepresentativeMeasures(subL))

        # Add remaining from own children
        n = len(L) - setLength(values["subgroups"])
        availableXs = values["children"]
        for i, X in enumerate(iter(availableXs)):
            if i >= n:
                break
            if not X.isLatent():
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
        
        for C in values["children"]:
            if not C.isLatent():
                A.add(C)
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
        self.verbose = False

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
    def structuralRankTest(self, Vs):
        As = set()
        for V in Vs:
            As.update(self.l.pickRepresentativeMeasures(V))
        Bs = self.getMeasuredVarList(self.l.getAllOtherMeasuresFromGroups(Vs))
        As = self.getMeasuredVarList(As)

        if len(Bs) < len(As):
            return -1
        else:
            return self.g.rankTest(As, Bs)


    def runStructuralRankTest(self, k=2, run=1):
        vprint("Starting structuralRankTest...", self.verbose)
        anyFound = False
        sufficientActiveVars = True
        while sufficientActiveVars:
            for As in generateSubset(self.l.activeSet, k):

                if setLength(As) < k:
                    sufficientActiveVars = False
                    break

                rankResult = self.structuralRankTest(As)
                if rankResult < 0:
                    sufficientActiveVars = False
                    break

                maxRank = rankResult
                for AsPrime in generateSubset(As, k-1):
                    rankResultPrime = self.structuralRankTest(AsPrime)
                    if rankResultPrime > maxRank:
                        maxRank = rankResultPrime

                if maxRank == k-1:
                    self.l.addToLatentSet(set(), As)
                    vprint(f"Found cluster {As}.", self.verbose)
                    anyFound = True

            self.l.mergeTempGroups(run)
            self.l.confirmTempGroups(run)
        pprint(self.l.latentDict, self.verbose)
        return anyFound


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

        #try:
        #    assert setLength(As) == r+k, "Wrong length"
        #except:
        #    set_trace()

        # Construct Bs set
        Bs = self.getMeasuredVarList(self.l.getAllOtherMeasuresFromXs(As))
        As = self.getMeasuredVarList(As)

        if len(Bs) < len(As):
            return -1
        else:
            return self.g.rankTest(As, Bs)

    def runParentSetTest(self, run=1):
        vprint("Starting parentSetTest...", self.verbose)
        anyFound = False
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
                        vprint(f"Found that {V} belongs to {Ls}.", self.verbose)
                        self.l.addToLatentSet(Ls, set([V]))
                        anyFound = True

                self.l.mergeTempGroups()
                k += 1
                if k >= len(self.l.latentSet):
                    sufficientVars = False
                    break

        self.l.confirmTempGroups()
        pprint(self.l.latentDict, self.verbose)
        return anyFound

    # Test if As forms a new group with Ls by adding a latent var
    def adjacentParentTest(self, Ls, Vs):
        r = setLength(Ls)
        k = setLength(Vs)
        assert k >= 2, "Length of Vs too short"
        As = set()

        # Assemble representatives for Ls
        for L in Ls:
            As.update(self.l.pickRepresentativeMeasures(L))

        # Assemble representatives for Vs
        for V in Vs:
            if V.isLatent():
                As.update(self.l.pickRepresentativeMeasures(V))
            else:
                As.add(V)

        # Assemble control set Bs
        # We don't want to test against anything in the Vs groups
        # Also don't want any measured vars already in As
        Bs = self.l.getAllOtherMeasuresFromGroups(Vs)
        Bs = Bs.intersection(self.l.getAllOtherMeasuresFromXs(As))

        # Construct Bs set
        Bs = self.getMeasuredVarList(Bs)
        As = self.getMeasuredVarList(As)

        if len(Bs) < len(As):
            return -1
        else:
            return self.g.rankTest(As, Bs)


    # k: Number of variable to test
    # r: Number of MinimalGroup Latents to test for as Co-Parent
    def runAdjacentParentTest(self, k=2, run=1):
        vprint("Starting adjacentParentTest...", self.verbose)
        r = 1
        sufficientVars = True
        anyFound = False

        while sufficientVars:
            for As in generateSubset(self.l.activeSet, r):
                if setLength(As) < r:
                    sufficientVars = False
                    break

                for Ls in combinations(self.l.latentSet, k):
                    Ls = set(Ls)

                    if len(Ls) == 0:
                        sufficientVars = False
                        break

                    rankResult = self.adjacentParentTest(Ls, As)
                    if rankResult < 0:
                        sufficientVars = False
                        break

                    rankDeficiency = setLength(Ls) + setLength(As) - rankResult
                    if rankDeficiency > 0:
                        vprint(f"Found that {As} belongs to {Ls} with co-parents.",
                                self.verbose)
                        self.l.addToLatentSet(Ls, As, rankDeficiency)
                        anyFound = True

            self.l.mergeTempGroups()
            self.l.confirmTempGroups()
            r += 1

            if r >= k:
                break

            if k >= len(self.l.latentSet):
                break

        pprint(self.l.latentDict, self.verbose)
        return anyFound

    # Test if Ls d-separates As and Bs
    def relationshipTest(self, A, B, L):
        pass

    # Algorithm to find the latent structure
    def findLatentStructure(self, verbose=True):
        self.verbose = verbose
        self.l = LatentGroups(self.g.xvars)
        run = 1
        while len(self.l.activeSet) > 0:
            k = 2
            while True:
                test2 = self.runParentSetTest(run)
                test1 = self.runStructuralRankTest(k, run)
                test3 = self.runAdjacentParentTest(k, run)
                k += 1
                if not any([test1, test2, test3]):
                    break

            # Move latentSet back into activeSet
            self.l.activeSet.update(self.l.latentSet)
            self.l.latentSet = set()
            run += 1

    # Return a node and edge set for plotting with networkx
    def reportDiscoveredGraph(self):
        nodes = set()
        edges = set()
        discoveredGraph = deepcopy(self.l.latentDict)
        while len(discoveredGraph) > 0:
            parents, values = discoveredGraph.popitem()
            childrenSet = values["children"]
            for parent in parents.vars:
                if not parent in nodes:
                    nodes.add(parent)
                for childGroup in childrenSet:
                    childlatent = childGroup.isLatent()
                    for child in childGroup.vars:
                        if not child in nodes:
                            nodes.add(child)
                        edges.add((parent, child))
        return (nodes, edges)


if __name__ == "__main__":
    g = GaussianGraph()
    g.scenario2()
    model = StructureFinder(g)
    model.findLatentStructure(verbose=True)


