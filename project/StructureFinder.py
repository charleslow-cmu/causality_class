from LatentGroups import LatentGroups
from pdb import set_trace
from misc import *
from math import floor

class StructureFinder:

    def __init__(self, g, alpha=0.05):
        self.g = g     # graph object
        self.l = None  # LatentGroups object
        self.verbose = False
        self.alpha = alpha # Critical value for testing

    # Sample of generated data from g
    def addSample(self, df):
        self.df = df
        n = df.shape[0]
        data = df.to_numpy()
        self.S = 1/(n-1) * data.T @ data  # Sample Covariance

    # df: pandas dataframe
    # m: Number of independent tests
    # Returns pValue
    def sampleRankTest(self, A, B):
        cols = list(self.df.columns)
        I = [cols.index(a) for a in A]
        J = [cols.index(b) for b in B]
        n = self.df.shape[0]
        pValue = determinantTest(self.S, I, J, n)
        return pValue


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
    # Returns True if rank deficient, False otherwise
    def structuralRankTest(self, Vs, run=1, sample=False):

        remainingVs = setDifference(self.l.activeSet, Vs)
        if len(Vs) > len(remainingVs):
            return None

        As = set()
        k = len(Vs) # We want a k by k test
        for V in Vs:
            As.update(self.l.pickRepresentativeMeasures(V))
        As = self.getMeasuredVarList(As)
                
        Bs = self.getMeasuredVarList(self.l.getAllOtherMeasuresFromGroups(Vs))
        if len(Bs) < len(As):
            return None

        if not sample:
            return self.g.rankTest(As, Bs)

        # Sample Test
        # Must have rank drop in every test to be deficient
        # Return True if rank is deficient (fail to reject null)
        # This step is very costly - need some way to control
        plist = []
        for B in combinations(Bs, len(As)):
            p = self.sampleRankTest(As, B)
            plist.append(p)
        plist.sort()
        test = bonferroniHolmTest(plist, self.alpha)
        return test


    def runStructuralRankTest(self, run=1, sample=False):
        vprint("Starting structuralRankTest...", self.verbose)
        r = 1
        anyFound = False
        sufficientActiveVars = True
        while sufficientActiveVars:
            for As in generateSubset(self.l.activeSet, r+1):

                if setLength(As) < r+1:
                    sufficientActiveVars = False
                    break

                rankDeficient = self.structuralRankTest(As, run, sample)
                if rankDeficient is None:
                    sufficientActiveVars = False
                    break

                if rankDeficient:
                    self.l.addToLatentSet(set(), As)
                    vprint(f"Found cluster {As}.", self.verbose)
                    anyFound = True

            r += 1
            self.l.mergeTempGroups(run)
            self.l.confirmTempGroups(run)
        pprint(self.l.latentDict, self.verbose)
        return anyFound


    # Test if V is children of Ls
    # V: a MinimalGroup
    def parentSetTest(self, Ls, V, sample=False):
        #print(f"Testing for {V} against {Ls}")

        # Must not use all latent variables
        # Leave some extra for use in control set Bs
        totalLatents = len(self.l.latentSet)
        if len(Ls) >= totalLatents:
            return None

        k = len(V)
        As = set()
        for L in Ls:
            As.update(self.l.pickRepresentativeMeasures(L))

        if V.isLatent():
            As.update(self.l.pickRepresentativeMeasures(V))
        else:
            assert k == 1, "Measured Var of length > 1."
            As.add(V)

        # Construct Bs set
        Bs = set()
        for L in Ls:
            Bs.update(self.l.pickRepresentativeMeasures(L, usedXs=As))

        # Cs is all remaining variables
        Cs = self.getMeasuredVarList(self.l.getAllOtherMeasuresFromXs(
                    As.union(Bs)))
        Bs = self.getMeasuredVarList(Bs)
        As = self.getMeasuredVarList(As)

        if len(Bs) < len(As)-1 or len(Cs) == 0:
            return None

        if not sample:
            return self.g.rankTest(As, Bs + Cs)

        # Sample Test
        # Must have rank drop in every test to be deficient
        # Return True if rank is deficient (fail to reject null)
        plist = []
        for C in Cs:
            p = self.sampleRankTest(As, Bs + [C])
            plist.append(p)
        plist.sort()
        test = bonferroniHolmTest(plist, self.alpha)
        return test


    # Find smallest set that is rankDeficient
    def runParentSetTestRecursive(self, Ls, V, sample=False):
        vprint(f"Running! {Ls} vs {V}", self.verbose)
        smallestLs = deepcopy(Ls)

        if len(Ls) <= 1:
            vprint("set is too small", self.verbose)
            return set()

        rankDeficient = self.parentSetTest(Ls, V, sample)
        if rankDeficient is None:
            vprint("Result is None", self.verbose)
            discovered = []
            for L in Ls:
                subLs = Ls - set([L])
                newLs = self.runParentSetTestRecursive(subLs, V, sample)
                discovered.append(newLs)

            smallestLs = set()
            for L in discovered:
                if len(L) > 0 and len(smallestLs) == 0:
                    smallestLs = L
                elif len(L) > 0 and len(L) < len(smallestLs):
                    smallestLs = L
            return smallestLs

        if not rankDeficient:
            vprint("This is full rank", self.verbose)
            return set()

        # If rank is deficient
        vprint("Rank is deficient!", self.verbose)
        smallestLs = deepcopy(Ls)
        for L in Ls:
            subLs = Ls - set([L])
            newLs = self.runParentSetTestRecursive(subLs, V, sample)
            if len(newLs) < len(smallestLs) and len(newLs) > 0:
                smallestLs = newLs
        return smallestLs



    # Too many false negatives for parentSetTest
    def runParentSetTest(self, run=1, sample=False):
        vprint("Starting parentSetTest...", self.verbose)
        anyFound = False
        for V in self.l.activeSet:
            Ls = deepcopy(self.l.latentSet)
            smallestLs = self.runParentSetTestRecursive(Ls, V, sample)
            if len(smallestLs) > 0:
                self.l.addToLatentSet(smallestLs, set([V]))
                vprint(f"Found that {V} belongs to {smallestLs}.", self.verbose)
                anyFound = True
        self.l.confirmTempGroups()
        pprint(self.l.latentDict, self.verbose)
        return anyFound



    # Test if Vs forms a new group with Ls by adding a latent var
    def adjacentParentTest(self, Ls, Vs, sample=False):
        k = setLength(Vs)
        assert k >= 2, "Length of Vs too short"
        As = set()

        # Must not use all latent variables
        # Leave some extra for use in control set Bs
        totalLatents = len(self.l.latentSet)
        if len(Ls) + len(Vs) - 1 > totalLatents:
            return None

        # Assemble representatives for Ls
        for L in Ls:
            As.update(self.l.pickRepresentativeMeasures(L))

        # Assemble representatives for Vs
        for V in Vs:
            if V.isLatent():
                As.update(self.l.pickRepresentativeMeasures(V))
            else:
                As.add(V)

        # Assemble control set Bs, made up of one measure per Latent in Ls
        Bs = set()
        for L in Ls:
            Bs.update(self.l.pickRepresentativeMeasures(L, usedXs=As))

        # Cs is remaining variables to permute over:
        # 1. Other latent variables available
        # 2. All other active vars (excluding Vs)
        Cs = set()
        remainingLs = self.l.latentSet - Ls
        for L in remainingLs.union(self.l.activeSet - Vs):
            Cs.update(self.l.pickRepresentativeMeasures(L))

        Bs = self.getMeasuredVarList(Bs)
        As = self.getMeasuredVarList(As)
        Cs = self.getMeasuredVarList(Cs)

        # If not enough remaining variables
        if len(Bs) < len(As) - len(Vs) or len(Cs) < len(Vs):
            return None

        if not sample:
            return self.g.rankTest(As, Bs + Cs)

        # Sample Test
        # Must have rank drop in every test to be deficient
        # Return True if rank is deficient (fail to reject null)
        plist = []
        for C in combinations(Cs, k):
            p = self.sampleRankTest(As, Bs + list(C))
            plist.append(p)
        plist.sort()
        test = bonferroniHolmTest(plist, self.alpha)
        return test


    def runAdjacentParentTestRecursive(self, Ls, Vs, sample):
        #print(f"Running! {Ls} vs {Vs}")
        smallestLs = set()

        if len(Ls) == 0:
            #print("Set is too small")
            return smallestLs

        rankDeficient = self.adjacentParentTest(Ls, Vs, sample)

        # Find the first deficient set
        if rankDeficient is None:
            #print("Result is None")
            for L in Ls:
                subLs = Ls - set([L])
                newLs = self.runAdjacentParentTestRecursive(subLs, Vs, sample)
                if len(newLs) > 0:
                    break
            return newLs

        if rankDeficient == False:
            #print("This is full rank")
            return smallestLs

        if rankDeficient:
            #print("Rank is deficient")
            smallestLs = deepcopy(Ls) # this set is deficient
            removeableLs = set()
            for L in Ls:
                subLs = Ls - set([L])
                #print(f"Testing remove {L} for {Vs}")
                rankDeficient = self.adjacentParentTest(subLs, Vs, sample)

                # This L is redundant for causing rankDeficiency
                if rankDeficient:
                    removeableLs.add(L)
            smallestLs = Ls - removeableLs
        return smallestLs


    # k: Number of variable to test
    # r: Number of MinimalGroup Latents to test for as Co-Parent
    def runAdjacentParentTest(self, run=1, sample=False):
        vprint("Starting adjacentParentTest...", self.verbose)
        k = 2
        anyFound = False

        # Maximum cardinality of activeSet to test
        maxActiveVars = floor(len(self.l.activeSet) / 2)

        # Start with all latent vars
        # Start with 2 xvars
        while k <= maxActiveVars:
            for Vs in generateSubset(self.l.activeSet, k):
                Ls = deepcopy(self.l.latentSet)
                smallestLs = self.runAdjacentParentTestRecursive(Ls, Vs, sample)
                if len(smallestLs) > 0:
                    self.l.addToLatentSet(smallestLs, Vs)
                    vprint(f"Found that {Vs} belongs to {smallestLs} with co-parents.",
                                    self.verbose)
                    anyFound = True
            k += 1
            self.l.mergeTempGroups()
        self.l.confirmTempGroups()
        pprint(self.l.latentDict, self.verbose)
        return anyFound

    # Test if Ls d-separates As and Bs
    def relationshipTest(self, A, B, L):
        pass

    # Algorithm to find the latent structure
    def findLatentStructure(self, verbose=True, sample=False):
        self.verbose = verbose
        self.l = LatentGroups(self.g.xvars)
        run = 1
        while len(self.l.activeSet) > 0:
            test1 = self.runStructuralRankTest(run, sample)
            test2 = self.runParentSetTest(run, sample)
            test3 = self.runAdjacentParentTest(run, sample)
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
                    for child in childGroup.vars:
                        if not child in nodes:
                            nodes.add(child)
                        edges.add((parent, child))
        return (nodes, edges)
