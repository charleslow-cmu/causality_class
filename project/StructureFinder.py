from LatentGroups import LatentGroups
from pdb import set_trace
from misc import *
from math import floor, sqrt
import numpy as np
import scipy.stats
import operator
from functools import reduce
from RankTester import RankTester
import IPython

class StructureFinder:

    def __init__(self, g, alpha=0.05):
        self.g = g     # graph object
        self.l = None  # LatentGroups object
        self.verbose = False
        self.alpha = alpha # Critical value for testing
        self.covList = []

    # Sample of generated data from g
    def addSample(self, df):
        self.df = df
        self.rankTester = RankTester(df, trials=1000, normal=True, 
                                    alpha=self.alpha)

    # Bootstrap sample our data and calculate covariance matrix
    # k times
    def prepareBootstrapCovariances(self, k=10):
        for _ in range(k):
            cov = bootStrapCovariance(self.df)
            self.covList.append(cov)

    # Test null hypothesis that rank(subcovariance[A,B]) <= rk
    # A, B: Columns for testing
    # Returns True if we fail to reject the null
    def sampleRankTest(self, A, B, rk):
        cols = list(self.df.columns)
        I = [cols.index(a) for a in A]
        J = [cols.index(b) for b in B]
        test = self.rankTester.test(I, J, r=rk)
        return not test

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


    # Test if A forms a group by seeing if rank(subcov[A,B]) <= k-1
    # A and B are sets of MinimalGroups
    # A: Take all measures of Vs
    # B: Take all measures not in Vs
    # Returns True if rank deficient, False otherwise
    def structuralRankTest(self, Vs, k, run=1, sample=False):

        remainingVs = setDifference(self.l.activeSet, Vs)

        if setLength(Vs) > setLength(remainingVs) + setLength(self.l.latentSet):
            return None

        As = set()
        for V in Vs:
            As.update(self.l.pickAllMeasures(V))
        As = self.getMeasuredVarList(As)
                
        Bs = set()
        for V in remainingVs:
            Bs.update(self.l.pickAllMeasures(V))

        for V in self.l.latentSet:
            Bs.update(self.l.pickAllMeasures(V))
        Bs = self.getMeasuredVarList(Bs)

        if not sample:
            return self.g.rankTest(As, Bs, k-1)

        else:
            return self.sampleRankTest(As, Bs, k-1)


    # Test a k subset of Variables to find Clusters
    def runStructuralRankTest(self, k=2, run=1, sample=False):
        vprint(f"Starting structuralRankTest k={k}...", self.verbose)
        anyFound = False

        # Terminate if not enough active variables
        if k > setLength(self.l.activeSet):
            insufficientVars = True
            return (anyFound, insufficientVars)
        else:
            insufficientVars = False

        for Vs in generateSubset(self.l.activeSet, k):
            Vs = set(Vs)

            if not Vs <= self.l.activeSet:
                continue

            if setLength(Vs) < k:
                break

            rankDeficient = self.structuralRankTest(Vs, k, run, sample)
            #vprint(f"Test {Vs}: Rank deficient {rankDeficient}", self.verbose)

            if rankDeficient is None:
                insufficientVars = True
                break

            if rankDeficient:
                self.l.addToLatentSet(Vs, k-1)
                vprint(f"Found cluster {Vs}.", self.verbose)
                anyFound = True

        return (anyFound, insufficientVars)


    # Test if V is children of Ls
    # V: a MinimalGroup
    def parentSetTest(self, Ls, V, run=1, sample=False):
        #print(f"Testing for {V} against {Ls}")

        k = setLength(Ls)
        remainingLs = self.l.latentSet - Ls
        remainingVs = self.l.activeSet - set([V])

        As = set()
        for L in Ls:
            As.update(self.l.pickRepresentativeMeasures(L))
        try:
            assert len(As) == setLength(Ls), "As must be same length as Ls"
        except:
            set_trace()

        if V.isLatent():
            As.update(self.l.pickAllMeasures(V))
        else:
            As.add(V)

        Bs = set()
        for L in Ls:
            Bs.update(self.l.pickRepresentativeMeasures(L, usedXs=As))

        # Terminate if we cannot make a big enough control set
        if setLength(Ls) + 1 > setLength(Bs) + setLength(remainingLs) +\
                setLength(remainingVs):
            return None

        for L in remainingLs:
            Bs.update(self.l.pickAllMeasures(L))
        for V in remainingVs:
            Bs.update(self.l.pickAllMeasures(V))

        Bs = self.getMeasuredVarList(Bs)
        As = self.getMeasuredVarList(As)

        if not sample:
            return self.g.rankTest(As, Bs, rk=k)
        else:
            return self.sampleRankTest(As, Bs, rk=k)


    # Too many false negatives for parentSetTest
    def runParentSetTest(self, k=1, run=1, sample=False):
        vprint(f"Starting parentSetTest k={k}...", self.verbose)
        anyFound = False

        # Terminate if not enough active variables
        if k > len(self.l.latentSet):
            insufficientVars = True
            return (anyFound, insufficientVars)
        else:
            insufficientVars = False

        for Ls in combinations(self.l.latentSet, k):
            Ls = set(Ls)
            for V in self.l.activeSet:

                if not V in self.l.activeSet:
                    continue

                rankDeficient = self.parentSetTest(Ls, V, run, sample)

                if rankDeficient is None:
                    break

                if rankDeficient:
                    vprint(f"Found parents {Ls} for {V}.", self.verbose)
                    self.l.addStrayChild(Ls, V)
                    anyFound = True

        return (anyFound, insufficientVars)


    # Algorithm to find the latent structure
    def findLatentStructure(self, verbose=True, sample=False):
        self.verbose = verbose
        self.l = LatentGroups(self.g.xvars)
        run = 1

        while True:
            k = 2
            testlist = []
            while True:
                test1 = self.runStructuralRankTest(k=k, run=run, sample=sample)
                testlist.append(test1[0])
                k += 1

                if test1[1]:
                    break

            k = 1
            while True:
                test2 = self.runParentSetTest(k, run, sample)
                testlist.append(test2[0])
                k += 1

                if test2[1]:
                    break

            if not any(testlist):
                break

            # self.l.confirmTempGroups()
            pprint(self.l.latentDict, self.verbose)

            # Move latentSet back into activeSet
            self.l.activeSet.update(self.l.latentSet)
            self.l.latentSet = set()

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

