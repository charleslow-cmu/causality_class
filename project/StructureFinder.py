from LatentGroups import LatentGroups
from pdb import set_trace
from misc import *
from math import floor, sqrt, isclose
import numpy as np
import scipy.stats
import operator
from functools import reduce
from RankTester import RankTester
from CCARankTester import CCARankTester
import IPython
from itertools import product
from statsmodels.multivariate.cancorr import CanCorr

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
        self.rankTester = CCARankTester(df, alpha=self.alpha)

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
        if setLength(Vs) > setLength(remainingVs):
            return None

        As = set()
        for V in Vs:
            As.update(self.l.pickAllMeasures(V))
        As = self.getMeasuredVarList(As)
                
        Bs = set()
        for V in remainingVs:
            Bs.update(self.l.pickAllMeasures(V))
        Bs = self.getMeasuredVarList(Bs)

        if not sample:
            return self.g.rankTest(As, Bs, k)

        else:
            return self.sampleRankTest(As, Bs, k)


    # Test a k subset of Variables to find Clusters
    def runStructuralRankTest(self, k=1, run=1, sample=False):
        vprint(f"Starting structuralRankTest k={k}...", self.verbose)
        anyFound = False

        # Terminate if not enough active variables
        if k > setLength(self.l.activeSet)/2:
            insufficientVars = True
            return (anyFound, insufficientVars)
        else:
            insufficientVars = False

        # Terminate if all the active variables already
        # belong to some Group
        discovered = set()
        for values in self.l.latentDict.values():
            discovered.update(values["children"])
        remaining = self.l.activeSet - discovered
        if len(remaining) == 0:
            insufficientVars = True
            return (anyFound, insufficientVars)

        allSubsets = generateSubset(self.l.activeSet, k+1)

        if allSubsets == [set()]:
            return (anyFound, True)

        for Vs in allSubsets:
            Vs = set(Vs)

            # Vs must not contain more than k elements from
            # any AtomicGroup with cardinality <= k-1
            testingExistingGroup = False
            for L, values in self.l.latentDict.items():
                LCardinality = len(L)
                existingGroup = values["children"] | values["subgroups"]
                commonElements = setIntersection(Vs, existingGroup)
                if len(commonElements) > LCardinality:
                    testingExistingGroup = True
                    break
            if testingExistingGroup:
                continue

            # Vs must not contain only variables that already
            # belong to a discovered group
            #alreadyDiscovered = set()
            #for existingGroup in self.l.invertedDict.keys():
            #    commonElements = Vs.intersection(existingGroup)
            #    alreadyDiscovered.update(commonElements)
            #if alreadyDiscovered == Vs:
            #    continue

            rankDeficient = self.structuralRankTest(Vs, k, run, sample)
            #vprint(f"Test {Vs}: Rank deficient {rankDeficient}", self.verbose)

            if rankDeficient is None:
                insufficientVars = True
                break

            # Test for the lowest rank that is deficient
            gap = 0
            if rankDeficient:
                rankDeficient2 = True
                while rankDeficient2 and k-gap > 0:
                    gap += 1
                    rankDeficient2 = self.structuralRankTest(Vs, k-gap, run, sample)
                if not rankDeficient2:
                    gap -= 1

            if rankDeficient:
                self.l.addToTempSet(Vs, k-gap)
                vprint(f"Found {k-gap}-cluster {Vs}.", self.verbose)
                anyFound = True

        return (anyFound, insufficientVars)


    # Algorithm to find the latent structure
    def findLatentStructure(self, verbose=True, sample=False):
        self.verbose = verbose
        self.l = LatentGroups(self.g.xvars)
        run = 1

        while True:
            k = 1
            foundList = []
            while True:
                anyFound, insufficientVars = \
                        self.runStructuralRankTest(k=k, run=run, sample=sample)

                self.l.mergeTempSets()
                self.l.confirmTempSets() 
                foundList.append(anyFound)

                k += 1
                if insufficientVars:
                    break

            print(f"{'='*10} End of Run {run} {'='*10}")
            print(f"Current State:")
            pprint(self.l.latentDict, self.verbose)
            run += 1

            # Update the activeSet
            self.l.updateActiveSet()
            print(f"Active Set: {self.l.activeSet}")
            print(f"{'='*30}")

            if not any(foundList):
                break

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

