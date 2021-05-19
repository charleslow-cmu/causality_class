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
import pydot

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

    # Extract all measured vars in a list from a set of Groups
    # A: a set of Groups
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
    # A and B are sets of Groups
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
                for subgroup in values["subgroups"]:
                    existingGroup.update(self.l.latentDict[subgroup]["children"])
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

            try:
                rankDeficient = self.structuralRankTest(Vs, k, run, sample)
            except:
                set_trace()
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
                #vprint(f"Found {k-gap}-cluster {Vs}.", self.verbose)
                anyFound = True

        return (anyFound, insufficientVars)


    # Run the refining step
    def refineClusters(self):
        print(f"{'='*10} Refining Clusters! {'='*10}")
        junctions = self.l.findJunctions()
        print(junctions)
        self.refineBranch(Group("root"), junctions)


    def refineBranch(self, junction, junctions):
        print(f"{'='*10} Refining Cluster {junction}! {'='*10}")

        # Kick off refining from Root
        if junction == Group("root"):
            subgroups = deepcopy(self.l.activeSet)
        else:
            values = self.l.latentDict[junction]
            subgroups = values["subgroups"]

        # The parent at the junction is our starting point
        if not junction == Group("root"):
            self.l.activeSet = set([junction])

        # Find out where to stop
        stopJunctions = set()
        if junction in junctions:
            stopJunctions = junctions[junction]

        # Dissolve all the down stream branches
        # until the stopJunctions
        for L in subgroups:
            self.l.dissolveRecursive(L, stopJunctions)

        # Rerun search procedure
        self.findLatentStructure()
        set_trace()

        # Then go down the tree
        if junction in junctions:
            for j in junctions[junction]:
                self.refineBranch(j, junctions)


    # Algorithm to find the latent structure
    def findLatentStructure(self, maxk=3, verbose=True, sample=False):
        self.verbose = verbose
        if self.l is None:
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

                if k > maxk:
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

    def printGraph(self, outpath):
        G = getGraph(self.l)
        G.toDot("example.dot")
        graphs = pydot.graph_from_dot_file('example.dot')
        graphs[0].set_size('"8,8!"')
        graphs[0].write_png(outpath)
    
