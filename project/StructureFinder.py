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

    def cca(self, Vs):
        remainingVs = setDifference(self.l.activeSet, Vs)
        cols = list(self.df.columns)

        As = set()
        for V in Vs:
            As.update(self.l.pickAllMeasures(V))
        As = self.getMeasuredVarList(As)
                
        Bs = set()
        for V in remainingVs:
            Bs.update(self.l.pickAllMeasures(V))
        Bs = self.getMeasuredVarList(Bs)

        X = self.df.loc[:, As]
        Y = self.df.loc[:, Bs]
        print(f"CCA for {As} vs {Bs}")
        return CanCorr(X, Y)

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
        for Vs in allSubsets:
            Vs = set(Vs)

            # Vs must not contain more than k elements from
            # any AtomicGroup with cardinality <= k-1
            testingExistingGroup = False
            for existingGroup in self.l.invertedDict.keys():
                commonElements = setIntersection(Vs, existingGroup)
                existingGroupCardinality = self.l.invertedDict[existingGroup]
                if len(commonElements) > existingGroupCardinality:
                    testingExistingGroup = True
                    break
            if testingExistingGroup:
                continue

            # Vs must not contain only variables that already
            # belong to a discovered group
            alreadyDiscovered = set()
            for existingGroup in self.l.invertedDict.keys():
                commonElements = Vs.intersection(existingGroup)
                alreadyDiscovered.update(commonElements)
            if alreadyDiscovered == Vs:
                continue

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

            # How to remove variables that are not important to 
            # the rank, using CCA?
            if rankDeficient:
                cca = self.cca(Vs)
                if k-gap == 2:
                    IPython.embed(); exit(1)
                test = self.verifyCluster(Vs, k-gap)
                if test:
                    self.l.addToTempSet(Vs, k-gap)
                    vprint(f"Found {k-gap}-cluster {Vs}.", self.verbose)
                    anyFound = True

        return (anyFound, insufficientVars)


    # Used Generalised Information Distance to Verify if a Cluster is true
    def verifyCluster(self, Vs, k, trials=20):
        print(f"Verifying cluster {Vs} of size {k}")

        # k=1 is always true cluster 
        if k == 1:
            return True

        # Assemble A1 and A2
        A1 = set()
        for V in Vs:
            A1.update(self.l.pickRepresentativeMeasures(self.l.latentDict, V))
        while len(A1) > k:
            A1.pop()
        usedX = next(iter(A1))

        A2 = set()
        for V in Vs:
            A2.update(self.l.pickRepresentativeMeasures(self.l.latentDict, V, usedX))
        while len(A2) > k:
            A2.pop()
        A = A1.union(A2)

        # Current Method of Bset is not good enough
        # Can still get low rank

        # Assemble the B set
        # Each entry in the Blist is a list of sets of Vs
        # i.e. [[Vs1, Vs2, ...], [Vs1, Vs2, ...]]
        # Such that each Vs1 is of size j corresponding to their cluster
        Blist = []
        dlist = []

        # Pick vars from k-1 or smaller groups
        for parent, values in self.l.latentDict.items():
            Vlist = self.l.pickKSets(parent, usedXs=A)
            Blist.append(Vlist)

        # Pick vars from activeSet
        remainingVs = self.l.activeSet - Vs
        for parent, values in self.l.latentDict.items():
            V = values["children"]
            remainingVs = setDifference(remainingVs, V)
        subsets = generateSubset(remainingVs, k)
        Blist.append(subsets)

        # Now take cartesian product of all in Blist
        Bsets = [reduce(set.union, B) for B in product(*Blist)]

        # Calculate infoDist
        A1 = self.getMeasuredVarList(A1)
        A2 = self.getMeasuredVarList(A2)
        for i, B in enumerate(Bsets):
            B = self.getMeasuredVarList(B)
            d1 = self.g.infoDistGen(A1, B)
            d2 = self.g.infoDistGen(A2, B)
            dlist.append(d1-d2)

            if i >= trials:
                break

        # Check if all dlist is the same
        # If true, it is a real cluster
        test = True
        for i in range(1, len(dlist)):
            j = i-1
            if not isclose(dlist[i], dlist[j]):
                test = False
                break

        if k == 3:
            set_trace()

        return test


    # Algorithm to find the latent structure
    def findLatentStructure(self, verbose=True, sample=False):
        self.verbose = verbose
        self.l = LatentGroups(self.g.xvars)
        run = 1

        while True:
            k = 1
            foundList = []
            while True:
                anyFound, insufficientVars = self.runStructuralRankTest(k=k, 
                        run=run, sample=sample)

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

            # Remove variables belonging to a Group from activeSet
            # Set Groups as activeSet
            for parent in self.l.latentDict.keys():
                self.l.activeSet.add(parent)

            for values in self.l.latentDict.values():
                self.l.activeSet = setDifference(self.l.activeSet, 
                                                 values["children"])
                
            self.l.activeSet = deduplicate(self.l.activeSet)
            print(f"Active Set: {self.l.activeSet}")
            print(f"{'='*30}")

            if not any(foundList):
                break




    # Run search until convergence on each k before moving on
    def findLatentStructure2(self, verbose=True, sample=False):
        self.verbose = verbose
        self.l = LatentGroups(self.g.xvars)
        run = 1

        k = 1
        while True:
            foundList = []

            # Run until convergence for this k value
            while True:
                anyFound, insufficientVars = self.runStructuralRankTest(k=k, 
                        run=run, sample=sample)
                self.l.mergeTempSets()
                self.l.confirmTempSets() 

                # Remove variables belonging to a Group from activeSet
                # Set Groups as activeSet
                for parent in self.l.latentDict.keys():
                    self.l.activeSet.add(parent)

                for values in self.l.latentDict.values():
                    self.l.activeSet = setDifference(self.l.activeSet, 
                                                     values["children"])
                
                self.l.activeSet = deduplicate(self.l.activeSet)
                print(f"Active Set: {self.l.activeSet}")

                foundList.append(anyFound)
                if not anyFound:
                    break

                if insufficientVars:
                    break

            print(f"{'='*10} End of Run {run} {'='*10}")
            print(f"Current State:")
            pprint(self.l.latentDict, self.verbose)
            run += 1
            print(f"{'='*30}")

            # Move on to new k value
            k += 1

            # If we found anything new in this search, reset k=1
            # Basically k can only advance if nothing new is found in all
            # smaller values
            if any(foundList):
                k = 1

            if k > setLength(self.l.activeSet)/2:
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

