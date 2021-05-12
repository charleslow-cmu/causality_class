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

            if rankDeficient:
                self.l.addToDict(Vs, k-gap, temp=True)
                self.l.addToTempSet(Vs, k-gap)
                vprint(f"Found cluster {Vs}.", self.verbose)
                #if run > 1 and k >= 2:
                #    set_trace()
                anyFound = True

        return (anyFound, insufficientVars)


    def verifyClusters1(self, run=1, sample=False):
        vprint(f"Verifying Clusters...", self.verbose)

        # Active variables (not part of any k group)
        clusteredVars = set()
        for p, v in self.l.tempDict.items():
            clusteredVars.update(v["children"])
        activeVars = setDifference(self.l.activeSet, clusteredVars)

        for Vs, k in self.l.tempSet:

            # No need to test k=1
            if k == 1:
                continue

            # Find the entry in tempDict
            for parent, values in self.l.tempDict.items():
                if values["children"] <= Vs:
                    break
            subgroups = values["subgroups"]
            children = values["children"]

            # Assemble subgroup vars
            S = set()
            for subgroup in subgroups:
                if not subgroup in children:
                    S.update(self.l.pickAllMeasures(subgroup))

            # Assemble the B group
            B = set()
            for V in activeVars:
                B.update(self.l.pickAllMeasures(V))

            for p, v in self.l.tempDict.items():
                if (p != parent) and (not p in subgroups):
                    B.update(self.l.pickRepresentativeMeasures(self.l.tempDict, p))
            B = self.getMeasuredVarList(B)

            # Test all subsets
            Slen = setLength(subgroups)
            gap = max(1, k-Slen)
            for childSet in generateSubset(children, k=gap):
                A = set()
                for child in childSet:
                    A.update(self.l.pickAllMeasures(child))
                A = A.union(S)
                Alist = self.getMeasuredVarList(A)
                rankDeficient = self.g.rankTest(Alist, B, k-1)
                if rankDeficient:
                    print(f"False cluster! {A}")
                    self.l.removeFromTempSet(A)


    # Verify if all k+1 subsets in a cluster get the rank deficiency
    #def verifyClusters2(self, run=1, sample=False):
    #    vprint(f"Verifying Clusters...", self.verbose)

    #    # Active variables (not part of k groups)
    #    clusteredVars = set()
    #    for p, v in self.l.tempDict.items():
    #        clusteredVars.update(v["children"])
    #    activeVars = setDifference(self.l.activeSet, clusteredVars)

    #    for Vs, k in self.l.tempSet:

    #        # Find the entry in tempDict
    #        for parent, values in self.l.tempDict.items():
    #            if values["children"] <= Vs:
    #                break
    #        subgroups = values["subgroups"]
    #        children = values["children"]

    #        # Assemble subgroup vars
    #        S = set()
    #        for subgroup in subgroups:
    #            if not subgroup in children:
    #                S.update(self.l.pickAllMeasures(subgroup))

    #        # Take up to k-1 vars from S
    #        # Take remaining vars from A
    #        # The remaining variables for A take from children
    #        j = 0
    #        Ssubsets = []
    #        if len(S) > 0:
    #            while True:
    #                j += 1
    #                Ssubsets = generateSubset(S, k=k-j)
    #                if len(Ssubsets) > 0:
    #                    break
    #                assert k-j >= 0, "k-j is negative"

    #        if len(S) > 0:
    #            gap = j
    #        else:
    #            gap = k
    #        Asubsets = generateSubset(children, k=gap)


    #        # Combine Ssubsets and Asubsets
    #        subsets = []
    #        for As in Asubsets:
    #            if len(Ssubsets) > 0:
    #                for Ss in Ssubsets:
    #                    subset = As.union(Ss)
    #                    subsets.append(subset)
    #            else:
    #                subsets.append(As)
    #        print(subsets)

    #        # Assemble the B group
    #        B = set()
    #        for V in activeVars:
    #            B.update(self.l.pickRepresentativeMeasures(self.l.latentDict, V))

    #        for p, v in self.l.tempDict.items():
    #            if (p != parent) and (not p in subgroups):
    #                B.update(self.l.pickRepresentativeMeasures(self.l.tempDict, p))

    #        # Take two distinct k subsets from B that are not rankDeficient
    #        Bsubsets = generateSubset(B, k)
    #        Blist = []
    #        for Bp in Bsubsets:
    #            Bp = self.getMeasuredVarList(Bp)
    #            fullrank = True
    #            for A in subsets:
    #                A = self.getMeasuredVarList(A)
    #                rankDeficient = self.g.rankTest(A, Bp, k-1)
    #                if rankDeficient:
    #                    fullrank = False
    #                    break
    #            if fullrank:
    #                Blist.append(Bp)
    #            if len(Blist) >= 2:
    #                break
    #        B1 = Blist[0]
    #        B2 = Blist[1]

    #        difflist = []
    #        Alist = []
    #        for A in subsets:
    #            A = self.getMeasuredVarList(A)
    #            assert len(A) == k, "A is not k length"
    #            assert len(set(A).intersection(set(B1))) == 0, "A overlap with B1"
    #            assert len(set(A).intersection(set(B2))) == 0, "A overlap with B2"

    #            if not sample:
    #                diff = self.g.infoDist(A, B1) - self.g.infoDist(A, B2)
    #                difflist.append(diff)
    #                Alist.append(set(A)-S)

    #        for diff in difflist:
    #            if not isclose(diff, difflist[0]):
    #                print(f"Not all belong to a cluster")
    #                print(*zip(difflist, Alist))


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

                self.verifyClusters1(run=run, sample=sample)
                #self.l.mergeTempSets()
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
            for parent, values in self.l.latentDict.items():
                self.l.activeSet.add(parent)
                self.l.activeSet = setDifference(self.l.activeSet, 
                                                 values["children"])
                self.l.activeSet = deduplicate(self.l.activeSet)


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

