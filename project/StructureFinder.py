from LatentGroups import LatentGroups
from pdb import set_trace
from misc import *
from math import floor, sqrt
import numpy as np
import scipy.stats
import operator
from functools import reduce
from RankTester import RankTester

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
        n = df.shape[0]
        data = df.to_numpy()
        self.S = 1/(n-1) * data.T @ data  # Sample Covariance
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
    # Returns pValue
    def sampleRankTest(self, A, B, rk):
        test = rankTester.test([0,5,9], [1,6,4], r=2)

        assert len(self.covList) > 0, "Need to Prepare Covariances"
        cols = list(self.df.columns)
        I = [cols.index(a) for a in A]
        J = [cols.index(b) for b in B]
        n = self.df.shape[0]

        detList = []
        k = len(self.covList)
        for S in self.covList:
            SIJ = S[np.ix_(I, J)]
            Sigma = np.linalg.svd(SIJ, compute_uv=False, full_matrices=False)
            Sigma = Sigma[0:(rk+1)]
            det = reduce(operator.mul, Sigma, 1)
            detList.append(det)
        return detList

        # Sum of squares of Z-distributed statistics follows chi2 distribution
        # With degrees of freedom k
        #sampleMean = 1/k * sum(detList)
        #sampleVar = 1/(k-1) * sum([pow(det - sampleMean, 2) for 
        #                            det in detList])
        #testStat = sampleMean / sqrt(sampleVar / 2)
        #pValue = 1 - scipy.stats.chi2.cdf(testStat, 1)
        #return pValue


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
        if len(Vs) > len(remainingVs):
            return None

        As = set()
        for V in Vs:
            As.update(self.l.pickAllMeasures(V))
        As = self.getMeasuredVarList(As)
                
        Bs = set()
        for V in remainingVs:
            Bs.update(self.l.pickAllMeasures(V))
        Bs = self.getMeasuredVarList(Bs)

        if len(Bs) < len(As):
            return None

        if not sample:
            return self.g.rankTest(As, Bs, k-1)

        else:
            pval = self.sampleRankTest(As, Bs, k-1)
            return pval >= self.alpha


    # Test a k subset of Variables to find Clusters
    def runStructuralRankTest(self, k=2, run=1, sample=False):
        vprint(f"Starting structuralRankTest k={k}...", self.verbose)
        anyFound = False
        for Vs in combinations(self.l.activeSet, k):
            Vs = set(Vs)

            if setLength(Vs) < k:
                break

            rankDeficient = self.structuralRankTest(Vs, k, run, sample)
            if rankDeficient is None:
                break

            if rankDeficient:
                self.l.addToLatentSet(set(), Vs)
                vprint(f"Found cluster {Vs}.", self.verbose)
                anyFound = True

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

        try:
            rankDeficient = self.parentSetTest(Ls, V, sample)
        except:
            import IPython; IPython.embed(); exit(1)
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


    # Algorithm to find the latent structure
    def findLatentStructure(self, verbose=True, sample=False):
        self.verbose = verbose
        self.l = LatentGroups(self.g.xvars)
        run = 1
        k = 2
        while len(self.l.activeSet) > 0:
            test1 = self.runStructuralRankTest(k=k, run=run, sample=sample)
            #test2 = self.runParentSetTest(run, sample)
            test2 = False
            if not any([test1, test2]):
                break

            # Move latentSet back into activeSet
            self.l.activeSet.update(self.l.latentSet)
            self.l.latentSet = set()
            k += 1

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

