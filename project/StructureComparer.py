import IPython
from copy import deepcopy
from pdb import set_trace
import math
from misc import *

class StructureComparer:

    def __init__(self, reference, test):
        assert reference.X == test.X, "LatentGroups not over same variables"
        self.reference = reference
        self.test = test
        self.X = reference.X
        self.n = len(self.X)

    def getSingleClustering(self, l):
        l = deepcopy(l)
        Clist = []

        # At the top level, non-clustered variables form one group
        # And the rest form another group
        # Only add if there are non-clustered variables
        #noClusterXs = set()
        #for V in l.activeSet:
        #    if not V.isLatent():
        #        noClusterXs.add(V)
        #if len(noClusterXs) > 0:
        #    Clist.append([noClusterXs, l.X - noClusterXs])

        newActiveSet = set()
        C = []
        for group in l.activeSet:

            Vs = l.pickAllMeasures(group)
            C.append(Vs)
            if group.isLatent():
                values = l.latentDict.pop(group)
                newActiveSet.update(values["children"])

        l.activeSet = newActiveSet
        return C

    def getGroups(self, l):
        l = deepcopy(l)
        C= []

        # Get strayChildren Groups
        while len(l.strayChildren) > 0:
            parents, values = l.strayChildren.popitem()
            Vs = values["children"]
            for L in values["subgroups"]:
                Vs.update(l.pickAllMeasures(L))
            C.append(Vs)

        # Get normal groups
        while len(l.latentDict) > 0:
            parents, values = l.latentDict.popitem()
            Vs = set()
            for V in values["children"]:
                Vs.update(l.pickAllMeasures(V))
            C.append(Vs)

        return C



    # i refers to the group index 
    def marginalProb(self, C, i):
        return len(C[i]) / self.n

    def jointProb(self, C, Cprime, i, j):
        intersectingElements = C[i].intersection(Cprime[j])
        return len(intersectingElements) / self.n

    def entropy(self, C):
        ent = 0
        for i in range(len(C)):
            pi = self.marginalProb(C, i)
            ent += -pi * math.log2(pi)
        return ent

    # We consider log0 = 0
    def mutualInfo(self, C, Cprime):
        mi = 0
        for i in range(len(C)):
            for j in range(len(Cprime)):
                jp = self.jointProb(C, Cprime, i, j)
                if jp > 0:
                    pi = self.marginalProb(C, i)
                    pj = self.marginalProb(Cprime, j)
                    mi += jp * math.log2(jp / (pi * pj))
        entC = self.entropy(C)
        entCprime = self.entropy(Cprime)
        return mi / sqrt(entC * entCprime)
                

    def makeMinGroup(self, Xs):
        return set([MinimalGroup(x) for x in Xs])

    # A measure of similarity between V1 and V2
    def smallestContainingSet(self, C, V1, V2):
        d = self.n
        for group in C:
            if V1 in group and V2 in group:
                l = len(group)
                if l < d:
                    d = l
        return d/self.n

    def makeDistanceMatrix(self, l, C):
        print(C)
        Xs = l.X
        dist = np.zeros((len(Xs), len(Xs)))
        for i, Xi in enumerate(Xs):
            for j, Xj in enumerate(Xs):
                if i == j:
                    continue
                dist[i,j] = self.smallestContainingSet(C, Xi, Xj)
        return dist

    def getScore(self):
        #C2 = [set([x for x in self.reference.X])]
        #C2 = [set([x]) for x in self.reference.X]
        #print(C2)
        C1 = self.getGroups(self.reference)
        C2 = self.getGroups(self.test)
        dist1 = self.makeDistanceMatrix(self.reference, C1)
        dist2 = self.makeDistanceMatrix(self.test, C2)
        norm1 = np.linalg.norm(dist1, ord="fro")
        norm2 = np.linalg.norm(dist2, ord="fro")
        score = 1 - np.linalg.norm(dist1 - dist2, ord="fro") / (norm1*norm2)
        return score


