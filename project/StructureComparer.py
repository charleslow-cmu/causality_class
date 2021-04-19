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

    def getClusterings(self, l):
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

    def getScore(self):
        C = self.getClusterings(self.reference)
        Cprime = self.getClusterings(self.test)
        score = self.mutualInfo(C, Cprime)
        return score


