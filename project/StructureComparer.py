import IPython
from copy import deepcopy
from pdb import set_trace

class StructureComparer:

    def __init__(self, reference, test):
        self.reference = reference
        self.test = test

    def getClusterings(self, l):
        l = deepcopy(l)
        Clist = []

        # At the top level, non-clustered variables form one group
        # And the rest form another group
        # Only add if there are non-clustered variables
        noClusterXs = set()
        for V in l.activeSet:
            if not V.isLatent():
                noClusterXs.add(V)
        if len(noClusterXs) > 0:
            Clist.append([noClusterXs, l.X - noClusterXs])

        set_trace()

        while len(l.latentDict) > 0:
            C = []
            newActiveSet = set()
            for group in l.activeSet:
                set_trace()
                Vs = l.pickAllMeasures(group)
                C.append(Vs)
                if group.isLatent():
                    values = l.latentDict.pop(group)
                    newActiveSet.update(values["children"])
                else:
                    newActiveSet.add(group)
            Clist.append(C)
            l.activeSet = newActiveSet


        IPython.embed(); exit(1)
        return Clist

                

    def getScore(self):
        Clist = self.getClusterings(self.reference)



