from MinimalGroup import MinimalGroup
from misc import *
from pdb import set_trace
from copy import deepcopy
import IPython

# Class to store discovered latent groups
class LatentGroups():
    def __init__(self, X):
        self.i = 1
        self.X = set([MinimalGroup(x) for x in X])
        self.activeSet = set([MinimalGroup(x) for x in X])
        self.latentDict = {}
        self.tempDict = {}
        self.tempSet = {}
        self.invertedDict = {}
        self.strayChildren = {}

    # When testing for k-AtomicGroups, as long as we have
    # overlap of 1 element that is newly found, we merge them
    # Need to merge, otherwise we create unnecessary latent vars
    def mergeTempSets(self):
        for k in self.tempSet:
            kTempSets = self.tempSet.pop(k)
            newTempSet = []
            while len(kTempSets) > 0:
                Vs = kTempSets.pop()
                overlap = False
                for i, tempSet in enumerate(newTempSet):
                    commonVs = Vs.intersection(tempSet)

                    # Remove elements that already belong to some AtomicGroup
                    for V in commonVs:
                        if self.inLatentDict(V):
                            commonVs = commonVs - set([V])
                    if len(commonVs) > 0:
                        newTempSet[i] = tempSet.union(Vs)
                        overlap = True
                        break
                if not overlap:
                    newTempSet.append(Vs)

            # Add back to self.tempSet
            self.tempSet[k] = newTempSet

    # Check if a variable V already belongs to an AtomicGroup
    def inLatentDict(self, V):
        for _, values in self.latentDict.items():
            if V in values["children"]:
                return True
        return False


    def addToTempSet(self, Vs, latentSize=1):
        if not latentSize in self.tempSet:
            self.tempSet[latentSize] = []
        self.tempSet[latentSize].append(Vs)

    def removeFromTempSet(self, Vs):
        for k in self.tempSet.keys():
            setlist = self.tempSet[k]
            for i, tempSet in enumerate(setlist):
                if len(tempSet.intersection(Vs)) > 0:
                    self.tempSet[k].pop(i)

    # Create a new Minimal Latent Group
    # As: Set of MinimalGroups to add as children
    # If Ls is empty, we simply create new latent variables
    def addToDict(self, Vs, latentSize=1, temp=False):

        if temp:
            d = self.tempDict
        else:
            d = self.latentDict

        #k = setLength(Vs) - 1 # size of Group
        k = latentSize

        # Find earlier discovered groups with children that
        # overlap with Vs
        overlappingGroups = []
        for parent, values in d.items():
            children = values["children"]
            if setOverlap(children, Vs):
                overlappingGroups.append(parent)

        dedupedGroups = deduplicate(set(overlappingGroups))
        overlappedCardinality = setLength(dedupedGroups)
        gap = k - overlappedCardinality
        dedupedGroups = list(dedupedGroups)

        # Reject cluster if gap < 0
        if gap < 0:
            print(f"AtomicGroup is {k} but overlap is {overlappedCardinality}")
            print(f"Vs {Vs} belong to {dedupedGroups}")
            return

        # If Vs just overlaps with exactly 1 Group that is of
        # cardinality k, we can merge them right in
        if len(dedupedGroups) == 1 and gap == 0:
            parent = dedupedGroups[0]
            values = d[parent]
            oldChildren = values["children"]
            Vs = oldChildren.union(Vs)

            # Remove previous children from Vs
            previousChildren = set()
            for group in overlappingGroups:
                if group != parent:
                    children = d[group]["children"]
                    previousChildren.update(children)
            Vs = Vs - previousChildren
            #print(f"previousChildren: {previousChildren}, Vs: {Vs}")

            # Deduplicate Children
            values["children"] = Vs
            values["children"] = deduplicate(values["children"])

            # Create a subgroup pointer for each latent var
            for V in Vs:
                if V.isLatent():
                    values["subgroups"].update([V])

            # Update the entry
            d[parent] = values

            # Update corresponding entry in invertedDict
            for existingGroup in self.invertedDict.keys():
                if oldChildren <= existingGroup:
                    self.invertedDict.pop(frozenset(existingGroup))
                    newGroup = values["children"].union(existingGroup)
                    self.invertedDict[frozenset(newGroup)] = len(parent)
                    break
            return

        # If Vs overlaps with more than 1 Group
        # If there is no overlap at all
        # If there is overlap but not enough to hit cardinality k
        # Then we need to create new Latent Variables
        newParentList = []
        for group in dedupedGroups:
            for parent in group.vars:
                newParentList.append(parent)

        for _ in range(gap):
            newParentList.append(f"L{self.i}")
            self.i += 1
        newParents = MinimalGroup(newParentList)

        # Create a subgroup pointer for each latent var
        subgroups = set()
        for V in Vs:
            if V.isLatent():
                subgroups.update([V])
        for group in dedupedGroups:
            subgroups.update([group])

        # Remove previous children from Vs
        previousChildren = set()
        for group in overlappingGroups:
            children = d[group]["children"]
            previousChildren.update(children)
        Vs = Vs - previousChildren

        # Deduplicate cases where Vs includes {L1, {L1, L3}}
        # into just {{L1, L3}}
        Vs = deduplicate(Vs)

        # Create new entry
        d[newParents] = {
                "children": Vs,
                "subgroups": subgroups
                }

        # Add to invertedDict
        Vs = deepcopy(Vs)
        for subgroup in subgroups:
            Vs.update(d[subgroup]["children"])
        self.invertedDict[frozenset(Vs)] = len(newParents)


    def confirmTempSets(self):
        for k in self.tempSet.keys():
            setlist = self.tempSet[k]
            for Vs in setlist:
                self.addToDict(Vs, k, temp=False)
        self.tempSet = {}
        self.tempDict = deepcopy(self.latentDict)


    # Recursive search for one X per latent var in minimal group L
    def pickRepresentativeMeasures(self, latentDict, L, usedXs=set()):
        assert isinstance(L, MinimalGroup), "L is not a MinimalGroup."

        if not L.isLatent():
            return set([L])

        A = set()
        values = latentDict[L]

        # Add one X per L from each subgroup
        if len(values["subgroups"]) > 0:
            for subL in values["subgroups"]:
                A.update(self.pickRepresentativeMeasures(latentDict, subL, usedXs))

        # Add remaining from own children
        #n = len(L) - setLength(values["subgroups"])
        availableXs = values["children"] - usedXs
        for V in availableXs:
            if not V.isLatent():
                A.add(V)

        # Make sure we return same cardinality as L
        while len(A) > len(L):
            A.pop()

        return A

    # As opposed to pickRepresentativeMeasures, pickAllMeasures 
    # recursively picks all measured variables that are in the subgroups
    # of the provided MinimalGroup.
    def pickAllMeasures(self, L):
        assert isinstance(L, MinimalGroup), "L is not a MinimalGroup."

        if not L.isLatent():
            return set([L])

        A = set()
        values = self.latentDict[L]

        if len(values["subgroups"]) > 0:
            for subL in values["subgroups"]:
                A.update(self.pickAllMeasures(subL))
        
        for C in values["children"]:
            if not C.isLatent():
                A.add(C)

        # Add Stray Children that belong
        for parent, values in self.strayChildren.items():
            if parent.vars <= L.vars:
                A.update(values["children"])

        return A

