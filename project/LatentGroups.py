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
        self.tempSet = []
        self.invertedDict = {}
        self.strayChildren = {}

    def addToTempSet(self, Vs, latentSize=1):
        self.tempSet.append((Vs, latentSize))

    # Create a new Minimal Latent Group
    # As: Set of MinimalGroups to add as children
    # If Ls is empty, we simply create new latent variables
    def addToDict(self, d, Vs, latentSize=1):

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
        #print(f"Vs {Vs} | gap {gap}")

        # Reject cluster if gap < 0
        if gap < 0:
            print(f"AtomicGroup is {k} but overlap is {overlappedCardinality}")
            return
            #print(Vs)
            #pprint(self.latentDict, True)
            #print(overlappingGroups)
            #print(dedupedGroups)

        #assert gap >= 0, "Cardinality Gap cannot be negative"

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
        for Vs, k in self.tempSet:
            self.addToDict(self.latentDict, Vs, k)
        self.tempSet = []
        self.tempDict = deepcopy(self.latentDict)


    def addStrayChild(self, Ls, V):
        newLlist = []
        for L in Ls:
            for var in L.vars:
                newLlist.append(var)
        newParents = MinimalGroup(newLlist)

        if not newParents in self.strayChildren:
            self.strayChildren[newParents] = {
                    "children": set([V]),
                    "subgroups": Ls
                    }
        else:
            self.strayChildren[newParents]["children"].add(V)
            self.strayChildren[newParents]["subgroups"].update(Ls)

        # Remove from activeSet
        self.activeSet = self.activeSet - set([V])


    # Recursive search for one X per latent var in minimal group L
    def pickRepresentativeMeasures(self, L, usedXs=set()):
        assert isinstance(L, MinimalGroup), "L is not a MinimalGroup."

        if not L.isLatent():
            return set([L])

        A = set()
        values = self.latentDict[L]
        n = len(L)

        # Add one X per L from each subgroup
        if len(values["subgroups"]) > 0:
            for subL in values["subgroups"]:
                A.update(self.pickRepresentativeMeasures(subL, usedXs))

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


    # Return a set of all other measures that are not in the groups of 
    # the provided Vs.
    # Each v is a MinimalGroup
    def getAllOtherMeasuresFromGroups(self, Vs=set()):

        # Measures that we should not include
        Vmeasures = set()
        for V in Vs:
            Vmeasures.update(self.pickAllMeasures(V))
        return self.X - Vmeasures
            
    def getAllOtherMeasuresFromXs(self, Xs=set()):
        return self.X - Xs
