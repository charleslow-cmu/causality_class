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
        self.invertedDict = {}
        self.strayChildren = {}

    # Create a new Minimal Latent Group
    # As: Set of MinimalGroups to add as children
    # If Ls is empty, we simply create new latent variables
    def addToLatentSet(self, Vs, latentSize=1):

        k = setLength(Vs) - 1 # size of Group

        # If Vs overlap with an earlier cluster, add to it
        overlappingGroups = []
        for parent, values in self.latentDict.items():
            children = values["children"]
            if len(children.intersection(Vs)) > 0:
                overlappingGroups.append(parent)

        overlappedCardinality = setLength(set(overlappingGroups))
        gap = k - overlappedCardinality

        # If Vs just overlaps with exactly 1 Group that is of
        # cardinality k, we can merge them right in
        if len(overlappingGroups) == 1 and gap == 0:
           values = self.latentDict[parent]
           values["children"] = values["children"].union(Vs)

           # Deduplicate Children
           values["children"] = deduplicate(values["children"])

           # Create a subgroup pointer for each latent var
           for V in Vs:
               if V.isLatent():
                   values["subgroups"].update([V])

           # Update the entry
           self.latentDict[parent] = values
           return

        # If Vs overlaps with more than 1 Group
        # If there is no overlap at all
        # If there is overlap but not enough to hit cardinality k
        # Then we need to create new Latent Variables
        newParentList = []
        for group in overlappingGroups:
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
        for group in overlappingGroups:
            subgroups.update([group])

        # Remove previous children from Vs
        previousChildren = set()
        for group in overlappingGroups:
            children = self.latentDict[group]["children"]
            previousChildren.update(children)
        Vs = Vs - previousChildren

        # Deduplicate cases where Vs includes {L1, {L1, L3}}
        # into just {{L1, L3}}
        newVs = deduplicate(Vs)
        print(f"Vs: {Vs}, newVs: {newVs}")

        # Create new entry
        self.latentDict[newParents] = {
                "children": newVs,
                "subgroups": subgroups
                }

        self.invertedDict[frozenset(newVs)] = len(newParents)


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
