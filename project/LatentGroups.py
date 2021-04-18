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
        self.latentSet = set()
        self.activeSet = set([MinimalGroup(x) for x in X])
        self.latentDict = {}
        self.latentDictTemp = {}

    # Create a new Minimal Latent Group
    # As: Set of MinimalGroups to add as children
    # If Ls is empty, we simply create new latent variables
    def addToLatentSet(self, Vs, latentSize=1):

        # If Vs overlap with an earlier cluster, add to it
        for parent, values in self.latentDictTemp.items():
            children = values["children"]
            if len(children.intersection(Vs)) > 0:
                values["children"] = children.union(Vs)

                # Create a subgroup pointer for each latent var
                for V in Vs:
                    if V.isLatent():
                        values["subgroups"].update([V])

                # Update the entry
                self.latentDictTemp[parent] = values
                return


        # If no overlap, create new Latent Variables
        newLlist = []
        for _ in range(latentSize):
            newLlist.append(f"L{self.i}")
            self.i += 1
        newParents = MinimalGroup(newLlist)

        # Create a subgroup pointer for each latent var
        subgroups = set()
        for V in Vs:
            if V.isLatent():
                subgroups.update([V])

        # Create new entry
        self.latentDictTemp[newParents] = {
                "children": Vs,
                "subgroups": subgroups
                }


    # Merge overlapping groups in dTemp
    #def mergeTempGroups(self, run=1):

    #    temp = deepcopy(self)

    #    if len(self.latentDictTemp) == 0:
    #        return

    #    # Ps for parent set, Cs for children set
    #    # Values of inv_list becomes a list of MinimalGroups
    #    inv_list = {}
    #    for P, values in self.latentDictTemp.items():
    #        Cs = values["children"]
    #        for C in Cs:
    #            inv_list[C] = inv_list.get(C, set()).union([P])

    #    # Merge Ps with overlapping elements in the same group
    #    foundGroups = []
    #    for Ps in inv_list.values():
    #        if len(Ps) > 1:
    #            if len(foundGroups) == 0:
    #                foundGroups.append(Ps)
    #                continue
    #
    #            for i, group in enumerate(foundGroups):
    #                if len(Ps.intersection(group)) > 0:
    #                    foundGroups[i] = foundGroups[i].union(Ps)
    #                else:
    #                    foundGroups.append(Ps)

    #    # foundGroups is now a list of parentSets with overlapping
    #    # children
    #    # Need to do a pairwise merge
    #    try:
    #        for group in foundGroups:
    #            mergeMap = self.getMergeMap(group)
    #            for oldp, p in mergeMap.items():
    #                values = self.latentDictTemp.pop(oldp)
    #                if not p in self.latentDictTemp:
    #                    self.latentDictTemp[p] = {}
    #                    self.latentDictTemp[p]["children"] = set()
    #                    self.latentDictTemp[p]["subgroups"] = set()
    #                self.latentDictTemp[p]["children"].update(values["children"])
    #                self.latentDictTemp[p]["subgroups"].update(values["subgroups"])
    #    except:
    #        IPython.embed();exit(1)

    # Return a 1-1 mapping from original group to new group
    # group: set of MinimalGroups of latent vars
    #def getMergeMap(self, group):
    #    mergeMap = {}

    #    # Find lowest keys
    #    k = len(next(iter(group)))
    #    nums = []
    #    for A in group:
    #        A1, A2 = self.findMergeableVars(A)
    #        Anums = [int(x[1:]) for x in A2]
    #        nums.extend(Anums)
    #    lowestKeys = sorted(nums)[0:k]
    #    lowestKeys = [f"L{x}" for x in lowestKeys]

    #    # Create new keys
    #    for A in group:
    #        A1, A2 = self.findMergeableVars(A)
    #        newkey = A1.union(lowestKeys)
    #        mergeMap[A] = MinimalGroup(list(newkey))
    #    return mergeMap

    # Given a set of MinimalGroups of latent vars, find
    # subgroups which are not minimal groups.
    #def findMergeableVars(self, A):
    #    A1 = set() # Non-mergeable
    #    A2 = set() # Mergeable
    #    for P in A.vars:

    #        # Not mergeable if already confirmed in latentDict
    #        if P in self.latentDict:
    #            A1.update([P])
    #        else:
    #            A2.update([P])
    #    return A1, A2


    # Move elements in latentDictTemp to latentDict
    def confirmTempGroups(self, run=1):
        while len(self.latentDictTemp) > 0:
            p, values = self.latentDictTemp.popitem()
            self.latentDict[p] = values

            # Update V by grouping variables
            self.activeSet = setDifference(self.activeSet, 
                                    values["children"])
            self.latentSet.add(p)


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
                break
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
