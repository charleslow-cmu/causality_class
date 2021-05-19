from Group import Group
from misc import *
from pdb import set_trace
from copy import deepcopy
import IPython
from functools import reduce

# Class to store discovered latent groups
class LatentGroups():
    def __init__(self, X):
        self.i = 1
        self.X = set([Group(x) for x in X])
        self.activeSet = set([Group(x) for x in X])
        self.latentDict = {}
        self.tempSet = {}

    def findChild(self, V):
        parents = set()
        for parent, values in self.latentDict.items():
            if V in values["children"]:
                parents.add(parent)
        return parents

    def updateActiveSet(self):

        # Remove variables belonging to a Group from activeSet
        # Set Groups as activeSet
        # Only update MinimalGroups
        for parent in self.latentDict.keys():
            if parent.isMinimal():
                self.activeSet.add(parent)

        for parent, values in self.latentDict.items():
            self.activeSet = setDifference(self.activeSet, 
                                             values["children"])
            
        self.activeSet = deduplicate(self.activeSet)



    # Root is always a split point
    # P is the ancestor junction, L is current node testing
    def findJunctions(self, L=None, P=None, junctions={}):

        if L is None:
            root = Group("root")
            junctions[root] = set()
            for L in self.activeSet:
                j = self.findJunctions(L, root)
                junctions.update(j)

        else:
            values = self.latentDict[L]
            subgroups = values["subgroups"]

            # This is a junction and we include it
            # Continue search
            if len(subgroups) > 1:
                #print(f"Found junction {L}")
                junctions[P] = junctions.get(P, set()) | set([L])
                for subgroup in subgroups:
                    j = self.findJunctions(subgroup, L)
                    junctions.update(j)

            # Continue search if only one subgroup
            if len(subgroups) == 1:
                #print(f"{L} is not a junction: 1 child")
                subgroup = next(iter(subgroups))
                junctions.update(self.findJunctions(subgroup, P))

            # Terminate search when we hit the root
            if len(subgroups) == 0:
                #print(f"{L} is not a junction: root")
                return junctions

        return junctions



    # When testing for k-AtomicGroups, as long as we have
    # overlap of 1 element that is newly found, we merge them
    # Need to merge, otherwise we create unnecessary latent vars

    # Reject clusters that cause overlap of too many AtomicGroups
    def mergeTempSets(self):
        tempSetCopy = deepcopy(self.tempSet)
        self.tempSet = {}
        while len(tempSetCopy) > 0:
            k, kTempSets = tempSetCopy.popitem()
            newTempSet = []

            # Check overlap of each variable
            if len(kTempSets) > 0:
                variables = reduce(set.union, kTempSets)
                for var in variables:
                    parents = self.findChild(var)

            while len(kTempSets) > 0:
                Vs = kTempSets.pop()

                # Check if this is a valid cluster
                #parents = set()
                #for V in Vs:
                #    parents.update(self.findChild(V))
                #parents = deduplicate(parents)

                #if setLength(parents) > k:
                #    continue

                overlap = False
                print(f"Found {k}-cluster {Vs}!")
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

    
    # Remove a cluster
    def dissolve(self, V):
        assert isinstance(V, Group), f"{V} must be Group"
        print(f"Dissolving {V}...")

        # Remove from subgroups of parents
        for parent, values in self.latentDict.items():
            if V in values["subgroups"]:
                self.latentDict[parent]["subgroups"] = \
                        self.latentDict[parent]["subgroups"] - set([V])


        values = self.latentDict.pop(V)
        self.activeSet.update(values["children"])
        self.activeSet = setDifference(self.activeSet, set([V]))
        return values["children"], values["subgroups"]


    # Recursively dissolve to the leaves of this branch
    # or if we hit the nextJunctions
    def dissolveRecursive(self, L, nextJunctions):
        if not L in nextJunctions:
            Vs, subgroups = self.dissolve(L)
            for subgroup in subgroups:
                self.dissolveRecursive(subgroup, nextJunctions)


    # Create a new Minimal Latent Group
    # As: Set of Groups to add as children
    # If Ls is empty, we simply create new latent variables
    def addToDict(self, Vs, latentSize=1):
        d = self.latentDict
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

        # if gap is < 0, something is wrong
        if gap < 0:
            print(f"AtomicGroup {Vs} is {k} but overlap is {overlappedCardinality}")
            #for group in dedupedGroups:
            #    print(f"Dissolving {group}...")
            #    self.dissolve(group)
            #return

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
            return

        # If Vs overlaps with more than 1 Group
        # If there is no overlap at all
        # If there is overlap but not enough to hit cardinality k
        # Then we need to create new Latent Variables
        
        # Whether this Group is minimal or not.
        isMinimal = gap > 0

        newParentList = []
        for group in dedupedGroups:
            for parent in group.vars:
                newParentList.append(parent)

        for _ in range(gap):
            newParentList.append(f"L{self.i}")
            self.i += 1
        newParents = Group(newParentList, isMinimal)

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
        #Vs = deepcopy(Vs)
        #for subgroup in subgroups:
        #    Vs.update(d[subgroup]["children"])
        #self.invertedDict[frozenset(Vs)] = len(newParents)



    def confirmTempSets(self):
        for k in self.tempSet.keys():
            setlist = self.tempSet[k]
            for Vs in setlist:
                self.addToDict(Vs, k)
        self.tempSet = {}


    def pickKSets(self, V, setlist=[set()], usedXs=set()):

        assert isinstance(V, Group), f"{V} is not a Group."
        if not V.isLatent():
            return [set([V])]
    
        # Get values
        values = self.latentDict[V]
        subgroups = values["subgroups"]
        children = values["children"]
        measures = set()
        for child in children:
            if not (child.isLatent() or child in usedXs):
                measures.add(child)
    
        # Get all combinations of measured children
        subgroupCardinality = setLength(subgroups)
        childCardinality = len(V) - subgroupCardinality
        subsets = scombinations(measures, childCardinality)

        # Add measures
        setlist = cartesian(setlist, subsets)

        # Hit leaf node, return setlist
        if subgroupCardinality <= 0:
            return setlist

        # Continue search if subgroups exist
        if subgroupCardinality > 0:
            for subgroup in subgroups:
                setlist = self.pickKSets(subgroup, setlist)
    
        return setlist


    # Recursive search for one X per latent var in minimal group L
    def pickRepresentativeMeasures(self, latentDict, L, usedXs=set()):
        assert isinstance(L, Group), "L is not a Group."
        if isinstance(usedXs, Group):
            usedXs = set([usedXs])

        if not L.isLatent():
            return set([L]) - usedXs

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
    # of the provided Group.
    def pickAllMeasures(self, L):
        assert isinstance(L, Group), "L is not a Group."

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

