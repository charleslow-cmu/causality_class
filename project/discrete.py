import random
import pandas as pd
import numpy as np
from itertools import product, combinations
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import List, Tuple
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'xtick.labelsize': 12})
plt.rcParams.update({'ytick.labelsize': 12})
plt.rcParams.update({'axes.labelsize': 18})
import seaborn as sns
from tqdm import tqdm
import sys
from pdb import set_trace

def sq(X):
    return math.pow(X, 2)

def pow(x, y):
    return int(math.pow(x, y))

# Return list of items in first that are not in second
def listdiff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

# Add 'key' column based on unique combi of keys
def add_key_column(df, vlist):
    df['key'] = df.loc[:, vlist].agg('-'.join, axis=1)
    return df


def dicttostr(d):
    l = []
    for k,v in d.items():
        l.append(f"{k}:{v}")
    return f"|{','.join(l)}|"


# Given a dictionary of key and values, retrieve
# corresponding row from df
def lookup(df, d):
    vset = set(df.columns)
    for k, v in d.items():
        if k in vset:
            df = df.loc[df[k] == v]
    #print(f"Lookup with {dicttostr(d)} found {df.shape[0]} entries.")
    return sum(df["p"])


# Given a dictionary of key and values, 
# Multiply all matching rows in df with "p" value
def replace(df: pd.DataFrame, d, p):
    vset = set(df.columns)
    l = []
    for k, v in d.items():
        if k in vset:
            l.append(f"({k} == '{v}')")
    idx = df.query(" & ".join(l)).index
    df.loc[idx, "p"] *= p



def sort_columns(df):
    l = ["L", "X", "p", "key"]
    cols = []
    for letter in l:
        cols.extend([col for col in df.columns 
                        if col.startswith(letter)])
    return df.reindex(cols, axis=1)


class Graph:
    def __init__(self):
        self.vars = []
        self.xvars = []
        self.lvars = []
        self.jpd = None

    # Assume binary variables
    def add_variable(self, name, parents=None):
        if isinstance(parents, str):
            parents = [parents]

        # Sanity checks
        if not parents is None:
            variables = set(self.lvars) | set(self.xvars)
            valid_parents = len(set(parents)-variables) == 0
            assert valid_parents, "Parents not found in graph!"

        # Keep track of variables
        self.vars.append(name)
        if "X" in name:
            self.xvars.append(name)
        else:
            self.lvars.append(name)

        # Adding first variable
        if len(self.vars) == 1:
            self.jpd = pd.DataFrame({"key": [0, 1], "p": self.make_column()})
            return

        # Adding subsequent vars
        self.make_jpd(parents, name)
        assert math.isclose(sum(self.jpd["p"]), 1), "JPD does not sum to 1"
        #print(f"Adding {name}... jpd is {self.jpd.shape}")

    # Randomly sample the conditional distribution
    def make_column(self):
        p = random.uniform(0.05, 0.95)
        lprobs = [p, 1-p]
        return pd.Series(lprobs)
    
    def cpdKeyToJpdKey(self, cpdKey, matchingIndices):
        key = 0
        shift = len(matchingIndices)-1
        i = 0
        while shift >= 0:
            leftmostBit = ((cpdKey & (1 << shift)) >> shift)
            key += (leftmostBit << matchingIndices[i])
            shift -= 1
            i += 1
        return key

    def makeMask(self, matchingIndices):
        mask = 0
        for idx in matchingIndices:
            mask += (1 << idx)
        return mask

    def varToBitIdx(self, varname):
        return len(self.vars)-self.vars.index(varname)-1

    def getValueAtBitIdx(self, keys, bitIdx):
        return (keys & (1 << bitIdx)).values >> bitIdx
    
    # Use bits of jpd.key as indicator variables
    def make_jpd(self, parents, name):
    
        # Duplicate the dataframe
        self.jpd["key"] = self.jpd["key"] * 2 # Bitwise left shift
        jpd2 = self.jpd.copy(deep=True)
        jpd2["key"] = jpd2["key"] + 1
        self.jpd = self.jpd.append(jpd2, ignore_index=True)
    
        # No parents case
        if parents is None:
            p = self.make_column()
            filter0 = ((self.jpd["key"]+1) & 1).astype(bool)
            filter1 = (self.jpd["key"] & 1).astype(bool)
            self.jpd.loc[filter0, "p"] *= p[0]
            self.jpd.loc[filter1, "p"] *= p[1]
    
        else:
            # Make CPD P(new var | parents)
            k = len(parents)
            matchingIndices = [len(self.vars)-self.vars.index(x)-1 for x in parents]
            matchingIndices.append(0)
            matchingIndices.sort(reverse=True)
            mask = self.makeMask(matchingIndices)
            cpd_keys = list(range(pow(2, k+1)))
            p = pd.concat([self.make_column() for x in range(pow(2, k))],
                          ignore_index=True)

            for i, _ in enumerate(cpd_keys):
                cpd_key = cpd_keys[i]
                jpd_key = self.cpdKeyToJpdKey(cpd_key, matchingIndices)
                jpdfilter = (mask & (self.jpd["key"]^jpd_key)).astype(bool)
                jpdfilter = ~jpdfilter
                self.jpd.loc[jpdfilter, "p"] *= p[i]

    ## Get Marginal Probability P(X)
    #def get_marginal(self, X: str):
    #    Dx = self.jpd.groupby(X)["p"].sum().reset_index()
    #    return Dx

    ## Get Conditional Probability P(X|Ys)
    #def get_conditional(self, X: str, Ys: List[str]):
    #    if isinstance(Ys, str):
    #        Ys = [Ys]
    #    Vs = Ys + [X]
    #    Jxy = self.jpd.groupby(Vs)["p"].sum().reset_index()
    #    Jxy["p"] = Jxy.groupby(Ys)["p"].apply(lambda x: x/x.sum())
    #    Jxy = Jxy.pivot(index=Ys, columns=X, values="p")
    #    return Jxy

    ## Get Joint Probability P(X,Y)
    #def get_joint(self, Xs: List[str], Ys: List[str]):
    #    if isinstance(Xs, str):
    #        Xs = [Xs]
    #    if isinstance(Ys, str):
    #        Ys = [Ys]
    #    Vs = Xs + Ys
    #    Jxy = self.jpd.groupby(Vs)["p"].sum().reset_index()
    #    Jxy = Jxy.pivot(index=Xs, columns=Ys, values="p")
    #    return Jxy

    def calc_infdist(self, Xs: List[str], Ys: List[str]):
        if isinstance(Xs, str):
            Xs = [Xs]
        if isinstance(Ys, str):
            Ys = [Ys]

        Jxy = self.jpd.groupby(Xs+Ys)["p"].sum().reset_index()
        Jxy = Jxy.pivot(index=Xs, columns=Ys, values="p")
        Jxy = np.array(Jxy)
        dxy = abs(np.linalg.det(Jxy))
        dx = Jxy.sum(axis=1).prod()
        dy = Jxy.sum(axis=0).prod()

        if math.isclose(dxy, 0, abs_tol=1e-16):
            return None
        
        dist = -math.log(dxy) + 0.5*(math.log(dx) + math.log(dy))
        return dist

    # Generalized Information Distance
    # If a list of variables is provided, these variables are "merged"
    # to form a new one.
    def calc_infdist2(self, Xs: List[str], Ys: List[str]):
        if isinstance(Xs, str):
            Xs = [Xs]
        if isinstance(Ys, str):
            Ys = [Ys]
        Vs = Xs + Ys
        Jxy = self.jpd.groupby(Vs)["p"].sum().reset_index()
        Jxy = Jxy.pivot(index=Xs, columns=Ys, values="p")
        Jxy = np.array(Jxy)
        u, d, v = np.linalg.svd(Jxy)
        tol = d.max() * max(Jxy.shape) * sys.float_info.epsilon
        d = d[d > tol]
        #print(f"Singular P_{X}{Y}: {d}")
        dx = Jxy.sum(axis=1).prod()
        dy = Jxy.sum(axis=0).prod()
        dist = -np.sum(np.log(d)) + 0.5*(math.log(dx) + math.log(dy))
        return dist

    def calc_phi(self, Xi:List[str], Xj:List[str], Xk:List[str]):
        # Check rank instead?
        A = self.calc_infdist(Xi, Xk)
        B = self.calc_infdist(Xj, Xk)
        if A == None or B == None:
            return None
        return A-B

    def calc_phi_all(self, Xi:str, Xj:str):
        for X in self.xvars:
            if X == Xi or X == Xj:
                pass
            else:
                print(f"{X}: {self.calc_phi(Xi, Xj, X)}")

    def generate_data(self, varlist, n):
        df_subset = self.jpd.sample(n=n, weights="p", replace=True)
        df_subset = df_subset.loc[:, varlist]
        return df_subset

    def rank_test(self, A:List[str], B:List[str]):
        A = list(A)
        B = list(B)
        C = A+B

        # Zero out non-involved variables and aggregate
        indices = [self.varToBitIdx(v) for v in C]
        mask = self.makeMask(indices)
        self.jpd["tempkey"] = (self.jpd["key"] & mask)
        Jab = self.jpd.groupby("tempkey")["p"].sum().reset_index()

        # Create variables
        for v in C:
            bitIdx = self.varToBitIdx(v)
            Jab[v] = self.getValueAtBitIdx(Jab["tempkey"], bitIdx)

        Jab = Jab.pivot(index=A, columns=B, values="p")
        Jab = np.array(Jab)

        return math.log2(np.linalg.matrix_rank(Jab))

    # Park this idea here for now, using hyperdeterminant to get vanishing 
    # ranks
    def hyperdet(self, A:str, B: str, C:str):
        Jpd = self.jpd.groupby([A, B, C]).sum().reset_index()
        X = np.zeros(shape = (2,2,2))
        for a in Jpd[A].unique().tolist():
            for b in Jpd[B].unique().tolist():
                for c in Jpd[C].unique().tolist():
                    ai=int(a); bi=int(b); ci=int(c)
                    X[ai,bi,ci] = \
                      Jpd.query(f"({A}=='{a}')&({B}=='{b}')&({C}=='{c}')")["p"]
        det = sq(X[0,0,0]) * sq(X[1,1,1]) + sq(X[0,0,1]) * sq(X[1,1,0]) + \
              sq(X[0,1,0]) * sq(X[1,0,1]) + sq(X[1,0,0]) * sq(X[0,1,1]) - \
              2*X[0,0,0] * X[0,0,1] * X[1,1,0] * X[1,1,1] - \
              2*X[0,0,0] * X[0,1,0] * X[1,0,1] * X[1,1,1] - \
              2*X[0,0,0] * X[0,1,1] * X[1,0,0] * X[1,1,1] - \
              2*X[0,0,1] * X[0,1,0] * X[1,0,1] * X[1,1,0] - \
              2*X[0,0,1] * X[0,1,1] * X[1,1,0] * X[1,0,0] - \
              2*X[0,1,0] * X[0,1,1] * X[1,0,1] * X[1,0,0] + \
              4*X[0,0,0] * X[0,1,1] * X[1,0,1] * X[1,1,0] + \
              4*X[0,0,1] * X[0,1,0] * X[1,0,0] * X[1,1,1]
        print(det)

    def scenarioA(self): 
        self.add_variable("L3", None)
        self.add_variable("L4", None)
        self.add_variable("X7", ["L3", "L4"])
        self.add_variable("X6", "L4")
        self.add_variable("X5", "L3")
        self.add_variable("L1", ["L3", "L4"])
        self.add_variable("L2", ["L3", "L4"])
        self.add_variable("X1", "L1")
        self.add_variable("X2", "L1")
        self.add_variable("X3", "L2")
        self.add_variable("X4", ["L1", "L2"])


    def scenarioB(self): 
        self.add_variable("L1", None)
        self.add_variable("L2", None)
        self.add_variable("L3", "L1")
        self.add_variable("L4", "L2")
        self.add_variable("X1", ["L1", "L2"])
        self.add_variable("X2", ["L1", "L2"])
        self.add_variable("X3", ["L1", "L2"])
        self.add_variable("X4", ["L3", "L4"])
        self.add_variable("X5", ["L3", "L4"])
        self.add_variable("X6", ["L3", "L4"])

    def scenarioPyramid(self): 
        self.add_variable("L1", None)
        self.add_variable("L2", "L1")
        self.add_variable("L3", ["L1", "L2"])
        self.add_variable("L4", ["L1", "L2"])
        self.add_variable("L8", "L3")
        self.add_variable("L9", "L3")
        self.add_variable("L10", "L4")
        self.add_variable("L11", "L4")

        self.add_variable("X1", "L1")
        self.add_variable("X2", "L1")
        self.add_variable("X5", "L2")
        self.add_variable("X6", "L2")
        self.add_variable("X7", "L8")
        self.add_variable("X8", "L8")
        self.add_variable("X9", "L9")
        self.add_variable("X10", ["L9", "L10"])
        self.add_variable("X11", "L10")
        self.add_variable("X12", "L10")
        self.add_variable("X13", "L11")
        self.add_variable("X14", "L11")


    # llist should be a list of latent variable frozensets
    def find_pure_clusters(self):
        V = set(self.xvars)
        G = LatentGroups(V)
        k = 2
        l = 1
        for run in range(2):
            k = 2
            sufficientActiveVars = True
            while sufficientActiveVars:
                print(f"{'='*10} k is {k} {'='*10}")
                for j in range(k-1):
                    print(f"{'-'*10} j is {j} {'-'*10}")
                    for Ap in generateSubset(G.V["latent"], j):
                        for Bp in G.pickActiveVars(k-j):
                            
                            if setLength(Bp) < k-j:
                                set_trace()
                                sufficientActiveVars = False
                                break

                            # C is get all other active vars
                            C = G.V["active"] - Bp
                            A = G.pick1XperL(Ap)
                            B = G.pick1XperL(Bp)
                            AB = A.union(B)
                            AB_fset = set(tuple([frozenset([x]) for x in AB]))
                            C = G.pick1XperL(C, AB_fset)

                            # Add in Latents in control set
                            latentControls = G.pick1XperL(G.V["latent"], AB_fset) 
                            C = C.union(latentControls)

                            if len(C) < len(AB):
                                print("Control set is too small!")
                                continue 

                            rankCheck = self.rank_test(AB, C)

                            if rankCheck == k-1:
                                print(f"Cluster found!: {Bp} with {Ap}")
                                G.addTempGroup(Ap, Bp)

                        G.mergeTempGroups()
                        G.confirmTempGroups()

                        # Break out of Ap loop
                        if not sufficientActiveVars:
                            break
                pprint(G.d)
                k+=1

            print("End of one cycle")
            G.V["active"].update(G.V["latent"])
            G.V["latent"] = set()

def pprint(d):

    def fsetToText(fset):
        l = [x for x in iter(fset)]
        return ",".join(l)

    for parents, v in d.items():
        subgroups = v["subgroups"]
        children = v["children"]
        parents = f"[{','.join([x for x in parents])}]"
        children = ",".join([fsetToText(fset) for fset in children])

        text = f"{parents} : {children}"
        if len(subgroups) > 0:
            text += " | "
            for subgroup in subgroups:
                text += "["
                text += ",".join([x for x in subgroup])
                text += "] "
        print(text)


def setLength(varset):
    if isinstance(varset, str):
        return 1

    n = 0
    for vset in varset:
        if isinstance(vset, frozenset):
            n += len(vset)
        else:
            n += 1
    return n

class LatentGroups():
    def __init__(self, V):
        self.d = {}
        self.dTemp = {}
        self.maxL = 1
        self.V = {"latent": set(), 
                  "active": set([frozenset([v]) for v in V])}

    # llist: list of frozensets of latent variables
    def addTempGroup(self, llist, B):

        # Create new Latent Variables
        sizeL = setLength(B) - 1
        newLlist = []
        for i in range(sizeL):
            newLlist.append(f"L{self.maxL}")
            self.maxL += 1
        newParents = frozenset(newLlist)

        # Append to existing group
        for element in llist:
            newParents = newParents.union(element)

        # Create new entry
        self.dTemp[newParents] = {
                "children": set(B),
                "subgroups": llist
                }


    # Merge overlapping groups in dTemp
    def mergeTempGroups(self):
        if len(self.dTemp) == 0:
            return

        # Each parents is a frozenset
        # Values of inv_list becomes a list of frozensets
        inv_list = {}
        for parents, values in self.dTemp.items():
            children = values["children"]
            for child in children:
                parentList = inv_list.get(child, []) + [parents]
                inv_list[child] = parentList
        groups = []
        print("loc1")

        # Resolve the merging of latent variables
        # 1. If one X is a child of two different groups, merge them
        # 2. How many vars to merge?
        # 3. Always merge the larger latent vars?

        # Each parentList is a set of frozensets
        for parentList in inv_list.values():
            parentList = set(tuple(parentList))
            if len(parentList) > 1:
                if len(groups) == 0:
                    groups.append(parentList)
                    continue
    
                for i, group in enumerate(groups):
                    print("loc2")
                    if len(parentList.intersection(group)):
                        groups[i] = groups[i].union(parentList)
                    else:
                        groups.append(parentList)

        print("loc3")

        # groups is now a list of sets
        # Each element in each set is a frozenset of latent vars
        # Need to do a pairwise merge
        if len(groups) > 0:
            for group in groups:
                mergeDict = self.mergeGroup(group)
                for oldkey, newkey in mergeDict.items():
                    v = self.dTemp.pop(oldkey)
                    if not newkey in self.dTemp:
                        self.dTemp[newkey] = {}
                        self.dTemp[newkey]["children"] = set()
                        self.dTemp[newkey]["subgroups"] = set()
                    self.dTemp[newkey]["children"].update(v["children"])
                    self.dTemp[newkey]["subgroups"].update(v["subgroups"])

    # findMergeableVars: Given a set of frozensets of latent vars, find
    #                    subgroups which are not already identified latent
    #                    variables.
    def findMergeableVars(self, A):
        A1 = set() # Non-mergeable
        A2 = set() # Mergeable
        for a in A:
            if a in self.d:
                A1.update([a])
            else:
                A2.update([a])
        return A1, A2

    # mergeGroup: return a 1-1 mapping from original group to new group
    # args:
    #     group: a list of sets of frozensets of latent vars
    def mergeGroup(self, group):
        mergeDict = {}

        # Find lowest keys
        k = len(next(iter(group)))
        nums = []
        for A in group:
            A1, A2 = self.findMergeableVars(A)
            Anums = [int(x[1:]) for x in A2]
            nums.extend(Anums)
        lowestKeys = sorted(nums)[0:k]
        lowestKeys = frozenset([f"L{x}" for x in lowestKeys])

        # Create new keys
        for A in group:
            A1, A2 = self.findMergeableVars(A)
            newkey = A1.union(lowestKeys)
            mergeDict[A] = frozenset(newkey)
        return mergeDict



    # Move elements in dTemp to d
    def confirmTempGroups(self):
        while len(self.dTemp) > 0:
            parent_set, child_set = self.dTemp.popitem()
            self.d[parent_set] = child_set

            # Update V by grouping variables
            self.V["active"] = setDifference(self.V["active"], 
                                    child_set["children"])
            self.V["latent"].add(parent_set)



    # Problem: Some usedXs appear in the result
    def pick1X(self, fset, usedXs=set()):
        A = set()
        v = self.d[fset]
        if len(v["subgroups"]) > 0:
            for sub_fset in v["subgroups"]:
                A.update(self.pick1X(sub_fset, usedXs))
        else:
            availableXs = v["children"] - usedXs
            X = next(iter(availableXs))
            A.add(next(iter(X)))
        return A



    # Recursive search for one X per latent var
    # llist: set of frozensets of latent variables. children can be looked up
    #        in self.d
    def pick1XperL(self, varlist, usedXs=set()):
        A = set()
        for vset in varlist:
            varnames = [str(v)[0] for v in vset]
            vartype = varnames[0]
            assert all([v == vartype for v in varnames]),\
                    "Mixed types in a vset!"

            if vartype == "X":
                assert len(vset) == 1, "X variable with len > 1!"
                A.add(next(iter(vset)))
            else:
                A.update(self.pick1X(vset, usedXs))
        return A


    # Take combinations of size num of measured vars after removing
    # those already used in A
    def pickActiveVars(self, num):
        remaining_vars = self.V["active"]
        Blist = generateSubset(remaining_vars, num)

        # Active vars cannot purely be from the same cluster
        Blist = [B for B in Blist if len(B) > 1]
        if len(Blist) == 0:
            return [set()]
        return Blist

# Take difference between sets of fsets
# If any fset in setA is a subset of any fset in setB, remove it too
def setDifference(setA, setB):
    diff = setA - setB # first remove any common elements
    newset = set()
    while len(diff) > 0:
        fsetA = diff.pop()
        newset.add(fsetA)
        for fsetB in setB:
            if fsetA <= fsetB:
                newset.remove(fsetA)
    return newset

# generateSubset: Generate set of frozensets of variables s.t. the resultant
#                 list of has dim(list) = j
# args:
#     vset: set of frozensets of variables
def generateSubset(vset, j):

    def recursiveSearch(d, gap, currSubset=set()):
        thread = f"currSubset: {currSubset}, d: {d}, gap is {gap}"
        d = deepcopy(d)
        currSubset = deepcopy(currSubset)
        llist = []

        # Terminate if empty list
        if len(d) == 0:
            return llist

        # Pop latent sets larger than current gap
        maxDim = max(d)
        while maxDim > gap:
            d.pop(maxDim)
            maxDim = max(d)

        # Pop one element
        newGroup = d[maxDim].pop()
        if len(d[maxDim]) == 0:
            d.pop(maxDim)

        # Branch to consider all cases
        # Continue current search without this element
        if len(d) > 0:
            llist.extend(recursiveSearch(d, gap, currSubset))

        # Terminate branch if newGroup overlaps with currSubset
        if groupInLatentSet(newGroup, currSubset):
            return llist

        gap -= maxDim
        currSubset.add(newGroup)

        # Continue search if gap not met
        if gap > 0 and len(d) > 0:
            llist.extend(recursiveSearch(d, gap, currSubset))

        # End of search tree
        if gap == 0:
            llist.append(currSubset)

        return llist

    if j == 0:
        return [set()]

    # Create dictionary where key is dimension size and v is a list 
    # of frozensets of variables
    d = {}
    for fset in vset:
        if isinstance(fset, str):
            fset = frozenset([fset])
        l = len(fset)
        d[l] = d.get(l, []) + [fset]

    # Run recursive search
    result = recursiveSearch(d, j)
    if len(result) == 0:
        return [set()]
    else:
        return result


# Unpack all elements in a frozenset into a list of strings
def unpack(fset): 
    l = []
    for item in fset:
        if isinstance(item, frozenset):
            l.extend(unpack(item))
        else:
            l.append(item)
    return l



# Check if new group of latent vars exists in a current
# list of latent vars
def groupInLatentSet(newGroup: frozenset, currSubset: set):
    for group in currSubset:
        if len(newGroup.intersection(group)) > 0:
            return True
    return False

def lprint(l):
    return ",".join(sorted(l))

# Check if a list of phi values are all close
def check_list_close(l):
    if len(l) == 0:
        return False

    first_element = l[0]
    for phi in l:
        if phi is None:
            return False
        if not math.isclose(first_element, phi):
            return False
    return True



# Checks if a set of values is a subset of any clusters in S
def check_subset(A, S):
    A = set(A)
    for s in S.values():
        if A.issubset(s):
            return True
    return False


if __name__ == "__main__":

    g = Graph()
    g.add_variable("X1", None)
    g.add_variable("X3", None)
    g.add_variable("X2", None)
    g.add_variable("X4", ["X1", "X2", "X3"])
    g.add_variable("X5", ["X2", "X4"])
    print(math.pow(2, g.rank_test(["X1", "X3"], ["X4", "X5"])))
    #g.scenarioPyramid()
    #g.find_pure_clusters()

