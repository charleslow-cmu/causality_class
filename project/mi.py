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

# Return list of items in first that are not in second
def listdiff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

def make_keys(k=1):
    return [str(i) for i in range(k+1)]

# Add 'key' column based on unique combi of keys
def add_key_column(df, vlist):
    df['key'] = df.loc[:, vlist].agg('-'.join, axis=1)
    return df


def dicttostr(d):
    l = []
    for k,v in d.items():
        l.append(f"{k}:{v}")
    return f"|{','.join(l)}|"


# Randomly sample the conditional distribution
def make_column(k=1):
    lprobs = []
    remainder = 1
    for j in range(k):
        p = random.uniform(0.05, 0.95) * remainder
        lprobs.append(p)
        remainder -= p
    lprobs.append(remainder)
    return pd.Series(lprobs)


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


def make_jpd(jpd, parents, name, k=1):

    # Insert this variable into the jpd
    lrows = []
    dlist = jpd.to_dict(orient="records")
    for i in range(len(dlist)):
        for v in make_keys(k):
            d = deepcopy(dlist[i])
            d[name] = v
            lrows.append(d)
    df = pd.DataFrame(lrows)
    df = sort_columns(df)

    # No parents case
    if parents is None:
        vlist = list(jpd.drop("p", axis=1).columns)
        df = add_key_column(df, vlist)
        p = make_column(k=k)
        clist = []
        for key in pd.unique(df['key']):
            clist.append(p)
        column = pd.concat(clist, ignore_index=True)
        df["p"] *= column
        df = df.drop("key", axis=1)

    else:
        # Make cpd from direct parents 
        df_subset = df.loc[:, parents + [name]]
        df_subset = df_subset.drop_duplicates().reset_index(drop=True)
        df_subset = add_key_column(df_subset, parents)
        clist = []
        for key in pd.unique(df_subset['key']):
            p = make_column(k=k)
            clist.append(p)
        column = pd.concat(clist, ignore_index=True)
        df_subset["p"] = column
        df_subset = df_subset.drop("key", axis=1)

        # Replace the jpd ps with new ones
        dlist = df_subset.to_dict(orient="records")
        for d in dlist:
            p = d.pop("p")
            replace(df, d, p)
    return df


def sort_columns(df):
    l = ["L", "X", "p", "key"]
    cols = []
    for letter in l:
        cols.extend([col for col in df.columns 
                        if col.startswith(letter)])
    return df.reindex(cols, axis=1)


class Graph:
    def __init__(self):
        self.xvars = []
        self.lvars = []
        self.jpd = None

    # k is number of categories for this variable
    def add_variable(self, name, parents=None, k=2):
        if isinstance(parents, str):
            parents = [parents]

        # Sanity checks
        if not parents is None:
            variables = set(self.lvars) | set(self.xvars)
            valid_parents = len(set(parents)-variables) == 0
            assert valid_parents, "Parents not found in graph!"

        if "X" in name:
            self.xvars.append(name)
        else:
            self.lvars.append(name)

        # Adding first variable
        if len(self.lvars) + len(self.xvars) == 1:
            d = {name: make_keys(k-1), "p": make_column(k=k-1)}
            self.jpd = pd.DataFrame(d)
            return

        # Adding subsequent vars
        self.jpd = make_jpd(self.jpd, parents, name, k=k-1)
        assert math.isclose(sum(self.jpd["p"]), 1), "JPD does not sum to 1"

    # Get Marginal Probability P(X)
    def get_marginal(self, X: str):
        Dx = self.jpd.groupby(X)["p"].sum().reset_index()
        return Dx

    # Get Conditional Probability P(X|Ys)
    def get_conditional(self, X: str, Ys: List[str]):
        if isinstance(Ys, str):
            Ys = [Ys]
        Vs = Ys + [X]
        Jxy = self.jpd.groupby(Vs)["p"].sum().reset_index()
        Jxy["p"] = Jxy.groupby(Ys)["p"].apply(lambda x: x/x.sum())
        Jxy = Jxy.pivot(index=Ys, columns=X, values="p")
        return Jxy

    # Get Joint Probability P(X,Y)
    def get_joint(self, Xs: List[str], Ys: List[str]):
        if isinstance(Xs, str):
            Xs = [Xs]
        if isinstance(Ys, str):
            Ys = [Ys]
        Vs = Xs + Ys
        Jxy = self.jpd.groupby(Vs)["p"].sum().reset_index()
        Jxy = Jxy.pivot(index=Xs, columns=Ys, values="p")
        return Jxy

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
        Jab = self.jpd.groupby(A+B)["p"].sum().reset_index()
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
        self.add_variable("L4", "L2")
        self.add_variable("L5", "L1")
        self.add_variable("L6", "L1")
        self.add_variable("L7", "L2")
        self.add_variable("L8", "L3")
        self.add_variable("L9", "L3")
        self.add_variable("L10", "L4")
        self.add_variable("L11", "L4")

        self.add_variable("X1", "L5")
        self.add_variable("X2", ["L5", "L6"])
        self.add_variable("X3", "L6")
        self.add_variable("X4", "L7")
        self.add_variable("X5", "L7")
        self.add_variable("X6", ["L2", "L7"])
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
                    for llist in generateSubset(G.V["latent"], j):
                        for B in G.pickActiveVars(k-j):
                            
                            if len(B) < k-j:
                                sufficientActiveVars = False
                                break

                            # C is get all other active vars
                            C = G.V["active"] - B
                            A = G.pick1XperL(llist)
                            B = G.pick1XperL(B)
                            AB = A.union(B)
                            C = G.pick1XperL(C, AB)

                            if run > 0:
                                set_trace()

                            # Add in Latents in control set
                            AB_fset = set(tuple([frozenset([x]) for x in AB]))
                            latentControls = G.pick1XperL(G.V["latent"], AB_fset) 
                            C = C.union(latentControls)

                            if len(C) < len(AB):
                                print("Control set is too small!")
                                continue 

                            try:
                                rankCheck = self.rank_test(AB, C)
                            except AttributeError:
                                set_trace()

                            if rankCheck == k-1:
                                print(f"Cluster found!: {B} with {llist}")
                                G.addTempGroup(llist, B)

                        G.mergeTempGroups()
                G.confirmTempGroups()
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
        sizeL = len(B) - 1
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
                "children": set([frozenset([b]) for b in B]),
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
            self.V["active"] = self.V["active"] - child_set["children"]
            self.V["latent"].add(parent_set)


    # Generate list of groups of latent variables s.t. the resultant
    # list of La has dim(La) = j
    #def generateLatentSubset(self, j):

    #    def recursiveSearch(d, gap, currSubset=[]):
    #        thread = f"currSubset: {currSubset}, d: {d}, gap is {gap}"
    #        d = deepcopy(d)
    #        currSubset = deepcopy(currSubset)
    #        llist = []

    #        # Terminate if empty list
    #        if len(d) == 0:
    #            return llist

    #        # Pop latent sets larger than current gap
    #        maxDim = max(d)
    #        while maxDim > gap:
    #            d.pop(maxDim)
    #            maxDim = max(d)

    #        # Pop one element
    #        newGroup = d[maxDim].pop()
    #        if len(d[maxDim]) == 0:
    #            d.pop(maxDim)

    #        # Branch to consider all cases
    #        # Continue current search without this element
    #        if len(d) > 0:
    #            llist.extend(recursiveSearch(d, gap, currSubset))

    #        # Terminate branch if newGroup overlaps with currSubset
    #        if groupInLatentSet(newGroup, currSubset):
    #            return llist

    #        gap -= maxDim
    #        currSubset.append(newGroup)

    #        # Continue search if gap not met
    #        if gap > 0 and len(d) > 0:
    #            llist.extend(recursiveSearch(d, gap, currSubset))

    #        # End of search tree
    #        if gap == 0:
    #            llist.append(currSubset)

    #        return llist

    #    if j == 0:
    #        return [[]]

    #    result = recursiveSearch(self.dimsDict, j)
    #    if len(result) == 0:
    #        return [[]]
    #    else:
    #        return result

    # fset: frozenset of latent variables
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
        return Blist


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
    g.scenarioPyramid()
    g.find_pure_clusters()

