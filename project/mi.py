import random
import pandas as pd
import numpy as np
from itertools import product, combinations
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import List
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'xtick.labelsize': 12})
plt.rcParams.update({'ytick.labelsize': 12})
plt.rcParams.update({'axes.labelsize': 18})
import seaborn as sns
from tqdm import tqdm
import sys

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
        Jab = self.jpd.groupby(A+B)["p"].sum().reset_index()
        Jab = Jab.pivot(index=A, columns=B, values="p")
        Jab = np.array(Jab)
        return math.log2(np.linalg.matrix_rank(Jab))

    def scenarioA(self): 
        self.add_variable("L3", None)
        self.add_variable("L4", None)
        self.add_variable("X6", "L3")
        self.add_variable("X7", "L4")
        self.add_variable("L1", ["L3", "L4"])
        self.add_variable("L2", ["L3", "L4"])
        self.add_variable("X1", "L1")
        self.add_variable("X2", "L1")
        self.add_variable("X5", "L1")
        self.add_variable("X3", "L2")
        self.add_variable("X4", ["L1", "L2"])


    def scenarioB(self): 
        self.add_variable("L1", None)
        self.add_variable("L2", None)
        self.add_variable("L3", "L1")
        self.add_variable("L4", "L4")
        self.add_variable("X1", ["L1", "L2"])
        self.add_variable("X2", ["L1", "L2"])
        self.add_variable("X3", ["L1", "L2"])
        self.add_variable("X4", ["L3", "L4"])
        self.add_variable("X5", ["L3", "L4"])
        self.add_variable("X6", ["L3", "L4"])


    def find_pure_clusters(self):
        l = 1
        V = set(self.xvars)
        S = {}
        k = 2
        while k<3:
            temp_S = {}
            print(f"{'='*10} k is {k} {'='*10}")
            for test_vars in combinations(V, k*2):
                A = list(test_vars[:k])
                B = list(test_vars[k:])
                #if check_subset(A, S) or check_subset(B, S):
                #    continue

                # Check if remaining_vars is a valid reference set
                remaining_vars = list(V - set(test_vars))
                rankCheck = self.rank_test(A+B, remaining_vars)
                print(f"{lprint(A+B)} vs {lprint(remaining_vars)}: {rankCheck}")
                if rankCheck <= k:
                    continue
                
                phi_list = []
                for C in combinations(remaining_vars, k):
                    A, B, C = list(A), list(B), list(C)
                    phi_list.append(g.calc_phi(A, B, C))
                    #if set(A+B) == set(["X4", "X5", "X3", "X6"]):
                    #    print(f"AC: {lprint(A+C)}| {g.calc_infdist(A, C)}")
                    #    print(f"BC: {lprint(B+C)}| {g.calc_infdist(B, C)}")
                    #    print(f"A:{A}, B:{B}, C:{C} | {phi_list[-1]}")

                if check_list_close(phi_list):
                    print(f"Found cluster: {A+B}")
                    print(phi_list)
                    L = f"L{l}"
                    temp_S[L] = A+B
                    l+=1

            S = merge_clusters(temp_S)
            #for k, vs in S.items():
            #    V = V - set(vs)
            #    V.add(k)
            print(S)
            print(V)
            k+=1

def lprint(l):
    return ",".join(l)

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


# Given a dictionary of clusters, merge overlapping clusters
# of the same cardinality.
def merge_clusters(S):
    inv_list = {}
    for k, vs in S.items():
        for v in vs:
            inv_list[v] = inv_list.get(v, []) + [k]

    groups = []
    for vs in inv_list.values():
        if len(vs) > 1:
            if len(groups) == 0:
                groups.append(set(vs))

            for i, group in enumerate(groups):
                if len(group.intersection(vs)) > 0:
                    groups[i].update(vs)
                else:
                    groups.append(set(vs))

    for group in groups:
        new_key = min(group)
        new_vals = set()
        for l in group:
            new_vals.update(S.pop(l))
        S[new_key] = list(new_vals)

    return S

# Checks if a set of values is a subset of any clusters in S
def check_subset(A, S):
    A = set(A)
    for s in S.values():
        if A.issubset(s):
            return True
    return False


if __name__ == "__main__":

    k=2
    g = Graph()
    g.scenarioA()

    g.find_pure_clusters()
