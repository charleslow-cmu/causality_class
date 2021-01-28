import random
import pandas as pd
import numpy as np
from itertools import product
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
from scipy.stats import norm
from scipy.stats.stats import pearsonr
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
            valid_parents = len(set(parents)-set(self.lvars)) == 0
            assert valid_parents, "Parents not found in graph!"
        elif "X" in name:
            assert False, "X variable must have a parent!"

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
    def get_joint(self, X: str, Y: str):
        Vs = [X, Y]
        Jxy = self.jpd.groupby(Vs)["p"].sum().reset_index()
        Jxy = Jxy.pivot(index=X, columns=Y, values="p")
        return Jxy

    def calc_infdist(self, X: str, Y: str):
        Jxy = self.jpd.groupby([X, Y])["p"].sum().reset_index()
        Jxy = Jxy.pivot(index=X, columns=Y, values="p")
        Jxy = np.array(Jxy)
        dxy = abs(np.linalg.det(Jxy))
        dx = Jxy.sum(axis=1).prod()
        dy = Jxy.sum(axis=0).prod()
        dist = -math.log(dxy) + 0.5*(math.log(dx) + math.log(dy))
        #print(f"Dist {X} to {Y} is {dist:.8f}")
        return dist

    def calc_infdist2(self, X: str, Y: str):
        Jxy = self.jpd.groupby([X, Y])["p"].sum().reset_index()
        Jxy = Jxy.pivot(index=X, columns=Y, values="p")
        Jxy = np.array(Jxy)
        u, d, v = np.linalg.svd(Jxy)
        tol = d.max() * max(Jxy.shape) * sys.float_info.epsilon
        d = d[d > tol]
        #print(f"Singular P_{X}{Y}: {d}")
        dx = Jxy.sum(axis=1).prod()
        dy = Jxy.sum(axis=0).prod()
        dist = -np.sum(np.log(d)) + 0.5*(math.log(dx) + math.log(dy))
        return dist

    def calc_phi(self, Xi:str, Xj:str, Xk:str):
        return self.calc_infdist(Xi, Xk) - self.calc_infdist(Xj, Xk)

    def calc_mi(self, X: str, Y: str, log=False, normalize=False):
        mi = 0
        xvals = pd.unique(self.jpd[X])
        yvals = pd.unique(self.jpd[Y])
        for x in xvals:
            for y in yvals:
                Pxy = lookup(self.jpd, {X: x, Y: y})
                Px = lookup(self.jpd, {X: x})
                Py = lookup(self.jpd, {Y: y})
                temp = math.log(Pxy) - math.log(Px) - math.log(Py)
                mi += (Pxy * temp)

        if normalize:
            Hx = 0
            for x in xvals:
                Px = lookup(self.jpd, {X:x})
                Hx += (-Px * math.log(Px))
            Hy = 0
            for y in yvals:
                Py = lookup(self.jpd, {Y:y})
                Hy += (-Py * math.log(Py))
            mi /= (Hx * Hy)

        if mi <= 0:
            mi = sys.float_info.min
        if log:
            mi = math.log(mi)
        return mi    


    def calc_jentropy(self, X: str, Y: str, log=False):
        ent = 0
        xvals = pd.unique(self.jpd[X])
        yvals = pd.unique(self.jpd[Y])
        for x in xvals:
            for y in yvals:
                Pxy = lookup(self.jpd, {X: x, Y: y})
                ent += (-Pxy * math.log(Pxy))
        if ent <= 0:
            ent = sys.float_info.min
        if log:
            ent = math.log(ent)
        return ent


    def calc_condmi(self, X: str, Y: str, Z: str, normalize=False):
        mi = 0
        combn = [[0,1]] * 3
        for x,y,z in product(*combn):
            Pxyz = lookup(self.jpd, {X: x, Y: y, Z: z})
            Pz = lookup(self.jpd, {Z: z})
            Pxz = lookup(self.jpd, {X: x, Z: z})
            Pyz = lookup(self.jpd, {Y: y, Z: z})
            mi += Pxyz * math.log((Pxyz*Pz) / (Pxz * Pyz))
        print(f"MI({X};{Y}|{Z}) = {mi:.10f}")
        return mi

    def calc_entropy(self, X: str):
        entropy = 0
        for x in [0,1]:
            Px = lookup(self.jpd, {X: x})
            entropy += Px * math.log(1/Px)
        return entropy

    def generate_data(self, varlist, n):
        df_subset = self.jpd.sample(n=n, weights="p", replace=True)
        df_subset = df_subset.loc[:, varlist]
        return df_subset

def mutual_info(df, X:str, Y:str):
    xvals = pd.unique(df[X])
    yvals = pd.unique(df[Y])
    n = df.shape[0]
    mi = 0
    for x in xvals:
        for y in yvals:
            Pxy = (df.loc[(df[X] == x) & (df[Y] == y)].shape[0]+1) / (n+1)
            Px = (df.loc[(df[X] == x)].shape[0]+1) / (n+1)
            Py = (df.loc[(df[Y] == y)].shape[0]+1) / (n+1)
            temp = math.log(Pxy) - math.log(Px) - math.log(Py)
            mi += Pxy * temp
    if mi <= 0:
        mi = sys.float_info.min
    return mi    




if __name__ == "__main__":

    correct = 0
    n = 10
    for _ in range(n):
        g = Graph()
        g.add_variable("L1", None, 2)
        g.add_variable("X1", "L1", 15)
        g.add_variable("X2", "L1", 15)

        A = g.calc_infdist2("X1", "L1")
        B = g.calc_infdist2("X2", "L1")
        C = g.calc_infdist2("X1", "X2")
        correct += math.isclose(A+B, C, abs_tol=1e-6)
        print(f"{A+B},{C}")
        J_X1X2 = g.get_joint("X1", "X2")
        print(np.linalg.matrix_rank(J_X1X2))
    print(f"Correct: {correct}/{n}")

    #k=2
    #g = Graph()
    #g.add_variable("L1", None, k=k)
    #g.add_variable("L2", "L1", k=k)
    #g.add_variable("L3", ["L1", "L2"], k=k)
    #g.add_variable("L4", "L3", k=k)
    #g.add_variable("X1", "L1", k=k)
    #g.add_variable("X2", "L1", k=k)
    #g.add_variable("X3", "L2", k=k)
    #g.add_variable("X4", "L2", k=k)
    #g.add_variable("X5", ["L1", "L2"], k=k)
    #g.add_variable("X6", ["L1", "L2"], k=k)
    #g.add_variable("X7", "L3", k=k)
    #g.add_variable("X8", "L3", k=k)
    #g.add_variable("X9", "L4", k=k)
    #g.add_variable("X10", "L4", k=k)



