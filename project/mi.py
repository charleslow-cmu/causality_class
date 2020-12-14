import random
import pandas as pd
import numpy as np
from itertools import product
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm

def dicttostr(d):
    l = []
    for k,v in d.items():
        l.append(f"{k}:{v}")
    return f"|{','.join(l)}|"


def make_column(d=1):
    #probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    lprobs = []
    for i in range(d):
        p = random.uniform(0.05, 0.95)
        lprobs.append(p)
        lprobs.append(1-p)
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
# Replace all matching rows in df with "p" value
def replace(df: pd.DataFrame, d, p):
    vset = set(df.columns)
    l = []
    for k, v in d.items():
        if k in vset:
            l.append(f"({k} == {v})")
    idx = df.query(" & ".join(l)).index
    df.loc[idx, "p"] *= p


def make_jpd(jpd, parents, name):

    # Insert this variable into the jpd
    lrows = []
    dlist = jpd.to_dict(orient="records")
    for i in range(len(dlist)):
        for v in [0, 1]:
            d = deepcopy(dlist[i])
            d[name] = v
            lrows.append(d)
    df = pd.DataFrame(lrows)
    df = sort_columns(df)

    if parents is None:
        df["p"] *= make_column(d=int(df.shape[0]/2))
        return df

    # Make cpd from direct parents 
    vnames = [v for v in parents] + [name]
    vlen = len(vnames)
    keys = [[0,1]] * vlen
    lrows = []
    for ptuple in product(*keys):
        d = {vnames[i]:v for i,v in enumerate(ptuple)}
        lrows.append(d)
    cpd = pd.DataFrame(lrows)
    cpd["p"] = make_column(d=int(cpd.shape[0]/2))
    #print(cpd)

    # Replace the jpd ps with new ones
    dlist = cpd.to_dict(orient="records")
    for d in dlist:
        p = d.pop("p")
        replace(df, d, p)
    #print(df)
    return df


def sort_columns(df):
    l = ["L", "X", "p"]
    cols = []
    for letter in l:
        cols.extend([col for col in df.columns 
                        if col.startswith(letter)])
    return df.reindex(cols, axis=1)


class Graph:
    def __init__(self):
        self.vars = {}
        self.jpd = None

    def add_variable(self, name, parents=None):
        if isinstance(parents, str):
            parents = [parents]
        var = Variable(name, parents)

        if not parents is None:
            valid_parents = len(set(parents)-set(self.vars.keys())) == 0
            assert valid_parents, "Parents not found in graph!"

        self.vars[name] = var

        # Adding first variable
        if len(self.vars) == 1:
            d = {name: [0,1], "p": make_column(d=1)}
            self.jpd = pd.DataFrame(d)
            return

        # Adding subsequent vars
        self.jpd = make_jpd(self.jpd, parents, name)
        assert math.isclose(sum(self.jpd["p"]), 1), "JPD does not sum to 1"
        return


    def calc_mi(self, X: str, Y: str, normalize=False):
        mi = 0
        for x in [0, 1]:
            for y in [0, 1]:
                Pxy = lookup(self.jpd, {X: x, Y: y})
                Px = lookup(self.jpd, {X: x})
                Py = lookup(self.jpd, {Y: y})
                mi += Pxy * math.log(Pxy / (Px * Py))
    
        if normalize:
            Hx = 0
            for x in [0, 1]:
                Px = lookup(self.jpd, {X: x})
                Hx += -Px * math.log(Px)
            mi = mi / Hx

        #if mi < 0:
        #    mi = 1e-20
        #print(f"MI({X};{Y}) = {mi:.10f}")
        mi += 1e-20
        return mi    

    def calc_condmi(self, X: str, Y: str, Z: str, normalize=False):
        mi = 0
        combn = [[0,1]] * 3
        for x,y,z in product(*combn):
            Pxyz = lookup(self.jpd, {X: x, Y: y, Z: z})
            Pz = lookup(self.jpd, {Z: z})
            Pxz = lookup(self.jpd, {X: x, Z: z})
            Pyz = lookup(self.jpd, {Y: y, Z: z})
            mi += Pxyz * math.log((Pxyz*Pz) / (Pxz * Pyz))

        if normalize:
            Hx_z = 0
            combn = [[0,1]] * 2
            for x,z in product(*combn):
                Pxz = lookup(self.jpd, {X:x, Z:z})
                Px = lookup(self.jpd, {X:x})
                Hx_z += -Pxz * math.log(Pxz/Px)
            mi = mi / Hx_z
        print(f"MI({X};{Y}|{Z}) = {mi:.10f}")
        return mi

    def calc_joint(self, d):
        Pxy = lookup(self.jpd, d)
        return Pxy

    def calc_entropy(self, X: str):
        entropy = 0
        for x in [0,1]:
            Px = lookup(self.jpd, {X: x})
            entropy += Px * math.log(1/Px)
        return entropy

    def calc_tetrad(self, X1, X2, X3, X4):
        tetrad =  math.log(self.calc_mi(X1, X2))
        tetrad += math.log(self.calc_mi(X3, X4))
        return tetrad


class Variable:
    def __init__(self, name, parents=None):
        self.name = name
        self.parents = parents


def run_expt1a():
    # Expt 1
    n = 100
    l1 = []
    l2 = []
    for i in tqdm(range(n)):
        g = Graph()
        g.add_variable("L1", None)
        g.add_variable("X1", "L1")
        g.add_variable("X2", "L1")
        g.add_variable("X3", "L1")
        g.add_variable("X4", "L1")
        l1.append(g.calc_tetrad("X1", "X2", "X3", "X4"))
        l2.append(g.calc_tetrad("X1", "X4", "X2", "X3"))

    print(np.corrcoef(l1, l2))
    df = pd.DataFrame({"A": l1, "B": l2})
    plt.rcParams.update({'font.size': 15})
    df.plot.scatter(x="A", y="B", alpha=0.6, figsize=(12,8))
    plt.savefig("plots/expt1a.png")


def run_expt1b():
    n = 100
    l1 = []
    l2 = []
    for i in tqdm(range(n)):
        g = Graph()
        g.add_variable("L1", None)
        g.add_variable("L2", None)
        g.add_variable("X1", "L1")
        g.add_variable("X2", "L1")
        g.add_variable("X3", "L2")
        g.add_variable("X4", "L2")
        l1.append(g.calc_tetrad("X1", "X2", "X3", "X4"))
        l2.append(g.calc_tetrad("X1", "X4", "X2", "X3"))

    print(np.corrcoef(l1, l2))
    df = pd.DataFrame({"A": l1, "B": l2})
    plt.rcParams.update({'font.size': 15})
    df.plot.scatter(x="A", y="B", alpha=0.6, figsize=(12,8))
    plt.savefig("plots/expt1b.png")


def run_expt1c():
    n = 100
    l1 = []
    l2 = []
    for i in tqdm(range(n)):
        g = Graph()
        g.add_variable("L1", None)
        g.add_variable("L2", "L1")
        g.add_variable("X1", "L1")
        g.add_variable("X2", "L1")
        g.add_variable("X3", "L2")
        g.add_variable("X4", "L2")
        l1.append(g.calc_tetrad("X1", "X2", "X3", "X4"))
        l2.append(g.calc_tetrad("X1", "X4", "X2", "X3"))

    print(np.corrcoef(l1, l2))
    df = pd.DataFrame({"A": l1, "B": l2})
    plt.rcParams.update({'font.size': 15})
    df.plot.scatter(x="A", y="B", alpha=0.6, figsize=(12,8))
    plt.savefig("plots/expt1c.png")

def run_expt2():
    n = 100
    l1 = []
    l2 = []
    for i in tqdm(range(n)):
        g = Graph()
        g.add_variable("L1", None)
        g.add_variable("L2", None)
        g.add_variable("X1", "L1")
        g.add_variable("X2", "L1")
        g.add_variable("X3", "L2")
        g.add_variable("X4", "L2")
        A = g.calc_joint({"X1":0, "X2":0}) * g.calc_joint({"X3":0, "X4":0})
        B = g.calc_joint({"X1":0, "X4":0}) * g.calc_joint({"X2":0, "X3":0})
        l1.append(A)
        l2.append(B)
    print(np.corrcoef(l1, l2))
    df = pd.DataFrame({"A": l1, "B": l2})
    plt.rcParams.update({'font.size': 15})
    df.plot.scatter(x="A", y="B", alpha=0.6, figsize=(12,8))
    plt.savefig("plots/expt2.png")


# Is I(X1;L1) * I(X2;L1) * H(L1) ~= I(X1;X2)?
# Corr = 0.990
def run_expt3():
    n = 200
    l1 = []
    l2 = []
    for i in tqdm(range(n)):
        g = Graph()
        g.add_variable("L1", None)
        g.add_variable("X1", "L1")
        g.add_variable("X2", "L1")
        A = math.log(g.calc_mi("X1", "X2"))
        B = math.log(g.calc_entropy("L1"))
        B += math.log(g.calc_mi("L1", "X1"))
        B += math.log(g.calc_mi("L1", "X2"))
        l1.append(A)
        l2.append(B)
    print(np.corrcoef(l1, l2))
    df = pd.DataFrame({"A": l1, "B": l2})
    plt.rcParams.update({'font.size': 15})
    df.plot.scatter(x="A", y="B", alpha=0.6, figsize=(12,8))
    plt.savefig("plots/expt3.png")

if __name__ == "__main__":
    run_expt3()

