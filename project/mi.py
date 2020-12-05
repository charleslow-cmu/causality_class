import random
import pandas as pd
import numpy as np
from itertools import product
import math

def dicttostr(d):
    l = []
    for k,v in d.items():
        l.append(f"{k}:{v}")
    return f"|{','.join(l)}|"


def make_column(d=1):
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    lprobs = []
    for i in range(d):
        p = random.choice(probs)
        lprobs.append(p)
        lprobs.append(1-p)
    return pd.Series(lprobs)


# Given a dictionary of key and values, retrieve
# corresponding row from df
def lookup(var, d):
    if isinstance(var, Variable):
        df = var.cpd
    else:
        df = var

    vset = set(df.columns)
    for k, v in d.items():
        if k in vset:
            df = df.loc[df[k] == v]
    #print(f"Lookup of {var.name} with {dicttostr(d)} found {df.shape[0]} entries.")
    return sum(df["p"])


def make_jpd(vars):
    vset = set()
    for var in vars:
        for col in var.cpd.columns:
            if col != "p":
                vset.add(col)
    keys = [[0,1] for i in range(len(vset))]
    vnames = list(vset)
    vnames.sort()
    lrows = []
    for ptuple in product(*keys):
        d = {}
        for i, k in enumerate(ptuple):
            d[vnames[i]] = k
        lrows.append(d)
    df = pd.DataFrame(lrows)
    df["p"] = None 

    for i in range(df.shape[0]):
        lprobs = []
        for var in vars:
            row = dict(df.iloc[i, :])
            row.pop("p")
            lprobs.append(lookup(var, row))
        df.loc[i, "p"] = np.prod(lprobs)
    return df

def sort_columns(df):
    l = ["L", "X", "p"]
    cols = []
    for letter in l:
        cols.extend([col for col in df.columns 
                        if col.startswith(letter)])
    return df.reindex(cols, axis=1)


class Variable:
    def __init__(self, name, causes=None):
        if isinstance(causes, Variable):
            causes = [causes]
        self.name = name
        self.causes = causes
        self.make_cpd(name, causes)

    def make_cpd(self, name, causes):
        if causes is None:
            df = pd.DataFrame({name: [0,1]})
            df["p"] = make_column(1)
            self.cpd = df
            return

        jpd = make_jpd(causes)
        lrows = []
        for i in range(jpd.shape[0]):
            for v in [0,1]:
                jpd_row = dict(jpd.iloc[i,:])
                jpd_row[name] = v
                lrows.append(jpd_row)
        df = pd.DataFrame(lrows)
        df = sort_columns(df)
        df["p"] *= make_column(d=int(df.shape[0]/2))
        self.cpd = df


def calc_mi(X: Variable, Y: Variable):
    jpd = make_jpd([X, Y])
    mi = 0
    for x in [0, 1]:
        for y in [0, 1]:
            row = jpd.loc[(jpd[X.name] == x) & (jpd[Y.name] == y), :]
            Pxy = lookup(jpd, {X.name: x, Y.name: y})
            Px = lookup(jpd, {X.name: x})
            Py = lookup(jpd, {Y.name: y})
            mi += Pxy * math.log(Pxy / (Px * Py))
    print(mi)
    return mi

def calc_condmi(X:Variable, Y:Variable, Z:Variable):
    jpd = make_jpd([X, Y, Z])
    mi = 0
    combn = [[0,1]] * 3
    for x,y,z in product(*combn):
        Pxyz = lookup(jpd, {X.name: x, Y.name: y, Z.name: z})
        Pz = lookup(jpd, {Z.name: z})
        Pxz = lookup(jpd, {X.name: x, Z.name: z})
        Pyz = lookup(jpd, {Y.name: y, Z.name: z})
        mi += Pxyz * math.log((Pxyz*Pz) / (Pxz * Pyz))
    print(mi)
    return mi


L1 = Variable("L1", None)
L2 = Variable("L2", None)
X1 = Variable("X1", L1)
X2 = Variable("X2", L1)
X3 = Variable("X3", L2)
X4 = Variable("X4", L2)
#import IPython; IPython.embed(); exit(1)
#mi = calc_mi(X1, X2)
#mi = calc_mi(X1, X3)
mi = calc_condmi(X1, X2, X3)
mi = calc_condmi(X1, X2, X4)
mi = calc_condmi(X3, X4, X1)
mi = calc_condmi(X3, X4, X2)
#import IPython; IPython.embed(); exit(1)

