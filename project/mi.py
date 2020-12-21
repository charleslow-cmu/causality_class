import random
import pandas as pd
import numpy as np
from itertools import product
import math
from copy import deepcopy
import matplotlib.pyplot as plt
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
        self.vars = set()
        self.jpd = None

    # k is number of categories for this variable
    def add_variable(self, name, parents=None, k=1):
        if isinstance(parents, str):
            parents = [parents]

        if not parents is None:
            valid_parents = len(set(parents)-self.vars) == 0
            assert valid_parents, "Parents not found in graph!"

        self.vars.add(name)

        # Adding first variable
        if len(self.vars) == 1:
            d = {name: make_keys(k), "p": make_column(k=k)}
            self.jpd = pd.DataFrame(d)
            return

        # Adding subsequent vars
        self.jpd = make_jpd(self.jpd, parents, name, k=k)
        assert math.isclose(sum(self.jpd["p"]), 1), "JPD does not sum to 1"
        return


    def calc_mi(self, X: str, Y: str):
        mi = 0
        xvals = pd.unique(self.jpd[X])
        yvals = pd.unique(self.jpd[Y])
        for x in xvals:
            for y in yvals:
                Pxy = lookup(self.jpd, {X: x, Y: y})
                Px = lookup(self.jpd, {X: x})
                Py = lookup(self.jpd, {Y: y})
                temp = math.log(Pxy) - math.log(Px) - math.log(Py)
                mi += Pxy * temp
        if mi <= 0:
            mi = sys.float_info.min
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


def tetrad_test(df, X1:str, X2:str, X3:str, X4:str, trials=1000):
    bootstrap_prob = 1.0
    lstats = []
    for i in range(trials):
        s= df.sample(frac=bootstrap_prob, replace=True)
        A = math.log(mutual_info(s, X1, X2)) + math.log(mutual_info(s, X3, X4))
        B = math.log(mutual_info(s, X1, X4)) + math.log(mutual_info(s, X2, X3))
        lstats.append(abs(A-B))
    sns.distplot(lstats)
    lstats = pd.Series(lstats)
    test_stat = np.mean(lstats)
    std = np.std(lstats)
    pval = 2*(1-norm.cdf(abs(test_stat), loc=0, scale=std))
    return pval


def run_expt1(n=100, ldim=1):

    def plot_graph(l1, l2, l3, l4, r1, r2, filename):
        print(r1); print(r2)
        df = pd.DataFrame({"A": l1, "B": l2, "C": l3, "D": l4})
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        df.plot.scatter(x="A", y="B", alpha=0.6, figsize=(6,8), ax=ax1)
        ax1.text(0.1, 0.8, f"Pearson R={r1:.5f}", transform=ax1.transAxes, family='monospace')
        df.plot.scatter(x="C", y="D", alpha=0.6, ax=ax2)
        ax2.text(0.1, 0.8, f"Pearson R={r2:.5f}", transform=ax2.transAxes, family='monospace')
        fig.savefig(f"plots/{filename}.png")

    # Expt 1a
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    for i in tqdm(range(n)):
        g = Graph()
        g.add_variable("L1", None, k=ldim)
        g.add_variable("L2", None, k=ldim)
        g.add_variable("X1", "L1")
        g.add_variable("X2", "L1")
        g.add_variable("X3", "L1")
        g.add_variable("X4", "L1")
        g.add_variable("X5", "L2")
        g.add_variable("X6", "L2")
        l1.append(g.calc_tetrad("X1", "X2", "X3", "X4"))
        l2.append(g.calc_tetrad("X1", "X4", "X2", "X3"))
        l3.append(g.calc_tetrad("X1", "X2", "X5", "X6"))
        l4.append(g.calc_tetrad("X1", "X6", "X2", "X5"))

    r1 = pearsonr(l1, l2)[0]
    r2 = pearsonr(l3, l4)[0]
    plot_graph(l1, l2, l3, l4, r1, r2, "expt1a")

    # Expt 1b
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    for i in tqdm(range(n)):
        g = Graph()
        g.add_variable("L1", None, k=ldim)
        g.add_variable("L2", "L1", k=ldim)
        g.add_variable("X1", "L1")
        g.add_variable("X2", "L1")
        g.add_variable("X3", "L1")
        g.add_variable("X4", "L1")
        g.add_variable("X5", "L2")
        g.add_variable("X6", "L2")
        l1.append(g.calc_tetrad("X1", "X2", "X3", "X4"))
        l2.append(g.calc_tetrad("X1", "X4", "X2", "X3"))
        l3.append(g.calc_tetrad("X1", "X2", "X5", "X6"))
        l4.append(g.calc_tetrad("X1", "X6", "X2", "X5"))

    r1 = pearsonr(l1, l2)[0]
    r2 = pearsonr(l3, l4)[0]
    plot_graph(l1, l2, l3, l4, r1, r2, "expt1b")

    # Expt 1c
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    for i in tqdm(range(n)):
        g = Graph()
        g.add_variable("L2", None, k=ldim)
        g.add_variable("L1", "L2", k=ldim)
        g.add_variable("X1", "L1")
        g.add_variable("X2", "L1")
        g.add_variable("X3", "L1")
        g.add_variable("X4", "L1")
        g.add_variable("X5", "L2")
        g.add_variable("X6", "L2")
        l1.append(g.calc_tetrad("X1", "X2", "X3", "X4"))
        l2.append(g.calc_tetrad("X1", "X4", "X2", "X3"))
        l3.append(g.calc_tetrad("X1", "X2", "X5", "X6"))
        l4.append(g.calc_tetrad("X1", "X6", "X2", "X5"))

    r1 = pearsonr(l1, l2)[0]
    r2 = pearsonr(l3, l4)[0]
    plot_graph(l1, l2, l3, l4, r1, r2, "expt1c")


def run_expt2(n, nlist, trials):
    n1 = []
    n2 = []
    for n_data in nlist:
        l1 = []
        l2 = []
        for i in tqdm(range(n)):
            g = Graph()
            g.add_variable("L1", None)
            g.add_variable("L2", None)
            g.add_variable("X1", "L1")
            g.add_variable("X2", "L1")
            g.add_variable("X3", "L1")
            g.add_variable("X4", "L1")
            g.add_variable("X5", "L2")
            g.add_variable("X6", "L2")
            df = g.generate_data(["L1", "X1", "X2", "X3", "X4", "X5", "X6"], n_data)
            test1 = tetrad_test(df, "X1", "X2", "X3", "X4", trials=trials)
            test2 = tetrad_test(df, "X1", "X2", "X5", "X6", trials=trials)
            l1.append(test1)
            l2.append(test2)
        l1 = pd.Series(l1)
        l2 = pd.Series(l2)
        n1.append((l1 > 0.05).sum()/n)
        n2.append((l2 < 0.05).sum()/n)
    dfA = pd.DataFrame({"n": nlist, "test1": n1, "test2": n2})

    n1 = []
    n2 = []
    for n_data in nlist:
        l1 = []
        l2 = []
        for i in tqdm(range(n)):
            g = Graph()
            g.add_variable("L1", None)
            g.add_variable("L2", "L1")
            g.add_variable("X1", "L1")
            g.add_variable("X2", "L1")
            g.add_variable("X3", "L1")
            g.add_variable("X4", "L1")
            g.add_variable("X5", "L2")
            g.add_variable("X6", "L2")
            df = g.generate_data(["L1", "X1", "X2", "X3", "X4", "X5", "X6"], n_data)
            test1 = tetrad_test(df, "X1", "X2", "X3", "X4", trials=trials)
            test2 = tetrad_test(df, "X1", "X2", "X5", "X6", trials=trials)
            l1.append(test1)
            l2.append(test2)
        l1 = pd.Series(l1)
        l2 = pd.Series(l2)
        n1.append((l1 > 0.05).sum()/n)
        n2.append((l2 < 0.05).sum()/n)
    dfB = pd.DataFrame({"n": nlist, "test1": n1, "test2": n2})

    n1 = []
    n2 = []
    for n_data in nlist:
        l1 = []
        l2 = []
        for i in tqdm(range(n)):
            g = Graph()
            g.add_variable("L2", None)
            g.add_variable("L1", "L2")
            g.add_variable("X1", "L1")
            g.add_variable("X2", "L1")
            g.add_variable("X3", "L1")
            g.add_variable("X4", "L1")
            g.add_variable("X5", "L2")
            g.add_variable("X6", "L2")
            df = g.generate_data(["L1", "X1", "X2", "X3", "X4", "X5", "X6"], n_data)
            test1 = tetrad_test(df, "X1", "X2", "X3", "X4", trials=trials)
            test2 = tetrad_test(df, "X1", "X2", "X5", "X6", trials=trials)
            l1.append(test1)
            l2.append(test2)
        l1 = pd.Series(l1)
        l2 = pd.Series(l2)
        n1.append((l1 > 0.05).sum()/n)
        n2.append((l2 < 0.05).sum()/n)
    dfC = pd.DataFrame({"n": nlist, "test1": n1, "test2": n2})
    return dfA, dfB, dfC

def plot_expt2_data(df, filename):
    df = df.melt(id_vars="n", value_vars=["test1", "test2"])
    ax = df.pivot("n", "variable", "value").plot(kind="bar")
    ax.set_ylim(0, 1)
    plt.savefig(f"plots/{filename}.png")


if __name__ == "__main__":
    A, B, C = run_expt2(n=100, nlist=[10000], trials=1000)
    plot_expt2_data(A, "expt2a")
    plot_expt2_data(B, "expt2b")
    plot_expt2_data(C, "expt2c")

