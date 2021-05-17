import numpy as np
from numpy.linalg import matrix_rank
import pandas as pd
from math import sqrt, log
from pdb import set_trace
import sys
import IPython
from itertools import combinations

class GaussianGraph:
    def __init__(self):
        self.vars = []
        self.xvars = []
        self.lvars = []
        self.L = None
        self.phi = []
        self.cov = None

    def random_variance(self):
        return np.random.uniform(1, 5)

    def random_coef(self):
        coef = 0
        while abs(coef) < 0.5:
            coef = np.random.uniform(low=-5, high=5)
        return coef

    def getParentIndex(self, parents):
        if isinstance(parents, str):
            return self.vars.index(parents)
        else:
            return [self.vars.index(parent) for parent in parents]

    def expandL(self, parents):
        n = len(self.vars)
        newL = np.zeros((n, n))
        newL[:(n-1), :(n-1)] = self.L  # Padded
        for parent in parents:
            pindex = self.getParentIndex(parent)
            newL[pindex, n-1] = self.random_coef()
        return newL


    def add_variable(self, name, parents=None):
        if isinstance(parents, str):
            parents = [parents]
        if parents is None:
            parents = []

        # Sanity checks
        if len(parents) > 0:
            variables = set(self.lvars) | set(self.xvars)
            valid_parents = len(set(parents)-variables) == 0
            assert valid_parents, "Parents not found in graph!"

        # Keep track of variables
        self.vars.append(name)
        if "X" in name:
            self.xvars.append(name)
        else:
            self.lvars.append(name)

        # Add exogeneous noise
        self.phi.append(self.random_variance())  # Add noise

        # Adding first variable
        if len(self.vars) == 1:
            self.L = np.zeros((1, 1))
        else:
            self.L = self.expandL(parents)

    def covariance(self):
        if self.cov is None:
            n = len(self.vars)
            lamb = np.identity(n) - self.L
            lamb_inv = np.linalg.inv(lamb)
            phi = np.diag(self.phi)
            self.cov = lamb_inv.T @ phi @ lamb_inv
        return self.cov


    def subcovariance(self, rowvars, colvars):
        cov = self.covariance()
        rowindex = self.getParentIndex(rowvars)
        colindex = self.getParentIndex(colvars)
        return cov[np.ix_(rowindex, colindex)]

    # The main workhorse method
    # Return True if rank is not full
    def rankTest(self, A, B, rk):
        A = sorted(A)
        B = sorted(B)
        cov = self.subcovariance(A, B)
        test = matrix_rank(cov) <= rk
        return test

    def infoDist(self, covAA, covBB, covAB):
        det1 = abs(np.linalg.det(covAB))
        det2 = abs(np.linalg.det(covAA))
        det3 = abs(np.linalg.det(covBB))
        dist = -log(det1 / sqrt(det2*det3))
        return dist 

    # Given a variable set A of size k and B with size >= k
    def infoDistRandom(self, A, B, trials=10):
        k = len(A)
        j = len(B)
        assert j >= k, "B must have greater cardinality"
        covAA = self.subcovariance(A, A) # (k, k)
        covBB = self.subcovariance(B, B) # (j, j)
        covAB = self.subcovariance(A, B) # (k, j)

        # Make random coefficients for A
        coefA1 = np.random.randn(k, k)
        coefA2 = np.random.randn(k, k)

        # Repeatedly trial for different values of B
        dlist = []
        for _ in range(trials):

            # Make random coefficients for B
            coefB = np.random.randn(k, j)
            covs1 = self.makeCov(coefA1, coefB, covAA, covBB, covAB)
            covs2 = self.makeCov(coefA2, coefB, covAA, covBB, covAB)
            d1 = self.infoDist(covs1[0], covs1[1], covs1[2])
            d2 = self.infoDist(covs2[0], covs2[1], covs2[2])
            dlist.append(d1-d2)
        return dlist


    def makeCov(self, coefA, coefB, covAA, covBB, covAB):
        newCovAA = coefA @ covAA @ coefA.T # (k, k)
        newCovBB = coefB @ covBB @ coefB.T # (k, k)
        newCovAB = coefA @ covAB @ coefB.T # (k, k)
        return newCovAA, newCovBB, newCovAB


    def infoDistGen(self, A, B):
        cov = self.subcovariance(A, B)
        covA = self.subcovariance(A, A)
        covB = self.subcovariance(B, B)
        u, d, v = np.linalg.svd(cov)
        tol = d.max() * max(cov.shape) * sys.float_info.epsilon
        d = d[d > tol]
        det2 = np.linalg.det(covA)
        det3 = np.linalg.det(covB)
        dist = -sum(np.log(d)) + 0.5*(log(det2) + log(det3))
        return dist 


    def generateData(self, n=100):
        df = pd.DataFrame(columns = self.xvars)
        data = np.zeros((n, len(self.vars)))
        for i, V in enumerate(self.vars):
            sd = sqrt(self.phi[i])
            noise = np.random.normal(loc=0, scale=sd, size=n)
            data[:, i] = noise + data[:, :i] @ self.L[:i, i]
            if V in self.xvars:
                df[V] = data[:, i]
        return df

    # Test all possible combinations of subcovariance rank test
    def allRankTests(self):
        rankDict = {}
        n = len(self.xvars)
        for i in range(2, n-1):
            Asets = list(combinations(self.xvars, i))
            for j in range(2, n-1):
                print(f"Testing i={i} vs j={j}...")
                Bsets = list(combinations(self.xvars, j))

                for A in Asets:
                    for B in Bsets:
                        Aset = frozenset(A)
                        Bset = frozenset(B)

                        if len(Aset.intersection(Bset)) > 0:
                            continue

                        key = frozenset([Aset, Bset])
                        if key in rankDict:
                            continue

                        A = sorted(A)
                        B = sorted(B)
                        cov = self.subcovariance(A, B)
                        rk = matrix_rank(cov)
                        rankDict[key] = rk
        return rankDict



