import numpy as np
from numpy.linalg import matrix_rank
import pandas as pd
from math import sqrt, log
from pdb import set_trace
import sys
import IPython

class GaussianGraph:
    def __init__(self):
        self.vars = []
        self.xvars = []
        self.lvars = []
        self.L = None
        self.phi = []

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
        n = len(self.vars)
        lamb = np.identity(n) - self.L
        lamb_inv = np.linalg.inv(lamb)
        phi = np.diag(self.phi)
        return lamb_inv.T @ phi @ lamb_inv

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

    def infoDist(self, A, B):
        cov = self.subcovariance(A, B)
        covA = self.subcovariance(A, A)
        covB = self.subcovariance(B, B)
        det1 = abs(np.linalg.det(cov))
        det2 = abs(np.linalg.det(covA))
        det3 = abs(np.linalg.det(covB))
        dist = -log(det1 / sqrt(det2*det3))
        return dist 

    # Given a variable set A of size k and B with size >= k
    # Take a random combination of B
    # Generate cov(A, B) and var(B)
    def generateCovariances(self, A, B):
        k = len(A)
        j = len(B)
        assert j >= k, "B must have greater cardinality"
        covAA = self.subcovariance(A, A)
        covAB = self.subcovariance(A, B)
        covBB = self.subcovariance(B, B)

        # Make random coefficients
        coefs = np.random.randn(k, j)
        newCovAB = covAB @ coefs.T
        newCovBB = coefs @ covBB @ coefs.T
        return covAA, newCovAB, newCovBB


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
