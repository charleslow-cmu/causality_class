import numpy as np
from numpy.linalg import matrix_rank
import pandas as pd
from math import sqrt
from pdb import set_trace

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
        try:
            test = matrix_rank(cov) <= rk
        except:
            set_trace()
        return test


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
