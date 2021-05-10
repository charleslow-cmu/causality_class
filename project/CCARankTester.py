import numpy as np
from pdb import set_trace
from statsmodels.multivariate.cancorr import CanCorr
from math import log, pow
from scipy.stats import chi2

class CCARankTester:

    def __init__(self, data, alpha=0.05):

        # Centre the data
        data = data - np.mean(data, axis=0)
        self.data = np.array(data)
        self.n = data.shape[0]
        self.alpha = alpha

        # Make Sample Covariance
        #self.S = 1/(self.n-1) * self.data.T @ self.data

    # Test null hypothesis that rank is less than or equal to r
    # Return True if reject null
    def test(self, pcols, qcols, r=1):
        p = len(pcols)
        q = len(qcols)
        X = self.data[:, pcols]
        Y = self.data[:, qcols]
        cca = CanCorr(X, Y)
        l = cca.cancorr[r:]

        testStat = 0
        for li in l:
            #print(f"li is {li}")
            testStat += log(1 - pow(li, 2))
        testStat = testStat * -(self.n - 0.5*(p+q+3))

        dfreedom = (p-r) * (q-r)
        criticalValue = chi2.ppf(1-self.alpha, dfreedom)
        #print(f"testStat: {testStat}, crit: {criticalValue}")

        return testStat > criticalValue

