import numpy as np
from pdb import set_trace

class RankTester:

    def __init__(self, data, trials=100, normal=False, alpha=0.05):

        # Centre the data
        data = data - np.mean(data, axis=0)
        self.data = np.array(data)
        self.n = data.shape[0]
        self.trials = trials
        self.normal = normal
        self.alpha = alpha

        # Make Sample Covariance
        self.S = 1/(self.n-1) * self.data.T @ self.data

    # Create test statistic according to Robin, Smith (2000)
    # And test the null hypothesis: rank(S[p,q]) = r
    # Against the alternative: rank(S[p,q]) > r
    # Returns True if null is rejected
    def test(self, pcols, qcols, r=1):

        assert len(pcols) >= len(qcols), "Must have more p columns"
        T = self.n
        B = self.S[np.ix_(pcols, qcols)]
        #print("B is:")
        #print(B)

        C = self.computeC(B, r)
        #print("C is:")
        #print(C)

        lambs, D = self.computeD(B, r)
        #print("D is:")
        #print(D)
        #print(f"lambs is: {lambs}")

        CD = np.kron(D, C)
        Omega = self.asymptoticCov(pcols, qcols)
        eqn = CD.T @ Omega @ CD
        Ws, V = eig(eqn)

        testStat = np.sum(lambs) * T

        # Simulate Weighted Sum of Chi Squares
        # draws: trials x # of eigs
        # instances: trials x 1
        draws = np.random.chisquare(1, size=self.trials * len(Ws))
        draws = np.reshape(draws, (self.trials, len(Ws)))
        instances = draws @ Ws
        criticalValue = np.percentile(instances, 100*(1-self.alpha))
        #print(f"TestStat {testStat} vs criticalValue {criticalValue}")
        return testStat > criticalValue

    # Computes estimator of C, the eigenvector matrix of 
    # B @ B.T : p x p
    # Take the terms from r+1 (r in python index) to p
    def computeC(self, B, r):
        p = B.shape[0]
        M = B @ B.T
        _, V = eig(M)
        return V[:, r:p]
        
    # Computes estimator of D, the eigenvector matrix of 
    # B.T @ B : q x q
    # Take the terms from r+1 (r in python index) to q
    def computeD(self, B, r):
        q = B.shape[1]
        M = B.T @ B
        W, V = eig(M)
        #print(f"W is {W}")
        return W[r:q], V[:, r:q]


    # Calculate the Asymptotic Covariance Matrix of subcovariance
    # data: n x d is our raw data
    # Omega: pq x pq
    def asymptoticCov(self, pcols, qcols):
        #print(f"p: {pcols}, q: {qcols}")
    
        cols = sorted(pcols + qcols)
        data = self.data[:, cols]
        pcols = [cols.index(i) for i in pcols]
        qcols = [cols.index(i) for i in qcols]
        n = data.shape[0]
        p = len(pcols)
        q = len(qcols)
    
        Omega = np.zeros((p*q, p*q))
        if not self.normal:
            for ei, e in enumerate(pcols):
                for fi, f in enumerate(qcols):
                    for gi, g in enumerate(pcols):
                        for hi, h in enumerate(qcols):
                            s_efgh = 1/(n-1) * np.sum(data[:,e] * data[:,f] *\
                                                data[:,g] * data[:,h])
                            s_ef = 1/(n-1) * np.sum(data[:,e] * data[:,f])
                            s_gh = 1/(n-1) * np.sum(data[:,g] * data[:,h])
                            row = ei + fi * p
                            col = gi + hi * p
                            Omega[row, col] = s_efgh - s_ef * s_gh
    
        else:
            for ei, e in enumerate(pcols):
                for fi, f in enumerate(qcols):
                    for gi, g in enumerate(pcols):
                        for hi, h in enumerate(qcols):
                            s_eg = 1/(n-1) * np.sum(data[:,e] * data[:,g])
                            s_fh = 1/(n-1) * np.sum(data[:,f] * data[:,h])
                            s_eh = 1/(n-1) * np.sum(data[:,e] * data[:,h])
                            s_fg = 1/(n-1) * np.sum(data[:,f] * data[:,g])
                            row = ei + fi * p
                            col = gi + hi * p
                            Omega[row, col] = s_eg * s_fh + s_eh * s_fg
        return Omega


# Do np.linalg.eig and then return sorted eigenvals and vectors
# in descending order of eigenvalues
def eig(M):
    W, V = np.linalg.eig(M)
    idx = W.argsort()[::-1]
    W = W[idx]
    V = V[:,idx]
    return W, V
