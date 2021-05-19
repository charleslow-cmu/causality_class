from Group import Group
from MixedGraph import MixedGraph
from copy import deepcopy
from math import factorial as fac
from math import sqrt
import numpy as np
from scipy.stats import norm
from itertools import combinations
from pdb import set_trace
import IPython

# Miscellaneous functions
def setLength(varset):
    assert not isinstance(varset, str), "Cannot be string."
    n = 0
    for vset in varset:
        if isinstance(vset, Group):
            n += len(vset)
        else:
            assert False, "Should be Group."
    return n


# Take difference between sets of Groups
def setDifference(As, Bs):
    diff = As - Bs # first remove any common elements
    newset = set()
    while len(diff) > 0:
        A = diff.pop()
        newset.add(A)
        for B in Bs:
            if len(A.intersection(B)) > 0:
                newset.remove(A)
                break
    return newset


# Check if any element exists in the intersection of two
# sets of Groups
def setOverlap(As, Bs):
    if len(As.intersection(Bs)) > 0:
        return True

    return len(setIntersection(As, Bs)) > 0


def setIntersection(As, Bs):

    Avars = set()
    for A in As:
        Avars.update(A.vars)
    Bvars = set()
    for B in Bs:
        Bvars.update(B.vars)

    return Avars.intersection(Bvars)


# Given a set of Groups
# Deduplicate cases where Vs includes {L1, {L1, L3}}
# into just {{L1, L3}}
def deduplicate(Vs):
    newVs = set()
    for Vi in Vs:
        for Vj in Vs:
            isDuplicate = False
            if Vi.vars < Vj.vars:
                isDuplicate = True
                break
        if not isDuplicate:
            newVs.add(Vi)
    return newVs


def vprint(s, verbose=False):
    if verbose:
        print(s)

# generateSubset: Generate set of Groups of variables 
# vset: set of Group of variables
# Returns: list of sets of Groups, each set has setLength = k
def generateSubset(vset, k=2):

    def recursiveSearch(d, gap, currSubset=set()):
        thread = f"currSubset: {currSubset}, d: {d}, gap is {gap}"
        d = deepcopy(d)
        currSubset = deepcopy(currSubset)
        setlist = []

        # Terminate if empty list
        if len(d) == 0:
            return setlist

        # Pop Groups larger than current gap, we cannot take
        # any of them in
        maxDim = max(d)
        while maxDim > gap:
            d.pop(maxDim)
            if len(d) == 0:
                return setlist
            maxDim = max(d)

        # Pop one Group
        v = d[maxDim].pop()
        if len(d[maxDim]) == 0:
            d.pop(maxDim)

        # Branch to consider all cases
        # Continue current search without this element
        if len(d) > 0:
            setlist.extend(recursiveSearch(d, gap, currSubset))

        # Terminate branch if newGroup overlaps with currSubset
        if groupInLatentSet(v, currSubset):
            return setlist

        gap -= maxDim
        currSubset.add(v)

        # Continue search if gap not met
        if gap > 0 and len(d) > 0:
            setlist.extend(recursiveSearch(d, gap, currSubset))

        # End of search tree
        if gap == 0:
            setlist.append(currSubset)

        return setlist

    if k == 0:
        return [set()]

    # Create dictionary where key is dimension size and v is a list 
    # of frozensets of variables
    d = {}
    for v in vset:
        assert isinstance(v, Group), "Should be Group."
        n = len(v)
        d[n] = d.get(n, set()).union([v])

    # Run recursive search
    result = recursiveSearch(d, k)
    if len(result) == 0:
        return [set()]
    else:
        return result


# Check if new group of latent vars exists in a current
# list of latent vars
def groupInLatentSet(V: Group, currSubset: set):
    for group in currSubset:
        if len(V.vars.intersection(group.vars)) > 0:
            return True
    return False

# Print latent Dict
def pprint(d, verbose=False):

    def fsetToText(fset):
        l = [x for x in iter(fset)]
        return ",".join(l)

    for P, v in d.items():
        subgroups = v["subgroups"]
        Cs = v["children"]
        Ctext = ",".join([str(C) for C in Cs])
        Ptext = str(P)

        text = f"{Ptext} : {Ctext}"
        if len(subgroups) > 0:
            text += " | "
            for subgroup in subgroups:
                text += f"[{str(subgroup)}]"
        if verbose:
            print(text)


# Centre the mean of data
def meanCentre(df):
    n = df.shape[0]
    return df - df.sum(axis=0)/n


# Return n choose r
def numCombinations(n, r):
    return fac(n) // fac(r) // fac(n-r)


# Compare against a reference solution
def compareStructure(latentDict, reference):
    reference = deepcopy(reference)
    score = 0
    groups = 0
    while len(reference) > 0:
        groups += 1
        parent, values = reference.popitem()
        refChildren = values["children"]
        refSubMeasures = getAllMeasures(latentDict, values["subgroups"])
        score += findEntry(latentDict, refChildren, refSubMeasures)
    percent = score / groups
    return percent


def getAllMeasures(latentDict, subgroups):
    measures = set()

    for subgroup in subgroups:
        values = latentDict[subgroup]
        childrenP = values["children"]
        subgroupsP = values["subgroups"]

        for child in childrenP:
            if not child.isLatent():
                measures.update(childrenP)

        if len(subgroupsP) > 0:
            measures.update(getAllMeasures(latentDict, subgroupsP))

    return measures


# Given a set of Children, try the exact same set in a dictionary
def findEntry(latentDict, refChildren, subgroupMeasures):
    for group in latentDict:
        values = latentDict[group]
        children = values["children"]
        if children == refChildren:
            subMeasures = getAllMeasures(latentDict, values["subgroups"])
            if subMeasures == subgroupMeasures:
                return True
    return False


#!! TESTS

# S: Sample Covariance
# I, J: Disjoint index sets, |I| = |J|
def traceMatrixCompound(S, I, J, k):
    X = I + J # Union of I and J
    SijInv = np.linalg.inv(S[np.ix_(X, X)])
    Inew = [X.index(i) for i in I]
    Jnew = [X.index(j) for j in J]
    Sij = SijInv[np.ix_(Inew, Jnew)]
    Sji = S[np.ix_(J, I)]
    A = Sji @ Sij

    m = A.shape[0]
    Sum = 0
    for Vs in combinations(range(m), k):
        Vs = list(Vs)
        Atemp = A[np.ix_(Vs, Vs)]
        Sum += np.linalg.det(Atemp)
    Sum = pow(-1, k) * Sum
    # print(f"traceMatrixCompound is {Sum}")
    return Sum

def determinantVariance(S, I, J, n):
    assert len(I) == len(J), "I and J must be same length"
    m = len(I)
    X = I + J
    SijDet = np.linalg.det(S[np.ix_(I, J)])
    SijijDet = np.linalg.det(S[np.ix_(X, X)])
    #print(f"SijDet is {SijDet}")

    Sum = 0
    for k in range(m):
        Sum += fac(m-k) * fac(n+2) / fac(n+2-k) *\
                traceMatrixCompound(S, I, J, k)
    firstTerm = fac(n) / fac(n-m) * pow(SijDet, 2) *\
                    (fac(n+2)/fac(n+2-m) - fac(n)/fac(n-m))
    secondTerm = fac(n) / fac(n-m) * SijijDet * Sum
    variance = firstTerm + secondTerm

    # Heuristic (better way to handle negative variance?)
    if variance < 0:
        return 1
    else:
        return variance


def determinantMean(S, I, J, n):
    Scatter = S * n
    x = np.linalg.det(Scatter[np.ix_(I, J)])
    return x


# Returns p value
def determinantTest(S, I, J, n):
    detMean = determinantMean(S, I, J, n)
    detVar = determinantVariance(S, I, J, n)
    zStat = abs(detMean) / sqrt(detVar)
    pValue = (1 - norm.cdf(zStat)) * 2
    return pValue

# Return true if fail to reject null
def bonferroniTest(plist, alpha):
    m = len(plist)
    return not any([p < alpha/m for p in plist])

# Return true if fail to reject null
def bonferroniHolmTest(plist, alpha):
    plist = sorted(plist)
    m = len(plist)
    tests = [p < alpha/(m+1-k) for k,p in enumerate(plist)]
    #print(sum(tests))
    #print(len(plist))
    #print(plist[0])
    return not any(tests)

# Given data df, bootstrap sample and make new covariance
def bootStrapCovariance(data):
    n = data.shape[0]
    index = np.random.randint(low=0, high=n, size=n)
    bootstrap = data.values[index]
    cov = 1/(n-1) * bootstrap.T @ bootstrap
    return cov


# Calculate the Asymptotic Covariance Matrix of subcovariance
# data: n x d is our raw data
# Omega: pq x pq
def asymptoticCov(data, pcols, qcols, normal=False):

    # Centre the data
    data = data - np.mean(data, axis=0)
    cols = sorted(pcols + qcols)
    data = data[:, cols]
    pcols = [cols.index(i) for i in pcols]
    qcols = [cols.index(i) for i in qcols]

    # Calculate Sample Covariance
    # B is estimator for subcovariance matrix
    # We are interested in rank of B
    n = data.shape[0]
    p = len(pcols)
    q = len(qcols)

    Omega = np.zeros((p*q, p*q))
    if not normal:
        for e in range(p):
            for f in range(q):
                for g in range(p):
                    for h in range(q):
                        s_efgh = 1/(n-1) * np.sum(data[:,e] * data[:,f] *\
                                            data[:,g] * data[:,h])
                        s_ef = 1/(n-1) * np.sum(data[:,e] * data[:,f])
                        s_gh = 1/(n-1) * np.sum(data[:,g] * data[:,h])
                        row = e + f * p
                        col = g + h * p
                        Omega[row, col] = s_efgh - s_ef * s_gh

    else:
        for e in range(p):
            for f in range(q):
                for g in range(p):
                    for h in range(q):
                        s_eg = 1/(n-1) * np.sum(data[:,e] * data[:,g])
                        s_fh = 1/(n-1) * np.sum(data[:,f] * data[:,h])
                        s_eh = 1/(n-1) * np.sum(data[:,e] * data[:,h])
                        s_fg = 1/(n-1) * np.sum(data[:,f] * data[:,g])
                        row = e + f * p
                        col = g + h * p
                        Omega[row, col] = s_eg * s_fh - s_eh * s_fg
    return Omega



def getGraph(l):

    # Start off with all Xvars connected
    l = deepcopy(l)
    Xvars = l.X

    G = MixedGraph()
    for X1 in Xvars:
        X1 = next(iter(X1.vars))
        G.addNode(X1, level=0)
        for X2 in Xvars:
            X2 = next(iter(X2.vars))
            if X2 != X1:
                G.addEdge(X1, X2, type=0)

    activeSet = G.nodes

    # Reverse order for discoveredGraph
    discoveredGraph = {}
    for parents, values in reversed(list(l.latentDict.items())):
        discoveredGraph[parents] = values

    # Work iteratively through the Graph Dictionary
    while len(discoveredGraph) > 0:
        parents, values = discoveredGraph.popitem()
        childrenSet = values["children"]

        # Remove all edges from children
        # Remove them from activeSet
        for childGroup in childrenSet:
            for child in childGroup.vars:
                G.removeUndirEdgesFromNode(child)
            activeSet = activeSet - childGroup.vars
            
        # Find highest child level
        highestLevel = 0
        for childGroup in childrenSet:
            for child in childGroup.vars:
                childLevel = G.nodeLevel[child]
                if childLevel > highestLevel:
                    highestLevel = childLevel

        # Add parents as new nodes
        for parent in parents.vars:
            if not parent in G.nodes:
                G.addNode(parent, level=highestLevel+1)
                activeSet.add(parent)

                # Add edges to all other nodes in activeSet
                for V in activeSet:
                    if parent != V:
                        G.addEdge(parent, V, type=0)

        # Add edges from children to new parents
        for parent in parents.vars:
            for childGroup in childrenSet:
                for child in childGroup.vars:
                    G.addEdge(parent, child, type=1)
    return G


# Return a powerset of elements up to cardinality k
def powerset(elements, k):
    subsets = []
    for i in range(1, k+1):
        combs = combinations(elements, i)
        subsets.extend([set(X) for X in combs])
    return subsets


def scombinations(elements, k):
    combs = [set(X) for X in combinations(elements, k)]
    if len(combs) == 0:
        return [set()]
    return combs

# Given two lists of sets, get the cartesian product
def cartesian(list1, list2):
    result = []
    for l1 in list1:
        for l2 in list2:
            result.append(l1.union(l2))
    return result

# Compare two rankDicts
def cmpDict(d1, d2):
    mismatches = {}
    same = True
    for key in d1:
        if d1[key] != d2[key]:
            mismatches[key] = (d1[key], d2[key])
            same = False

        if len(mismatches) > 50:
            break
    return same, mismatches

