from misc import *
from MinimalGroup import MinimalGroup
from LatentGroups import LatentGroups

A = MinimalGroup(["L1", "L2"])
B = MinimalGroup(["L3", "L4"])
C = MinimalGroup(["L5"])
D = MinimalGroup(["L6"])
E = MinimalGroup(["L1"])
#
#for subset in generateSubset(set([A, B, C, D]), 3):
#    print(subset)

# Test deduplicate
assert deduplicate(set([A, E])) == set([A])

from statsmodels.multivariate.cancorr import CanCorr
from GaussianGraph import GaussianGraph
from scenarios import *
from StructureFinder import StructureFinder
import numpy as np
from numpy.linalg import det
from math import log,sqrt
import IPython

g = GaussianGraph()
g.add_variable("L1", None)
g.add_variable("L2", "L1")
g.add_variable("L3", ["L1", "L2"])
g.add_variable("L4", ["L1", "L2"])
g.add_variable("X1", ["L1", "L2"])
g.add_variable("X2", ["L1", "L2"])
g.add_variable("X3", ["L1", "L2"])
g.add_variable("X4", ["L1", "L2"])
g.add_variable("X5", ["L3", "L4"])
g.add_variable("X6", ["L3", "L4"])

A1 = ["X1", "X2"]
A2 = ["X1", "X3"]
B = ["X5", "X6"]
L = ["L1", "L2"]

d1 = g.infoDist(A1, B)
d2 = g.infoDist(A2, B)
d3 = g.infoDist(A1, A2)
d4 = g.infoDist(L, B)
print((d1+d2-d3)/2)
print(d4)


#Sigma = g.subcovariance(["L1", "L2"], ["L1", "L2"])
#A1 = g.L[np.ix_([0,1], [2,3])]
#A2 = g.L[np.ix_([0,1], [4,5])]
#covAA = g.subcovariance(["X1", "X2"], ["X1", "X2"])
#covBB = g.subcovariance(["X3", "X4"], ["X3", "X4"])
#covAB = g.subcovariance(["X1", "X2"], ["X3", "X4"])
#print(g.infoDist(covAA, covBB, covAB))

IPython.embed(); exit(1)


#g = scenarios["7b"]()
#rankDict1 = g.allRankTests()
#g = scenarios["7c"]()
#rankDict2 = g.allRankTests()
#
#test, mismatches = cmpDict(rankDict1, rankDict2)
#print(test)
#print(len(rankDict1))
#print(mismatches)


#g = GaussianGraph()
#g.add_variable("L1", None)
#g.add_variable("L2", "L1")
#g.add_variable("L3", "L2")
#g.add_variable("L4", ["L1", "L2"])
#g.add_variable("L5", ["L1", "L2"])
#
#g.add_variable("X1", ["L4", "L5"])
#g.add_variable("X2", ["L4", "L5"])
#g.add_variable("X3", ["L4", "L5"])
#g.add_variable("X4", ["L4", "L5"])
#
#g.add_variable("X5", "L3")
#g.add_variable("X6", "L3")
#g.add_variable("X7", ["L1", "L2", "L3"])
#g.add_variable("X8", ["L1", "L2", "L3"])
#g.add_variable("X9", ["L1", "L2", "L3"])
#
#
#model = StructureFinder(g, alpha=0.05)
#l = LatentGroups(g.xvars)
#l.latentDict[MinimalGroup(["L4", "L5"])] = {
#        "children": [MinimalGroup(x) for x in ["X1", "X2", "X3", "X4"]],
#        "subgroups": set([MinimalGroup("L4")])}
#l.latentDict[MinimalGroup("L4")] = {
#        "children": [MinimalGroup(x) for x in ["X11", "X12"]],
#        "subgroups": set()}
#l.latentDict[MinimalGroup(["L1", "L2"])] = {
#        "children": [MinimalGroup(["L4", "L5"])],
#        "subgroups": set([MinimalGroup(["L4", "L5"])])}
#l.latentDict[MinimalGroup("L3")] = {
#        "children": [MinimalGroup(x) for x in ["X9", "X10"]],
#        "subgroups": set()}
#l.latentDict[MinimalGroup(["L1", "L2", "L3"])] = {
#        "children": [MinimalGroup(x) for x in ["X5", "X6", "X7", "X8"]],
#        "subgroups": set([MinimalGroup(["L1", "L2"]), MinimalGroup("L3")])}
#
#setlist = getSets(l.latentDict, MinimalGroup(["L1", "L2", "L3"]))
#print(setlist)



