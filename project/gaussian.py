import numpy as np
from numpy.linalg import matrix_rank
from math import sqrt, pow, floor
from math import factorial as fac
from itertools import combinations
from copy import deepcopy
from pdb import set_trace
from scipy.stats import norm
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pydot

from GaussianGraph import GaussianGraph
from MinimalGroup import MinimalGroup
from LatentGroups import LatentGroups
from StructureFinder import StructureFinder
from RankTester import RankTester
from MixedGraph import MixedGraph
from StructureComparer import StructureComparer
from misc import *
from scenarios import *

def printGraph(model, outpath):
    G = getGraph(model.l)
    G.toDot("example.dot")
    graphs = pydot.graph_from_dot_file('example.dot')
    graphs[0].set_size('"8,8!"')
    graphs[0].write_png(outpath)


def runScenario(scenario=None, maxk=3):
    def run(scenario):
        g = scenarios[scenario]()
        model = StructureFinder(g, alpha=0.05)
        df = g.generateData(2000)
        model.addSample(df)
        g.df = df
        model.findLatentStructure(maxk=maxk, verbose=True, sample=False)
        printGraph(model, f'plots/scenario{scenario}.png')
        return model

    if scenario is None:
        for scenario in scenarios.keys():
            run(scenario)
    else:
        return run(scenario)


if __name__ == "__main__":
    model = runScenario("5c", maxk=3)
    junctions = model.l.findJunctions(model.l.activeSet)
    IPython.embed(); exit(1)

    #sampleSize = 2000
    #scenario = "7"
    #trials = 1

    #for trial in range(trials):
    #    g = scenarios[scenario]()
    #    model = StructureFinder(g, alpha=0.05)
    #    df = g.generateData(n=sampleSize)
    #    model.addSample(df)
    #    model.findLatentStructure2(verbose=True, sample=False)

    #    printGraph(model, f'plots/scenario{scenario}_{trial}.png')
