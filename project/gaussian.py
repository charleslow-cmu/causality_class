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

# Function to run trials on a scenario
def runTests(scenario, nTrials, sampleSizes, alpha=0.01, verbose=False):
    scores = []
    for sampleSize in sampleSizes:
        print(f"Running for Sample Size {sampleSize}")
        scorelist = []
        trial = 0
        for trial in tqdm(range(nTrials)):

            # Create the reference solution
            g = scenario()
            #refModel = StructureFinder(g, alpha=alpha)
            #refModel.findLatentStructure(verbose=verbose, sample=False)
            
            # Run our model on the sample
            testModel = StructureFinder(g, alpha=alpha)
            df = g.generateData(n=sampleSize)
            testModel.addSample(df)
            testModel.findLatentStructure(verbose=verbose, sample=True)
            #comparer = StructureComparer(refModel.l, testModel.l)
            #score = comparer.getScore()
            #print(f"Score is {score}")
            #scorelist.append(score)

        #mean = sum(scorelist) / nTrials
        #sd = 1/nTrials * sum([pow(x - mean, 2) for x in scorelist])
        #print(f"Avg Score: {mean:.6f}")
        #scores.append({"mean": mean, "sd": 2*sd, "n": sampleSize})
    return scores
        
def plot_scores(scores):
    df = pd.DataFrame(scores)
    df.to_csv("temp.csv", index=False)

    df = pd.read_csv("temp.csv")
    df["n"] = df["n"].astype(str)
    plt.plot(df.n, df["mean"], color="b")
    plt.errorbar(df.n, df["mean"], df.sd, color="b")
    plt.scatter(df.n, df["mean"], marker="o", color="b", s=50)
    plt.savefig("test.png")

def printGraph(model, outpath):
    G = getGraph(model.l)
    G.toDot("example.dot")
    graphs = pydot.graph_from_dot_file('example.dot')
    graphs[0].set_size('"8,8!"')
    graphs[0].write_png(outpath)


def runScenario(scenario=None):
    def run(scenario):
        g = scenarios[scenario]()
        model = StructureFinder(g, alpha=0.05)
        model.addSample(g.generateData(2000))
        model.findLatentStructure(verbose=True, sample=False)
        printGraph(model, f'plots/scenario{scenario}.png')

    if scenario is None:
        for scenario in scenarios.keys():
            run(scenario)
    else:
        run(scenario)


if __name__ == "__main__":
    runScenario("5b")

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
