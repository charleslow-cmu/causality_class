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


from GaussianGraph import GaussianGraph
from MinimalGroup import MinimalGroup
from LatentGroups import LatentGroups
from StructureFinder import StructureFinder
from RankTester import RankTester
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
            refModel = StructureFinder(g, alpha=alpha)
            refModel.findLatentStructure(verbose=verbose, sample=False)
            
            # Run our model on the sample
            testModel = StructureFinder(g, alpha=alpha)
            df = g.generateData(n=sampleSize)
            testModel.addSample(df)
            testModel.findLatentStructure(verbose=verbose, sample=True)
            comparer = StructureComparer(refModel.l, testModel.l)
            score = comparer.getScore()
            scorelist.append(score)

            #if trial % 5 == 0:
            #    print(f"Avg Score: {sum(scorelist) / trial:.6f}")
        mean = sum(scorelist) / nTrials
        sd = 1/nTrials * sum([pow(x - mean, 2) for x in scorelist])
        print(f"Avg Score: {mean:.6f}")
        scores.append({"mean": mean, "sd": 2*sd, "n": sampleSize})
    return scores
        

if __name__ == "__main__":

    # Run Trials
    nTrials = 10
    sampleSizes = [100, 500, 1000, 2000, 5000]
    scores = runTests(scenario1, nTrials, sampleSizes, alpha=0.05, verbose=False)

    df = pd.DataFrame(scores)
    df.to_csv("temp.csv", index=False)

    df = pd.read_csv("temp.csv")
    df["n"] = df["n"].astype(str)
    plt.plot(df.n, df["mean"], color="b")
    plt.errorbar(df.n, df["mean"], df.sd, color="b")
    plt.scatter(df.n, df["mean"], marker="o", color="b", s=50)
    plt.savefig("test.png")





    # Testing
    #k = 1000
    #reject = 0
    #for _ in range(100):
    #    g = scenario1()
    #    df = g.generateData(500)

    #    rankTester = RankTester(df, trials=1000, normal=True, alpha=0.05)
    #    test = rankTester.test([0,5,9], [1,6,4], r=2)
    #    reject += test
    #print(reject)

    #plt.hist(detList, bins=500, density=True)
    #x = np.arange(0, 10, .01)
    #kde = stats.gaussian_kde(detList, 0.25)
    #plt.plot(x, kde.pdf(x), lw=1)
    #plt.savefig("test.png")

