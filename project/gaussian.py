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
    for sampleSize in sampleSizes:
        print(f"Running for Sample Size {sampleSize}")
        scorelist = []
        trial = 0
        while trial < nTrials:

            # Create the reference solution
            g = scenario()
            refModel = StructureFinder(g, alpha=alpha)
            refModel.findLatentStructure(verbose=verbose, sample=False)
            exit(1)
            
            # Run our model on the sample
            testModel = StructureFinder(g, alpha=alpha)
            df = g.generateData(n=sampleSize)
            testModel.addSample(df)
            testModel.findLatentStructure(verbose=verbose, sample=False)
            comparer = StructureComparer(refModel.l, testModel.l)
            score = comparer.getScore()
            scorelist.append(score)

            trial += 1
            if trial % 5 == 0:
                percent = sum([score==1 for score in scorelist]) \
                                    / len(scorelist)
                print(f"% Correct: {percent*100:.1f}%")

        percent = sum([score==1 for score in scorelist]) / len(scorelist)
        print(f"Percent correct for sample {sampleSize}: {percent*100:.1f}%")




if __name__ == "__main__":

    # Run Trials
    nTrials = 1
    sampleSizes = [5000]
    runTests(scenario4, nTrials, sampleSizes, alpha=0.1, verbose=True)

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

