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

# Function to run trials on a scenario
def runTests(scenario, nTrials, sampleSizes, k=10, alpha=0.01):
    for sampleSize in sampleSizes:
        print(f"Running for Sample Size {sampleSize}")
        scorelist = []
        trial = 0
        while trial < nTrials:

            g = scenario()
            model = StructureFinder(g, alpha=alpha)
            model.findLatentStructure(verbose=False, sample=False)
            reference = model.l.latentDict

            df = g.generateData(n=sampleSize)
            model.addSample(df)
            model.prepareBootstrapCovariances(k=k)

            try:
                model.findLatentStructure(verbose=False, sample=True)
                score = compareStructure(model.l.latentDict, reference)
                scorelist.append(score)
                pprint(model.l.latentDict, verbose=True)

                trial += 1
                if trial % 1 == 0:
                    percent = sum([score==1 for score in scorelist]) \
                                        / len(scorelist)
                    print(f"% Correct: {percent*100:.1f}%")
            except:
                pass

        percent = sum([score==1 for score in scorelist]) / len(scorelist)
        print(f"Percent correct for sample {sampleSize}: {percent*100:.1f}%")


from GaussianGraph import GaussianGraph
from MinimalGroup import MinimalGroup
from LatentGroups import LatentGroups
from StructureFinder import StructureFinder
from RankTester import RankTester
from misc import *
from scenarios import *


if __name__ == "__main__":

    # Run Trials
    #nTrials = 100
    #sampleSizes = [1000]
    #runTests(scenario1, nTrials, sampleSizes, k=1000, alpha=0.05)

    # Testing
    k = 1000
    reject = 0
    for _ in range(100):
        g = scenario1()
        df = g.generateData(500)

        rankTester = RankTester(df, trials=1000, normal=True, alpha=0.05)
        test = rankTester.test([0,5,9], [1,6,4], r=2)
        reject += test
    print(reject)

    #plt.hist(detList, bins=500, density=True)
    #x = np.arange(0, 10, .01)
    #kde = stats.gaussian_kde(detList, 0.25)
    #plt.plot(x, kde.pdf(x), lw=1)
    #plt.savefig("test.png")

