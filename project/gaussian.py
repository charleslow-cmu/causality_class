import numpy as np
from numpy.linalg import matrix_rank
from math import sqrt, pow, floor
from math import factorial as fac
from itertools import combinations
from copy import deepcopy
from pdb import set_trace
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Function to run trials on a scenario
def runTests(scenario, nTrials, sampleSizes, k=10):
    for sampleSize in sampleSizes:
        print(f"Running for Sample Size {sampleSize}")
        scorelist = []
        trial = 0
        while trial < nTrials:

            g = scenario()
            model = StructureFinder(g, alpha=0.01)
            model.findLatentStructure(verbose=False, sample=False)
            reference = model.l.latentDict

            df = g.generateData(n=sampleSize)
            model.addSample(df)
            model.prepareBootstrapCovariances(k=k)

            try:
                model.findLatentStructure(verbose=False, sample=True)
                score = compareStructure(model.l.latentDict, reference)
                scorelist.append(score)

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
from misc import *
from scenarios import *

if __name__ == "__main__":

    # Run Trials
    nTrials = 10
    sampleSizes = [500, 1000]
    runTests(scenario1, nTrials, sampleSizes, k=100)

