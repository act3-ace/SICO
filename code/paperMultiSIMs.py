'''
Perform simulations for paper - COVID-19 scenarios.

Copyright 2023 Matrix Research, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import os
import time 
import pandas as pd
import numpy as np
from random import choices
from itertools import product
 
from utils.multiSIMs import multiSIMs

#======================================================================================================
# Run ID and output directory
dataPath = os.path.dirname(__file__)
outPath = os.path.join(dataPath, "output")
runtime = 'covid19_paper_scenarios_beta04'


#======================================================================================================
# main parameters, not changed below
mainDict = dict(
    popSize = 10000,
    timeHorizon = 120,
    betaDaily = 0.4,
    externalExposureProbDaily = 0.005,
    initialInfected = 200, # 2% of population
    initProportionVaccinated = 0.5,
    fractionSymptomatic = 0.5,
    infectiousViralLoadCut = 10**3, 
    daysTilSusceptible = 30, 
    quarantineLength = 10,
    selfIsolationOnSymptomsProb = 0.7,
    vaccinesAvailablePerDay = 20,
    vaccineAcceptProbMean = 0.7,
    vaccineAcceptProbStd = 0.05,
    vaccineInfectionProb = 0.2,
    poolSize = 5,       # updated by iteration
    noTestingPostQuarantineDays = 0,
    fprSingle = 0.005,   # updated by iteration
    fnrSingle = 0.05,    # updated by iteration
    daysBetweenTesting = 3, # updated by iteration
    daysDelayTestResults = 2, # updated by iteration
    detectionCut = 25,  # updated by iteration
    firstDayOfTesting = 7, 
    testRunSeed = 2023,
    id = None, 
    DEBUG = False
)


#======================================================================================================
# fixed test profiles 
fixedScenarios = dict(
    fprSingle = np.array([0.014, 0.007]),
    fnrSingle = np.array([0.06, 0.15]),
    detectionCut = np.array([100, 1e6]), 
    daysDelayTestResults = np.array([3, 0]),
)

# comparison options 
iterateOver = dict(
    initProportionVaccinated = np.array([0, 0.5]),
    poolSize = np.array([1,5,10]),
    daysBetweenTesting = np.array([4,7,14]),
    vaccinesAvailablePerDay = np.array([0, 50]),
    firstDayOfTesting = np.array([7,120]),
    repeat = np.arange(0,50,1)  # can lower this for reduced computation
    )

scenarios = [list(a) + list(b) for a in zip(*fixedScenarios.values()) for b in product(*iterateOver.values())]
allKeys = [*fixedScenarios.keys(),*iterateOver.keys()]

# all scenarios to evaluate 
sampleDict = {k:l for k,l in zip(allKeys, zip(*scenarios))}


#======================================================================================================
# instantiate the multiSIMs class
x1 = multiSIMs(
        storageFld = os.path.join(outPath, runtime),
        mainDict = mainDict,
        sampleDict = sampleDict,
        DEBUG = False
)

# run sims
start = time.time()
x1.runMultiSIMs(batchLen = None)


#======================================================================================================
# aggregate and save results
tpath = os.path.join(outPath, runtime)
tfiles = os.listdir(tpath)

tlist = []
i = -1
for tfile in tfiles:
    i += 1
    if i%100 == 0: print(i)
    if i > 1e9:
        break
    try: 
        tf = pd.read_pickle(os.path.join(tpath, tfile))
    except: 
        continue 
    tlist.append(tf)

trials = pd.concat(tlist)
trialsData = pd.read_pickle(os.path.join(outPath, runtime + "_samplesDF.pkl"))
trials.to_pickle(os.path.join(outPath, runtime + "_samples_full.pkl"))
trialsData.to_csv(os.path.join(outPath, runtime + "_samplesDF.csv"), index = None)

trialsCost = (
    trials.groupby('id')
    .agg({'testCount':'sum'})
    .reset_index()
    .rename(columns = {'testCount':'tests_total'})
)
td1 = pd.merge(trialsData, trialsCost, on = 'id', how = 'left')

end = time.time()
print(f'Total time: {end-start}')
print(f'Time per run: {(end-start)/len(sampleDict["repeat"])}')