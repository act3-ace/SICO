'''
Run single testing scenario.

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
import pandas as pd 
import matplotlib.pyplot as plt 

from utils.ABM import ABM

#======================================================================================================
# Run ID and output directory
dataPath = os.path.dirname(__file__)
outPath = os.path.join(dataPath, "output")
if not os.path.isdir(outPath):
    os.makedirs(outPath)

runId = 'mitigate'


#======================================================================================================
# main parameters, not changed below
mainDict = dict(
    popSize = 10000,
    timeHorizon = 120,
    betaDaily = 0.4,
    externalExposureProbDaily = 0.005,
    initialInfected = 200, # 2% of population
    initProportionVaccinated = 0,
    fractionSymptomatic = 0.5,
    infectiousViralLoadCut = 10**3, 
    daysTilSusceptible = 30, 
    quarantineLength = 10,
    selfIsolationOnSymptomsProb = 0.7,
    vaccinesAvailablePerDay = 100,
    vaccineAcceptProbMean = 0.7,
    vaccineAcceptProbStd = 0.05,
    vaccineInfectionProb = 0.2,
    poolSize = 10,      
    poolingType = 'average',
    noTestingPostQuarantineDays = 0,
    fprSingle = 0.014,   
    fnrSingle = 0.06,   
    daysBetweenTesting = 7,
    daysDelayTestResults = 3,  
    detectionCut = 1e6,  
    firstDayOfTesting = 7, 
    testRunSeed = 2020,
    DEBUG = False
)


#======================================================================================================
# instantiate the ABM class
xx = ABM(**mainDict)

# Run ABM
xx.runFullTime()


#======================================================================================================
# Save results 
dspath = os.path.join(outPath, f"dailyStatus_{runId}.csv")
ds = pd.DataFrame(xx.dailyStatus)
ds.to_csv(dspath, index = None)

paramspath = os.path.join(outPath, f"params_{runId}.csv")
pars = pd.DataFrame(mainDict.items()).rename(columns = {0:'key',1:'value'})
pars.to_csv(paramspath, index = None)

tcpath = os.path.join(outPath, f"testCount_{runId}.csv")
testCount = pd.DataFrame(xx.testCounter)
testCount.to_csv(tcpath, index = None)

# Plot results 
plt.figure(figsize=(6,4.5))
ds.plot(y=['S','E','I','R','FI','TI'],
        color=['tab:blue','tab:orange','tab:brown','tab:red','tab:purple','tab:green'], 
        label = ['susceptible','exposed','infectious','recovered','falsely isolated','isolated while infectious'], xlabel='time step', ylabel='number of agents')
plt.savefig(os.path.join(outPath, f"scenario_{runId}.png"),  dpi=300, bbox_inches='tight')
plt.close()

testCount.plot(ylabel='number of tests administered', xlabel='time step', legend=False)
plt.savefig(os.path.join(outPath, f"testCount_{runId}.png"))
plt.close()



# plot for R0 - only considering infection spread within population 
exposuresPerDay = (ds['newInfections'])/ds['I'].shift(1) 
averageTimeInfectious = 12.25
R0 = exposuresPerDay * averageTimeInfectious
fig = plt.figure(figsize=(4,3))
R0.plot(label='internal reproduction number', xlabel='time step', ylabel='reproduction number (internal)')
plt.savefig(os.path.join(outPath, f"R0_{runId}.png"), dpi=300, bbox_inches='tight')

# effective R0 - includes all infections 
exposuresPerDay = (ds['newInfections'] + ds['newOutsideInfections'])/ds['I'].shift(1) 
averageTimeInfectious = 12.25
R0 = exposuresPerDay * averageTimeInfectious
R0.plot(label='effective overall R0', xlabel='time step', ylabel='R0')
plt.legend()
plt.savefig(os.path.join(outPath, f"effR0_{runId}.png"), dpi=300, bbox_inches='tight')