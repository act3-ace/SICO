'''
Script simulating the early spread of a new disease variant .
    - delayed testing, low number initially infected, high beta value.

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

runId = 'new_disease'


#======================================================================================================
# main parameters, not changed below
mainDict = dict(
    popSize = 10000,
    timeHorizon = 120,
    betaDaily = 0.4,
    initialInfected = 40, 
    initProportionVaccinated = 0,
    fractionSymptomatic = 0.3,
    infectiousViralLoadCut = 100, 
    daysTilSusceptible = 20, 
    quarantineLength = 10,
    selfIsolationOnSymptomsProb = 0.9,
    vaccinesAvailablePerDay = 0,
    poolSize = 1,        
    noTestingPostQuarantineDays = 0,
    fprSingle = 0.005,    
    fnrSingle = 0.05,    
    daysBetweenTesting = 7, 
    daysDelayTestResults = 2, 
    detectionCut = 25,  
    firstDayOfTesting = 30, # takes awhile to develop a test 
    testRunSeed = None,
    id = None, 
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
ds.plot(y=['S','E','I','R','FI','TI'],
        color=['tab:blue','tab:orange','tab:brown','tab:red','tab:purple','tab:green'], 
        label = ['susceptible','exposed','infectious','recovered','falsely isolated','isolated while infectious'], xlabel='time step', ylabel='number of agents')
plt.savefig(os.path.join(outPath, f"scenario_{runId}.png"))

testCount.plot(ylabel='number of tests administered', xlabel='time step', legend=False)
plt.savefig(os.path.join(outPath, f"testCount_{runId}.png"))