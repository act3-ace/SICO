'''
Auxiliary functions for calculating and testing the viral load model.

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

import numpy as np
from matplotlib import pyplot as plt
from itertools import cycle


def viralHingeProfile(t0, tP, Vp, tF, Vd=3, tD=0, ticklen=1, tickforecast=100):
    '''
    Viral evolution model from (Larremore, 2020)

    t0: time detectable
    tP: time peak viral load is reached
    Vp: value of log10 peak viral load 
    tF: time disease is no longer detectable
    Vd: detectable viral load (log10) 
    tD: infection tick (date of infection)
    ticklen: length of one tick in days 
    dayforecast: number of days in the future to forecast the disease
    '''

    ticks = np.arange(tD, tD + tickforecast*ticklen, ticklen)
    viralLoad = np.zeros(tickforecast)
    t0tP = (ticks >= t0) & (ticks <= tP)
    tPtF = (ticks >= tP) & (ticks <= tF)

    # y = m(x-x1) + y1
    viralLoad[t0tP] = 10**(Vd + (Vp-Vd)/(tP-t0) * (ticks[t0tP] - t0)) # convert from log10
    viralLoad[tPtF] = 10**(Vd + (Vd-Vp)/(tF-tP) * (ticks[tPtF] - tF))

    return viralLoad

def allocateViralParameters(symptomaticIds, tD): 
    '''
    Distribute parameters for viral model 

    Input
        symptomaticIds: boolean array marking symptomatic agents 
        tD: date of infection (tick index)
    Return
        dictionary of infection parameters
    '''
    nAgents = len(symptomaticIds)
    tS = np.array([99999.0]*nAgents) 
    tF = np.array([99999.0]*nAgents) 

    t0 = tD + np.random.uniform(2.5, 3.5, (nAgents))  # time first detectable
    tP = t0 + np.minimum(np.random.gamma(1.5, 1, (nAgents)) + 0.5, 3) # time of peak viral load
    Vp = np.random.uniform(7, 11, (nAgents)) # peak viral load value
    # Symptomatics
    tS[symptomaticIds] = tP[symptomaticIds] + np.random.uniform(0, 3, (np.sum(symptomaticIds))) # time symptoms appear
    tF[symptomaticIds] = tS[symptomaticIds] + np.random.uniform(4, 9, (np.sum(symptomaticIds))) # time VL no longer detectable
    # Asymptomatics 
    tF[~symptomaticIds] = tP[~symptomaticIds] + np.random.uniform(4, 9, (np.sum(~symptomaticIds))) # time VL no longer detectable

    return dict(t0=t0, tP=tP, Vp=Vp, tS=tS, tF=tF)

def computeAllViralLoads(numAgents, fractionSymptomatic, tD=0, ticklen=1, tickforecast=100):
    '''
    Conductor function 

    Input:
        numAgents: number of infected agents 
        fractionSymptomatic: fraction of infected agents that experience symptoms
        tD: array of ticks when agents became infected 
        ticklen: length of one tick (days)
        tickforecast: days in the future to forecast viral load
    Return:
        viralLoadArr: Array with shape (numAgents x tickforecast) that holds viral load values (absolute, not log10)
        symptom array: Array shape (numAgents) - date symptoms start if agent is symptomatic, else \infty
    '''
    # assign symptomatic agents
    symptomaticIdxs = np.random.choice(numAgents, int(np.round(numAgents*fractionSymptomatic)), replace=False)
    symptomaticArr = np.array([x in symptomaticIdxs for x in range(numAgents)])

    v = allocateViralParameters(symptomaticArr, tD=tD)
    try: 
        viralLoadArr = [viralHingeProfile(t0=t0, 
                                        tP=tP, 
                                        Vp=Vp, 
                                        tF=tF, 
                                        Vd=3, tD=tD, 
                                        ticklen=ticklen, 
                                        tickforecast=(tickforecast-tD) if tD < 0 else tickforecast)[-tickforecast:]  
                                        for (t0, tP, Vp, _, tF, tD) in zip(*v.values(),tD)]
    except: 
        viralLoadArr = [viralHingeProfile(t0=t0, 
                                        tP=tP, 
                                        Vp=Vp, 
                                        tF=tF, 
                                        Vd=3, tD=tD, 
                                        ticklen=ticklen, 
                                        tickforecast=(tickforecast-tD) if tD < 0 else tickforecast)[-tickforecast:]  
                                        for (t0, tP, Vp, _, tF, tD) in zip(*v.values(),cycle([tD]))]

    return viralLoadArr, v['tS'] 

#======================================================================================================
# Hinge function tests
if __name__ == "__main__":
    
    from matplotlib import pyplot as plt

    currDay = 0
    tickLen = 0.25
    tickForecast = 100
    viralLoadArr, symptomsStart = computeAllViralLoads(50, 0.6, tD=currDay, ticklen=tickLen)

    plt.figure(figsize=(4,3))
    for s, arr in zip(symptomsStart, viralLoadArr):
        ticks = np.arange(currDay, currDay + tickForecast*tickLen, tickLen)
        if s < 99999:
            plt.plot(ticks[arr>0], np.log10(arr[arr>0]), color='skyblue', alpha=0.4, label='Symptomatic')
        else:
            plt.plot(ticks[arr>0], np.log10(arr[arr>0]), color='salmon', alpha=0.4, label='Asymptomatic')

    plt.xlabel('Days after infection')
    #plt.xlim([0,16])
    #plt.legend()
    plt.xticks(list(range(0,16,2)))
    plt.ylabel(r'$\log_{10}(V)$ in cp/ml')
    plt.rcParams['figure.dpi'] = 500
    plt.rcParams['savefig.dpi'] = 500
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.savefig('FILL/YOUR/PATH/batchViralLoads.png')
