'''
Auxiliary function for disease testing.

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
import random

from utils.testGroup import testGroup


def testingDay(
    ctIDs = None,
    viralLoadArray = None,
    detectionCut = None,
    poolSize = None,
    poolFunType = None,
    testAccuracyArray = None,
    DEBUG = False
    ):
    """
    Perform appropriate testing procedure

    Parameters
    ----------
    ctIDs : list of int
        IDs of each agent in the current test pool
    viralLoadArray : Numpy ndarray
        Viral load for every agent in the population
    detectionCut : double [0,1]
        Limit of detection for tests
    poolSize : int
        Number of agents in the testing pool
    poolFunType : string
        Type of pooling function to use ('average' or 'exponential')
    testAccuracyArray : dict of {fprSingle, fnrSingle, fprPool, fnrPool}
        Dictionary with false positive or false negative testing rates for individual 
        tests and for testing a pool of the current size (poolSize)
    DEBUG : boolean
        Should the function output diagnostic statements?

    Returns
    -------
    dict : {FNIDs=list of int, FPIDs=list of int, TPIDs=list of int, TNIDs=list of int, testCount=int}
        Dictionary where each key is a list of agent IDs that were detected in that fashion
    """

    # unpack data needed
    fprSingle = testAccuracyArray['fprSingle']
    fnrSingle = testAccuracyArray['fnrSingle']

    # start here with ctSet
    currentTestable = len(ctIDs)

    # which ones should be detectable by this test?
    detectableIDs = [x for x in ctIDs if viralLoadArray[x] > detectionCut]
    undetectableIDs = [x for x in ctIDs if x not in detectableIDs]

    # Individual testing
    if poolSize == 1:
        return(testGroup(
            detectableIDs=detectableIDs,
            groupIDs=ctIDs,
            fprSingle=fprSingle,
            fnrSingle=fnrSingle,
            pooledTest=False
        ))

    # Pooled test 
    else:                
        # compute number of pools
        poolCount = currentTestable // poolSize
        poolSizes = [poolSize]*poolCount
        tailCount = currentTestable % poolSize

        if tailCount > 0:
            poolCount += 1
            poolSizes.append(tailCount)

        if DEBUG: print(f"poolSizes={poolSizes}")

        # draw pools from currentTestable - better - shuffle then split
        np.random.shuffle(ctIDs)
        pools = dict()
        dets = dict()
        poolDetectable = dict()
        tdict = dict()
        for i in range(poolCount):
            pools[i] = ctIDs[(i*poolSize):((i+1)*poolSize)]
            dets[i] = [x for x in pools[i] if x in detectableIDs] 
            avgPoolViralLoad = sum(viralLoadArray[pools[i]])/len(pools[i])
            if avgPoolViralLoad > detectionCut: 
                poolDetectable[i] = True
            else: 
                poolDetectable[i] = False 
            tdict[i] = testGroup(
                detectableIDs=dets[i],
                poolDetectable=poolDetectable[i],
                groupIDs=pools[i],
                fprSingle=fprSingle,
                fnrSingle=fnrSingle,
                pooledTest=True,
                pooledType=poolFunType
            )

        # aggregate results
        return(dict(
            FNIDs = [x for _,y in tdict.items() for x in y['FNIDs']],
            FPIDs = [x for _,y in tdict.items() for x in y['FPIDs']],
            TNIDs = [x for _,y in tdict.items() for x in y['TNIDs']],
            TPIDs = [x for _,y in tdict.items() for x in y['TPIDs']],
            testCount = sum([y['testCount'] for _,y in tdict.items()])        
            ))
