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

import random
import numpy as np

from utils.poolFalseRates import poolFNR, poolFPR


def testGroup(
    detectableIDs = None,
    poolDetectable = None,
    groupIDs = None,
    fprSingle = None,
    fnrSingle = None,
    pooledTest = False,
    pooledType = None
    ):
    """
    Perform group testing

    This function performs testing over a group and returns a dictionary of 
    results. It exist early if a pool tests negative.

    Parameters
    ----------
    detectableIDs : list of int
        List of agent IDs that have a detectable viral load
    poolDetectable : bool 
        Whether pool is detectable 
    groupIDs : list of int
        List of agent IDs in the current test group
    fprSingle : double [0,1]
        False positive rate for one test of a single sample
    fnrSingle : double [0,1]
        False negative rate for one test of a single sample
    pooledTest: boolean
        Whether or not to perform pooled testing

    Returns
    -------
    dict : {FNIDs=list of int, FPIDs=list of int, TPIDs=list of int, TNIDs=list of int, testCount=int}
        Dictionary where each key is a list of agent IDs that were detected in that fashion
    """

    psize = len(groupIDs)
    detectableCnt = len(detectableIDs)
    testCount = 0
    undetectableIDs = [x for x in groupIDs if x not in detectableIDs]

    # pooled testing check 
    if pooledTest:
        testCount += 1
        if pooledType == 'exponential': 
            # compute separate pooled FPR and FNR based on pool characteristics 
            fprPool = poolFPR(poolSize=psize, detectableCount=detectableCnt, fprSingle=fprSingle)
            fnrPool = poolFNR(poolSize=psize, detectableCount=detectableCnt, fnrSingle=fnrSingle)
        elif pooledType == 'average': 
            # use average viral load to determine detectability of pool (noted as poolDetectable) 
            fprPool = fprSingle
            fnrPool = fnrSingle
            if poolDetectable:  
                detectableCnt = 1
            else: detectableCnt = 0

        testprob = np.random.uniform(0,1)
        if (detectableCnt > 0) and (testprob < fnrPool):
            # false negative pool, return arrays
            return(dict(FNIDs = detectableIDs,
                        FPIDs = [],
                        TPIDs = [],
                        TNIDs = undetectableIDs,
                        testCount = testCount
                        ))
        elif (detectableCnt > 0):
            pass # correctly flagged pool with virus, continue
        elif (np.random.uniform(0,1) < fprPool):
            pass # incorrectly flagged virus-free pool, continue
        else:
            # correctly flagged virus-free pool, return array
            return(dict(FNIDs = detectableIDs,
                        FPIDs = [],
                        TPIDs = [],
                        TNIDs = undetectableIDs,
                        testCount = testCount
                        ))

    # continue here for individual tests
    testCount += psize
    FNIDs = [x for x in detectableIDs if np.random.uniform(0,1) < fnrSingle]
    TPIDs = [x for x in detectableIDs if x not in FNIDs]
    FPIDs = [x for x in undetectableIDs if np.random.uniform(0,1) < fprSingle]
    TNIDs = [x for x in undetectableIDs if x not in FPIDs]

    return(dict(FNIDs = FNIDs,
                FPIDs = FPIDs,
                TPIDs = TPIDs,
                TNIDs = TNIDs,
                testCount = testCount
                ))
