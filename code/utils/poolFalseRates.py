'''
Auxiliary functions calculating the pool-wide false-negative (poolFNR) or false 
positive (poolFPR) rates.  

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

def poolFNR(
    poolSize = None,
    detectableCount = None,
    fnrSingle = None
    ):
    """
    Calculates pool-level false negative rate

    Parameters
    ----------
    poolSize : int
        Number of samples in the pool being tested
    detectableCount : int
        Number of samples in the pool that can be detected
    fnrSingle : double [0,1]
        False negative rate for one test of a single sample

    Returns
    -------
    double
        false negative rate for the pool
    """

    return fnrSingle**detectableCount

def poolFPR(
    poolSize = None,
    detectableCount = None,
    fprSingle = None
    ):
    """
    Calculates pool-level false positive rate

    Parameters
    ----------
    poolSize : int
        Number of samples in the pool being tested
    detectableCount : int
        Number of samples in the pool that can be detected
    fprSingle : double [0,1]
        False positive rate for one test of a single sample

    Returns
    -------
    double
        false positive rate for the pool
    """

    return fprSingle
