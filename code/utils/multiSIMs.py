'''
Class for running a batch of ABM simulations.

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
import numpy as np
from tqdm import tqdm

from utils.ABM import ABM

#======================================================================================================
class multiSIMs():
    """
    Class for running a batch of ABM simulations


    Attributes
    ----------
    sampleDict : dict {}

    mainDict : dict {}

    samples : Pandas dataframe

    DEBUG : boolean
        Run in debug mode?

    completedSamples : list of str
        Holds list of file names of complete samples


    Methods
    -------
    setupStorageFld()
    sampleOutputFile(id)
    prepSamples()
    retrieveCompletedSamples()
    writeSamplesDF()
    readSamplesDF()
    runSample(id)
    runMultiSIMs(batchLen=None)
    """

    #==================================================================================================
    def __init__(
        self,
        storageFld = None,
        mainDict = None,
        sampleDict = None,
        DEBUG = False
        ):
        """
        Parameters
        ----------
        storageFld : str (default is None)
            name of storage folder for results
        mainDict : dict (default is None)
            Parameters for the ABM sims
        sampleDict : dict (default is None)

        DEBUG : boolean (default is False)
            Run in debug mode?
        """

        self.sampleDict = sampleDict
        self.mainDict = mainDict
        self.samples = None
        self.DEBUG = DEBUG
        self.completedSamples = []


        if self.mainDict['testRunSeed'] is not None: 
            np.random.seed(self.mainDict['testRunSeed'])
            self.mainDict['testRunSeed'] = None 

        if storageFld is None:
            print('ERROR! I need storageFld for this run!!!')
        else:
            self.storageFld = storageFld

    #==================================================================================================
    def setupStorageFld(self):
        """
        Create storage directory if it doesn't exist

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if not os.path.isdir(self.storageFld):
            os.makedirs(self.storageFld)

    #==================================================================================================
    def sampleOutputFile(self, id):
        """
        Create output file name

        The file name is the id provided with a zero fill to 8 characters length.

        Parameters
        ----------
        id : int
            ID of the simulation to run

        Returns
        -------
        str
            Name of output file
        """

        self.setupStorageFld()
        return os.path.join(self.storageFld, str(id).zfill(8) + ".pkl")

    #==================================================================================================
    def prepSamples(self):
        """
        Prepare samples for batch simulation

        This function sets up the output storage and builds a dictionary of samples 
        to run.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.setupStorageFld()
        self.samples = pd.DataFrame(self.sampleDict)
        self.samples['id'] = np.arange(self.samples.shape[0])
        
    #==================================================================================================
    def retrieveCompletedSamples(self):
        """
        Gets list of completed simulations

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.setupStorageFld()
        ldir = os.listdir(self.storageFld)
        ldir = [x for x in ldir if os.path.isfile(os.path.join(self.storageFld, x))]
        self.completedSamples = [x.replace('.pkl','') for x in ldir]

    #==================================================================================================
    def writeSamplesDF(self):
        """
        Writes simulation output to pickle file

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.setupStorageFld()
        self.samples.to_pickle(self.storageFld + "_samplesDF.pkl")

    #==================================================================================================
    def readSamplesDF(self):
        """
        Reads simulation output from pickle file

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.setupStorageFld()
        self.samples = pd.read_pickle(os.path.join(self.storageFld, "samplesDF.pkl"))

    #==================================================================================================
    def runSample(self, id):
        """
        Performs a single simulation

        Parameters
        ----------
        id : int
            ID of the simulation to run

        Returns
        -------
        None
        """

        tr = self.samples.loc[self.samples['id']==id]
        if tr.shape[0] > 0:
            tr = tr.to_dict('records')[0]
            if self.DEBUG: print(f"runSample: tr={tr}")
            rdict = self.mainDict.copy()
            rdict.update(tr)
            rdict.pop('repeat')
            xx = ABM(**rdict)
            xx.runFullTime()
            ds = pd.DataFrame(xx.dailyStatus)
            ds = ds.fillna(0)
            ds['id'] = rdict['id']
            ds.to_pickle(self.sampleOutputFile(id))
        else:
            pass

    #==================================================================================================
    def runMultiSIMs(self, batchLen = None):
        """
        Peforms multiple simulations

        Parameters
        ----------
        batchLen : int
            Number of simulations to run

        Returns
        -------
        None
        """

        if self.samples is None:
            self.prepSamples()
            self.writeSamplesDF()
        self.retrieveCompletedSamples()
        self.toComplete = [x for x in self.samples['id'] if x not in self.completedSamples]
        if batchLen is None: 
            batchLen = len(self.toComplete)
        else:
            self.toComplete = self.toComplete[:min(batchLen, len(self.toComplete))]
        for id in tqdm(self.toComplete):
            self.runSample(id)
