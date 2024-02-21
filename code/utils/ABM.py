'''
ABM class definition.

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
from collections import Counter
import datetime as dt

from utils.viralLoadEvolution import computeAllViralLoads
from utils.testingDay import testingDay

#======================================================================================================
class ABM:
    """
    ABM class doc
    """
    
    #==================================================================================================
    def __init__(self,
        DEBUG = False,
        popSize = 100,
        timeHorizon = 120,
        betaDaily = 0.7,
        externalExposureProbDaily = 0.01,
        initialInfected = 5,
        initProportionVaccinated = 0.5,
        fractionSymptomatic = 0.8,
        infectiousViralLoadCut = 10**5, 
        daysTilSusceptible = 30,
        quarantineLength = 10,
        selfIsolationOnSymptomsProb = 0.9,
        vaccinesAvailablePerDay = 10,
        vaccineAcceptProbMean = 0.7,
        vaccineAcceptProbStd = 0.05,
        vaccineInfectionProb = 0.3,
        poolSize = 5,
        poolingType = 'average',
        noTestingPostQuarantineDays = 0,
        fprSingle = 0.005,
        fnrSingle = 0.05, 
        daysBetweenTesting = 3,
        daysDelayTestResults = 2,
        detectionCut = 10**5, 
        firstDayOfTesting = 1, # should be < daysBetweenTesting
        testRunSeed = None,
        trackArrays = False,
        id = None, # Maybe remove this? - KP
        **kwargs
    ):
        """
        DEBUG: DEBUG mode 
        popSize: Number of agents in population
        timeHorizon: Number of days to run simulation
        betaDaily: Beta parameters for infection propagation 
        externalExposureProbDaily: Daily probability of getting exposed outside of the pop. 
        initialInfected: Number of agents initially infected 
        initProportionVaccinated : Number of agents initially vaccinated
        fractionSymptomatic: Fraction of infected agents who become symptomatic 
        infectiousViralLoadCut: Viral load needed to become infectious (cp/ml)
        daysTilSusceptible: Number of days until susceptible again after recovery
        quarantineLength: Number of days quarantine persists 
        selfIsolationOnSymptomsProb: Probability of agent self-isolating if symptomatic 
        vaccinesAvailablePerDay: Number of vaccines available to distribute 
        vaccineAcceptProbMean: Mean probability of accepting vaccination 
        vaccineAcceptProbStd: Standard deviation on probability of vaccination 
        vaccineInfectionProb: Probability of being infected if already vaccinated 
        poolSize: Pool size to be used for processing tests 
        poolingType: Pooling function to be used for computing results ('average' or 'exponential')
        noTestPostQuarantineDays: 0 for no blackout for testing postQuarantine; should be positive anyway; 
                                make it very large (at least as timeHorizon) to not test recovered
        fprSingle: False positive rate of a single test 
        fnrSingle: False negative rate of a single test 
        daysBetweenTesting: Days between administering tests 
        daysDelayTestResults: Delay in test results after test is administered 
        detectionCut: Viral load needed for detection on test (cp/ml)
        firstDayOfTesting: Day the first test is administered 
        testRunSeed: Random seed 
        trackArrays: track informative datafames 


        # Agent status labels:
        # S = susceptible
        # I = infected, not infectious
        # T = transmitting (infectious)
        # R = recovered
        # D = dead
        # TQ = transmitting, in quarantine
        # SQ = susceptible, in quarantine
        # RQ = recovered, in quarantine (due to onset-detection mismatch)
        #       (note: RQ not yet used as of 2021/09/29)

        """
        # Run details 
        self.version = '3.0.0'
        self.DEBUG = DEBUG
        self.popSize = popSize
        self.timeHorizon = timeHorizon
        self.infinity = 999999

        # Set seed 
        self.testRunSeed = testRunSeed
        if self.testRunSeed is not None:
            np.random.seed(self.testRunSeed)

        # Disease parameters 
        self.betaDaily = betaDaily
        self.initialInfected = initialInfected
        self.externalExposureProbDaily = externalExposureProbDaily
        self.fractionSymptomatic = fractionSymptomatic
        self.infectionInitTime = np.array([self.infinity]*self.popSize)
        self.infectiousViralLoadCut = infectiousViralLoadCut 
        self.symptomsDayStart = np.array([self.infinity]*self.popSize) 
        self.daysTilSusceptible = daysTilSusceptible 

       # Quarantine 
        self.quarantineLength = quarantineLength
        self.quarantineStart = np.array([-1]*self.popSize) # start of quarantine - may be delayed due to testing results delays
        self.quarantineUntil = np.array([-1]*self.popSize) # only changed upon testing!
        self.quarantineReason = np.array([None]*self.popSize) # values: TP, FP, None
        self.selfIsolationOnSymptomsProb = selfIsolationOnSymptomsProb

        # Vaccination
        self.isVaccinated = np.array([0]*popSize) #@@ start with no vaccinations - may change!
        self.initProportionVaccinated = initProportionVaccinated
        self.vaccinesAvailablePerDay = vaccinesAvailablePerDay
        self.vaccineAcceptProbMean = vaccineAcceptProbMean
        self.vaccineAcceptProbStd = vaccineAcceptProbStd
        self.vaccineAccept = np.array([np.random.normal(loc = self.vaccineAcceptProbMean, scale = self.vaccineAcceptProbStd)
                                         for _ in range(self.popSize)])
        self.vaccineInfectionProb = vaccineInfectionProb

        # Testing
        self.poolSize = poolSize
        self.poolingType = poolingType
        self.noTestingPostQuarantineDays = noTestingPostQuarantineDays
        self.noTestUntil = np.array([-1]*self.popSize) # only for those out of quarantine
        self.fprSingle = fprSingle
        self.fnrSingle = fnrSingle
        self.detectionCut = detectionCut
        self.daysBetweenTesting = daysBetweenTesting
        self.daysDelayTestResults = daysDelayTestResults
        self.firstDayOfTesting = self.daysBetweenTesting-1 if firstDayOfTesting is None else firstDayOfTesting

        # Setup population characteristics
        self.idArray = np.arange(popSize)
        self.statusArray = np.array(['S']*popSize,dtype='<U2')
        self.testCounter = [0]
        self.testCount = 0
        self.dailyStatus = []
        self.dayRecovered = np.array([self.infinity]*self.popSize)
        self.viralLoadArray = np.array([0.0]*popSize)
        self.positiveResultDate = np.array([self.infinity]*popSize)
        self.typeResult = np.array(['']*popSize, dtype='<U2')

        self.initialVaccinated = np.random.choice(self.idArray, int(self.initProportionVaccinated*popSize))
        self.isVaccinated[self.initialVaccinated] = 1

        self.nonQuarantineStatus = ['S','E','I','R']
        self.tick = -1

        self.viralLoadTimeline = np.zeros((self.popSize, self.timeHorizon))

        self.trackArrays = trackArrays
        self.tracking = dict() # list of dataframes to track...

    #==================================================================================================
    def trackSIM(self, whichsrc = 'vaccinate', data = dict()):
        tdt = pd.DataFrame(data)
        tdt['tick'] = self.tick
        if self.DEBUG: print(f"trackSIM: {whichsrc}, tdt.shape={tdt.shape}")
        if whichsrc not in self.tracking.keys():
            self.tracking[whichsrc] = [tdt]
        else:
            self.tracking[whichsrc].append(tdt)

    #==================================================================================================
    def exportTracking(self, outpath):
        if self.testRunSeed is not None:
            if not os.path.isdir(outpath):
                os.makedirs(outpath)
            for tc in self.tracking.keys():
                dt = pd.concat(self.tracking[tc])
                dt.to_csv(os.path.join(outpath, f"tracking_{tc}.csv"))

    #==================================================================================================
    def initializeInfection(self):
        # and assign initial infected
        self.initInfected = np.random.choice(self.idArray, self.initialInfected)
        self.statusArray[self.initInfected] = 'E'
        self.willSelfQuarantine = [True if np.random.uniform(0,1) < self.selfIsolationOnSymptomsProb else False for _ in self.idArray]

        tIDs = self.initInfected
        self.infectionInitTime[tIDs] = -np.random.randint(low = 0, high = 10, size = len(self.initInfected))
        self.viralLoadTimeline[tIDs,:], self.symptomsDayStart[tIDs] = computeAllViralLoads(len(tIDs), 
                                                                        self.fractionSymptomatic, 
                                                                        self.infectionInitTime[tIDs], 
                                                                        ticklen=1, 
                                                                        tickforecast=self.timeHorizon)
        self.newInfectionCount = self.initialInfected
        self.newOutsideInfectionCount = 0

        if self.trackArrays: 
            self.trackSIM(whichsrc = 'infection', data = dict(ID = tIDs))
        
    #==================================================================================================
    def monitorArrays(self, location = None):
        print("="*60)
        print(f"=============== {location}, tick: {self.tick} ======================")
        print("="*60)
        print(f"Counter(statusArray)={Counter(self.statusArray)}")
        print(f"Counter(quarantineStart)={Counter(self.quarantineStart)}")
        print(f"Counter(quarantineUntil)={Counter(self.quarantineUntil)}")
        print(f"Counter(quarantineReason)={Counter(self.quarantineReason)}")
        print(f"Counter(noTestUntil)={Counter(self.noTestUntil)}")
        print(f"Counter(isVaccinated)={Counter(self.isVaccinated)}")
        print("="*60)

    #==================================================================================================
    def computeDiseaseState(self):
        """
        Computes the new viral load for those infected, so their status and infectiousness can be updated
        """
        # filter to those currently infected
        tfilt = self.infectionInitTime != self.infinity
        tIDs = self.idArray[tfilt]

        self.viralLoadArray = self.viralLoadTimeline[:, self.tick]
        if self.trackArrays: 
            self.trackSIM(whichsrc = 'computeDiseaseState', data = dict(ID = self.idArray, viralLoad = self.viralLoadArray))

    #==================================================================================================
    def changeStatus(self):
        if self.DEBUG: self.monitorArrays(location = 'start changeStatus')

        #=================================
        # Non-quarantined becoming infectious
        #=================================
        tfilt = (self.quarantineUntil == -1) & (self.viralLoadArray > self.infectiousViralLoadCut)
        self.statusArray[tfilt] = 'I'
        if self.DEBUG: print(f"changeStatus: tick={self.tick}, {sum(tfilt)} changed from non-Q to I: {self.idArray[tfilt]}")
        del tfilt

        #=================================
        # Anybody getting positive results and not already in quarantine => enter quarantine
        #=================================
        tfilt = (self.quarantineUntil == -1) & (self.positiveResultDate == self.tick)
        # True positive results
        tpfilt = tfilt & (self.typeResult=='TP')
        self.statusArray[tpfilt] = 'TI'
        self.quarantineUntil[tpfilt] = self.tick + self.quarantineLength
        self.newTI = sum(tpfilt)
        if self.DEBUG: print(f"changeStatus: {sum(tpfilt)} changed from TP to TI")
        del tpfilt
        # False positive results
        fpfilt = tfilt & (self.typeResult=='FP')
        self.statusArray[fpfilt] = 'FI'
        self.quarantineUntil[fpfilt] = self.tick + self.quarantineLength
        self.newFI = sum(fpfilt)
        if self.DEBUG: print(f"changeStatus: {sum(fpfilt)} changed from FP to FI")
        del fpfilt
        # either way, no tests until...
        self.noTestUntil[tfilt] = self.tick + self.quarantineLength + self.noTestingPostQuarantineDays
        del tfilt
        # reset result types for those just receiving them
        tfilt = (self.positiveResultDate == self.tick)
        self.positiveResultDate[tfilt] = self.infinity
        self.typeResult[tfilt] = ''
        del tfilt

        #=================================
        # Quarantine exits - multiple reasons (FP/TP) and viral loads possible!
        #=================================
        tfilt = (self.quarantineUntil == self.tick)
        # still infectious, regardless of current status
        tfilt_t = tfilt & (self.viralLoadArray >= self.infectiousViralLoadCut)
        if self.DEBUG: print(f"changeStatus: {Counter(self.statusArray[tfilt_t])} changed to I")
        self.statusArray[tfilt_t] = 'I'
        # true quarantined, non-infectious post-peak
        tfilt_r = tfilt & (self.viralLoadArray < self.infectiousViralLoadCut) & (self.statusArray == 'TI')
        if self.DEBUG: print(f"changeStatus: {Counter(self.statusArray[tfilt_r])} changed to R")
        self.statusArray[tfilt_r] = 'R'
        self.dayRecovered[tfilt_r] = self.tick
        # false quarantined, non-infectious post-peak
        tfilt_r = tfilt & (self.viralLoadArray < self.infectiousViralLoadCut) & (self.statusArray == 'FI')
        if self.DEBUG: print(f"changeStatus: {Counter(self.statusArray[tfilt_t])} changed to S")
        self.statusArray[tfilt_r] = 'S'
        # either way, get them out of quarantine...
        self.quarantineReason[tfilt] = None
        self.quarantineUntil[tfilt] = -1
        self.symptomsDayStart[tfilt] = self.infinity
        # ... and shelter from testing (if so setup!)
        self.noTestUntil[tfilt] = self.tick + self.noTestingPostQuarantineDays

        #=================================
        # Infectious non-quarantined recovering
        #=================================
        tfilt = (self.statusArray == 'I') & (self.viralLoadArray < self.infectiousViralLoadCut)
        self.statusArray[tfilt] = 'R'
        self.dayRecovered[tfilt] = self.tick
        self.quarantineUntil[tfilt] = -1
        if self.DEBUG: print(f"changeStatus: tick={self.tick}, {sum(tfilt)} changed from I to R {self.idArray[tfilt]}")

        # Reset blackout period for those exiting it
        tfilt = (self.noTestUntil == self.tick)
        self.noTestUntil[tfilt] = -1

        #=================================
        # Recovered becoming susceptible again
        #=================================
        tfilt = (self.statusArray == 'R') & ((self.dayRecovered + self.daysTilSusceptible) == self.tick)
        self.statusArray[tfilt] = 'S'

        if self.DEBUG: self.monitorArrays(location = 'End changeStatus')

        # tracking
        if self.trackArrays: 
            self.trackSIM(whichsrc = 'changeStatus', data = dict(ID = self.idArray, status = self.statusArray))

    #==================================================================================================
    def vaccinate(self):
        # who gets to be vaccinated? depends on:
        #   number of vaccines available, 
        #   who is already vaccinated
        #   propensity to vaccinate among non-vaccinated
        #   (later) disease state (e.g. infected or recent recovery)
        tfilt = (self.isVaccinated == 0) & (self.quarantineUntil <= self.tick) 

        tfilt = tfilt & ([np.random.uniform(0,1) <= x for x in self.vaccineAccept])
        if self.DEBUG: print(f"vaccination status: {Counter(tfilt)}")
        if sum(tfilt) >= self.vaccinesAvailablePerDay:
            newVaccinations = np.random.choice(self.idArray[tfilt], self.vaccinesAvailablePerDay, replace = False)
        else:
            newVaccinations = self.idArray[tfilt]
        if self.DEBUG: print(f"newVaccinations on tick {self.tick}: {len(newVaccinations)} of {self.vaccinesAvailablePerDay} available")
        self.isVaccinated[newVaccinations] = 1

        #tracking
        if self.trackArrays: 
            if len(newVaccinations)>0:
                self.trackSIM(whichsrc = 'vaccinate', data = dict(ID = newVaccinations))

    #==================================================================================================
    def selfIsolate(self):
        """
        Assumption: only symptomatics will self-isolate; no false positive self-isolation for now
        """
        # symptomatics not detected have quarantineUntil = -1
        tfilt = (self.tick >= self.symptomsDayStart) & (self.quarantineUntil == -1)  & (self.willSelfQuarantine) 
        tids = self.idArray[tfilt]

        if self.DEBUG: print(f"selfIsolate: {len(tids)} self-isolated from {Counter(self.statusArray[tids])} to TI")
        self.statusArray[tids] = 'TI'
        self.newTI = self.newTI + len(tids)
        self.quarantineUntil[tids] = self.tick + self.quarantineLength
        self.noTestUntil[tids] = self.tick + self.quarantineLength + self.noTestingPostQuarantineDays

        # tracking
        if self.trackArrays: 
            if len(tids)>0:
                self.trackSIM(whichsrc = 'selfIsolate', data = dict(ID = tids))

    #==================================================================================================
    def testingResults(self, detectionCut = None):
        """
        Assume that those in quarantine or post-quarantine "blackout" have noTestUntil != -1
        """
        # testing eligible - use noTestUntil array
        testableFilt = (self.noTestUntil == -1)
        testableIDs = self.idArray[testableFilt]
        # actual testing
        tresults = testingDay(
            ctIDs = testableIDs,
            viralLoadArray = self.viralLoadArray,
            detectionCut = detectionCut,
            poolSize = self.poolSize,
            poolFunType = self.poolingType,
            testAccuracyArray = dict(fprSingle = self.fprSingle, fnrSingle = self.fnrSingle),
            DEBUG = self.DEBUG
        )
        self.testCount = tresults['testCount']
        # TP and FP will get positiveResult notifications later on
        tfilt = tresults['FPIDs'] + tresults['TPIDs']
        self.positiveResultDate[tfilt] = [self.tick + self.daysDelayTestResults + 1] # testing results take effect the following day
        self.typeResult[tresults['FPIDs']] = 'FP'
        self.typeResult[tresults['TPIDs']] = 'TP'
        self.typeResult[tresults['FNIDs']] = 'FN'
        self.typeResult[tresults['TNIDs']] = 'TN'
        if self.DEBUG:
            print('='*40+f' testingResults for tick={self.tick} ' + '='*40)
            print(dict(
                FPs=len(tresults['FPIDs']),
                TPs=len(tresults['TPIDs']),
                FNs=len(tresults['FNIDs']),
                TNs=len(tresults['TNIDs']),
            ))
            print(f"Counter(self.typeResult)={Counter(self.typeResult)}")


    #==================================================================================================
    def outsideInfection(self): 
        '''
        Document agents infected outside the population 
        '''
        currentlySusceptible = sum(self.statusArray == 'S')

        # how many are potentially getting infected outside population 
        newExposedCount = int(np.round(self.externalExposureProbDaily * currentlySusceptible))

        # who is potentially getting infected 
        newExposed = np.random.choice(self.idArray[self.statusArray == 'S'], newExposedCount, replace = False)

        if len(newExposed) > 0: 
            # vaccinated individuals have lower chance of infection 
            newInfected = [x for x in newExposed if ((self.isVaccinated[x]==0) or (np.random.uniform(0,1) < self.vaccineInfectionProb))]

            self.newOutsideInfectionCount = len(newInfected)
            self.statusArray[newInfected] = 'E'

            self.infectionInitTime[newInfected] = self.tick
            
            # pre-compute the infection evolution8
            if len(newInfected) > 0:
                tIDs = newInfected
                t0 = dt.datetime.now()

                self.viralLoadTimeline[tIDs,self.tick:self.timeHorizon], self.symptomsDayStart[tIDs] = \
                    computeAllViralLoads(len(tIDs), 
                                        self.fractionSymptomatic, 
                                        tD=self.infectionInitTime[tIDs], 
                                        tickforecast=self.timeHorizon-self.tick)  
                t1 = dt.datetime.now()

            # tracking
            if self.trackArrays: 
                self.trackSIM(whichsrc = 'infection', data = dict(ID = newInfectedTrim))


    #==================================================================================================
    def infectionPropagation(self):
        '''
        Propagate infection within the population 
        '''
        currentlySusceptible = sum(self.statusArray == 'S')
        # how many are infectious free-roaming?
        tfilt = (self.viralLoadArray > self.infectiousViralLoadCut) & (self.quarantineUntil == -1)
        currentlyInfectious = sum(tfilt)

        # how many are getting infected
        newExposedCount = int(np.round(currentlyInfectious*self.betaDaily*currentlySusceptible/self.popSize))

        if self.DEBUG: print(f"currentlyInfectious={currentlyInfectious}, newExposedCount={newExposedCount}, currentlySusceptible={currentlySusceptible}")
        
        # who is getting infected
        #@@ NOTE the potential for duplicated infection (by design!)
        newExposed = np.random.choice(self.idArray[self.statusArray == 'S'], newExposedCount, replace = True)
        # deduplication
        newExposed = np.array(list(set(newExposed)))
        self.newInfectionCount = len(newExposed)
        if len(newExposed) > 0:
            if self.DEBUG: print(f"tick={self.tick}, {self.idArray[newExposed]} ({len(newExposed)}) initially selected to be infected!")
            # lower/remove infection probability for those vaccinated
            newInfected = [x for x in newExposed if ((self.isVaccinated[x]==0) or (np.random.uniform(0,1) < self.vaccineInfectionProb))]
            self.newInfectionCount = len(newInfected)
            if self.DEBUG: print(f"tick={self.tick}, {self.idArray[newInfected]} ({len(newInfectedTrim)}) finally infected after vaccination trim")
            # infect the selected ones
            self.statusArray[newInfected] = 'E'

            self.infectionInitTime[newInfected] = self.tick
            # pre-compute the infection evolution8
            if len(newInfected) > 0:
                tIDs = newInfected
                t0 = dt.datetime.now()

                self.viralLoadTimeline[tIDs,self.tick:self.timeHorizon], self.symptomsDayStart[tIDs] = \
                    computeAllViralLoads(len(tIDs), 
                                        self.fractionSymptomatic, 
                                        tD=self.infectionInitTime[tIDs], 
                                        tickforecast=self.timeHorizon-self.tick)  
                t1 = dt.datetime.now()
                if self.DEBUG: print(f"infectionPropagation, tick={self.tick} spent {t1-t0}")
            # tracking
            if self.trackArrays: 
                self.trackSIM(whichsrc = 'infection', data = dict(ID = newInfectedTrim))

    #==================================================================================================
    def trackStatus(self):
        tdStatus = Counter(self.statusArray)
        tdStatus['tick']=self.tick

        # add vaccinations
        tdStatus['vaccinated'] = sum(self.isVaccinated)

        # record isolation 
        try: 
            tdStatus['newFI'] = self.newFI
            tdStatus['newTI'] = self.newTI
        except: 
            tdStatus['newFI'] = 0
            tdStatus['newTI'] = 0
            # initialize compartments 
            if tdStatus['tick'] == -1: 
                tdStatus['FI'] = tdStatus['newFI']
                tdStatus['TI'] = tdStatus['newTI']

        tdStatus['newInfections'] = self.newInfectionCount
        tdStatus['newOutsideInfections'] = self.newOutsideInfectionCount

        if (((self.tick - self.firstDayOfTesting) % self.daysBetweenTesting == 0) 
                and (self.tick >= self.firstDayOfTesting)):
            tdStatus['test'] = 1
            self.testCounter.append(self.testCounter[-1] + self.testCount)
        else:
            tdStatus['test'] = 0
            self.testCounter.append(self.testCounter[-1])
        tdStatus['testCount'] = self.testCount
        self.dailyStatus.append(tdStatus.copy())
        self.testCount = 0

    #==================================================================================================
    def runOneTick(self, detectionCut = 0):

        # increment tick
        self.tick += 1

        # infection outside of population 
        self.outsideInfection()

        # compute disease state evolution (viral load)
        self.computeDiseaseState()

        # changeStatus (update infectiousness, symptoms etc)
        self.changeStatus()

        # selfIsolation due to symptoms onset
        self.selfIsolate()

        # testing (if a testing day)
        if ((self.tick - self.firstDayOfTesting) % self.daysBetweenTesting == 0 
                and self.tick >= self.firstDayOfTesting):
            self.testingResults(detectionCut)

        # infection propagation
        self.infectionPropagation()

        # vaccination
        self.vaccinate()

        # at the end of each tick, keep track of the arrays situation
        if self.DEBUG: self.monitorArrays(location = 'end iteration')
        self.trackStatus()

    #==================================================================================================
    def runFullTime(self):
        self.initializeInfection()
        self.trackStatus()
        for ttick in range(self.timeHorizon):
            if ttick % 10 == 0: print(ttick)
            self.runOneTick(self.detectionCut)
