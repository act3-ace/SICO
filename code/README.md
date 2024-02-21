# SICO - Code Organization 

* `code/*` : scripts for defining scenarios, running simulations, and visualizing results 
  * `paperMultiSIMs.py` : Generate scenarios from the paper 
  * `analyzeMultiSIMResults.py` : Generate figures and process results associated with ABM simulation batches 
  * `singleRunABM.py` : script defining a single scenario
  * `newDiseaseRun.py` : script defining a single scenario describing the early stages of the spread of a new disease   

* `code/utils/*` : helper classes and functions 
  * `ABM.py` : main class for running simulations 
  * `multiSIMs.py` : class for running batches of ABM simulations 
  * `viralLoadEvolution.py` : module for computing viral load profiles of agents  
  * `testingDay.py` : simulate administration of tests 
  * `testGroup.py` : test pool

### To recreate results from the paper: 
1. Run `paperMultiSIMs.py` to perform simulations
2. Run `analyzeMultiSIMs.py` (pointing to the `run_prefix` from 1) to get corresponding results  