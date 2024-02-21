'''
Visualize and analyze results of ABM runs.

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
from matplotlib import colors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#======================================================================================================
# Aux func
def get_ids(df, fprSingle, propVax, #fnrSingle, detectionCut, 
        poolSize, daysBetweenTesting, vaxAvail, firstDay=7):
    '''
    Get ids in dataframe matching condition
    '''
    filtered_df = df[df.fprSingle.eq(fprSingle) & df.initProportionVaccinated.eq(propVax) & 
                        df.poolSize.eq(poolSize) & df.vaccinesAvailablePerDay.eq(vaxAvail) & 
                        df.daysBetweenTesting.eq(daysBetweenTesting) & df.firstDayOfTesting.eq(firstDay)]
    return filtered_df.id


#======================================================================================================
# Run ID and output directory
run_prefix = 'covid19_paper_scenarios_beta04'
dataPath = os.path.dirname(__file__)
DATA = os.path.join(dataPath, "output")

plotsFld = os.path.join(DATA, run_prefix, 'plots')
if not os.path.isdir(plotsFld):
    os.makedirs(plotsFld)


#======================================================================================================
# Read and organize data
setup_information = pd.read_csv(os.path.join(DATA, f'{run_prefix}_samplesDF.csv'))
runs = pd.read_pickle(os.path.join(DATA, f'{run_prefix}_samples_full.pkl'))

for k in setup_information.columns[:-2]:
    print(f'{k}: {set(setup_information[k])}')

FPR = sorted(set(setup_information['fprSingle']))
#FNR = sorted(set(setup_information['fnrSingle']))
#detectionCut = sorted(set(setup_information['detectionCut']))
propVax = sorted(set(setup_information['initProportionVaccinated']))
ps = sorted(set(setup_information['poolSize']))
#beta = sorted(set(setup_information['betaDaily']))
testInterval = sorted(set(setup_information['daysBetweenTesting']))
vaxAvail = sorted(set(setup_information['vaccinesAvailablePerDay']))
firstDayOfTesting = sorted(set(setup_information['firstDayOfTesting']))

scenarios = [[FPR[0], propVax[0], ps[0], testInterval[0], vaxAvail[0], 120], # baseline - no tests
            [FPR[0], propVax[0], ps[0], testInterval[0], vaxAvail[0]], # PCR, no vaccines 
            [FPR[0], propVax[0], ps[1], testInterval[0], vaxAvail[0]],
            [FPR[0], propVax[0], ps[0], testInterval[1], vaxAvail[0]],
            [FPR[0], propVax[0], ps[1], testInterval[1], vaxAvail[0]],
            [FPR[1], propVax[0], ps[0], testInterval[0], vaxAvail[0]], # Rapid test, no vaccines 
            [FPR[1], propVax[0], ps[1], testInterval[0], vaxAvail[0]],
            [FPR[1], propVax[0], ps[0], testInterval[1], vaxAvail[0]],
            [FPR[1], propVax[0], ps[1], testInterval[1], vaxAvail[0]],
            [FPR[0], propVax[1], ps[0], testInterval[0], vaxAvail[1]], # PCR test, continued vaccine rollout  
            [FPR[0], propVax[1], ps[1], testInterval[0], vaxAvail[1]],
            [FPR[0], propVax[1], ps[0], testInterval[1], vaxAvail[1]],
            [FPR[0], propVax[1], ps[1], testInterval[1], vaxAvail[1]],
            [FPR[1], propVax[1], ps[0], testInterval[0], vaxAvail[1]], # Rapid test, continued vaccine rollout  
            [FPR[1], propVax[1], ps[1], testInterval[0], vaxAvail[1]],
            [FPR[1], propVax[1], ps[0], testInterval[1], vaxAvail[1]],
            [FPR[1], propVax[1], ps[1], testInterval[1], vaxAvail[1]],
            [FPR[0], propVax[0], ps[0], testInterval[0], vaxAvail[1]], # PCR test, vaccine rollout 
            [FPR[0], propVax[0], ps[1], testInterval[0], vaxAvail[1]],
            [FPR[0], propVax[0], ps[0], testInterval[1], vaxAvail[1]],
            [FPR[0], propVax[0], ps[1], testInterval[1], vaxAvail[1]],
            [FPR[1], propVax[0], ps[0], testInterval[0], vaxAvail[1]], # Rapid test, vaccine rollout
            [FPR[1], propVax[0], ps[1], testInterval[0], vaxAvail[1]],
            [FPR[1], propVax[0], ps[0], testInterval[1], vaxAvail[1]],
            [FPR[1], propVax[0], ps[1], testInterval[1], vaxAvail[1]]]

tests = []
broken_axes = False
plotScenarios = True
plotComparisons = True 

#==================================================
# Plot all scenarios
if plotScenarios: 
    for s in scenarios:
        s_names = ['FPR','propVax','PS','TI','vaxAvail']
        totalTQ = []
        totalFQ = []
        totalInfections = []
        ids = get_ids(setup_information, *s)   
        s_runs = runs[runs['id'].isin(ids)]
        if broken_axes: 
            fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [2, 1.5]}, figsize=(5,3.5))
            fig.subplots_adjust(hspace=0.025)
        else:
            fig, ax1 = plt.subplots(1,1)
        for i,id in enumerate(ids):
            totalTQ.append(s_runs[s_runs['id']==id]['newTI'].sum())
            totalFQ.append(s_runs[s_runs['id']==id]['newFI'].sum())
            totalInfections.append(s_runs[s_runs['id']==id]['newInfections'].sum())

            s_runs[s_runs['id']==id].plot('tick',
                                            ['S','E','I','R','FI','TI'], 
                                            color=['lightblue','peachpuff','sandybrown','lightcoral','thistle','lightgreen'], 
                                            alpha=0.5,
                                            #label = ['susceptible','infected','transmitting','recovered','falsely quarantined','quarantined while transmitting'],
                                            legend=False,
                                            ax=ax1)
            if broken_axes: 
                s_runs[s_runs['id']==id].plot('tick',
                                            ['S','E','I','R','FI','TI'], 
                                            color=['lightblue','peachpuff','sandybrown','lightcoral','thistle','lightgreen'], 
                                            alpha=0.5,
                                            #label = ['susceptible','infected','transmitting','recovered','falsely quarantined','quarantined while transmitting'],
                                            legend=False,
                                            ax=ax2)

        avg_run = s_runs.groupby('tick').mean()
        avg_run['testCount'] = avg_run.testCount.cumsum()
        avg_run.plot(y=['S','E','I','R','FI','TI'],
                        ax=ax1,
                        color=['tab:blue','tab:orange','tab:brown','tab:red','tab:purple','tab:green']) 
                        #label = ['susceptible','infected','transmitting','recovered','falsely quarantined','quarantined while transmitting'])
        if broken_axes: 
            avg_run.plot(y=['S','E','I','R','FI','TI'],
                        ax=ax2,
                        legend=False,
                        color=['tab:blue','tab:orange','tab:brown','tab:red','tab:purple','tab:green']) 
                        #label = ['susceptible','infected','transmitting','recovered','falsely quarantined','quarantined while transmitting'])
            ax1.set_ylim(8000, 10000) 
            ax2.set_ylim(0, 1400)  
            ax1.spines.bottom.set_visible(False)
            ax2.spines.top.set_visible(False)
            ax1.xaxis.tick_top()
            ax1.tick_params(labeltop=False)  
            ax2.xaxis.tick_bottom()
            kwargs = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
            ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
            ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
            ax2.set_xlabel('Simulation days')
            fig.text(0.02, 0.5, 'Number of agents', ha='center', va='center', rotation='vertical')
            fig.tight_layout()

        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 100
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.savefig(os.path.join(plotsFld, f"""{run_prefix}_scenario_{"_".join([x+"{:.3f}".format(y).replace('.','') for x,y in zip(s_names, s)])}.png"""))
        plt.close()

        print(f'''{"_".join([x+'.'+"{:.3f}".format(y) for x,y in zip(s_names, s)])}''')
        print(f'Total TQ: {sum(totalTQ)/len(totalTQ)}')
        print(f'Total FQ: {sum(totalFQ)/len(totalFQ)}')
        print(f'Total Infections: {sum(totalInfections)/len(totalInfections)}')
        print(f'''Test Count: {avg_run['testCount'].iloc[-1]}''')

        tests.append(avg_run['testCount'])
        test_ax = avg_run.plot(y='testCount')
        test_ax.set_ylabel('Tests')
        plt.savefig(os.path.join(plotsFld, f"""{run_prefix}_test_{"_".join([x+"{:.3f}".format(y).replace('.','') for x,y in zip(s_names, s)])}.png"""))
        plt.close()

        vax_ax = avg_run.plot(y='vaccinated')
        vax_ax.set_ylabel('Number of vaccinated')
        vax_ax.set_ylim([0,10000])
        plt.savefig(os.path.join(plotsFld, f"""{run_prefix}_vax_{"_".join([x+"{:.3f}".format(y).replace('.','') for x,y in zip(s_names, s)])}.png"""))
        plt.close()

#==================================================
# Plot comparisons
if plotComparisons: 
    vax = [(propVax[0],vaxAvail[0]), (propVax[0],vaxAvail[1]), (propVax[1],vaxAvail[1])]
    scenarios = [[FPR[0], ps[0], testInterval[0], firstDayOfTesting[1]], # no tests 
            [FPR[0], ps[0], testInterval[0], firstDayOfTesting[0]],
            [FPR[0], ps[1], testInterval[0], firstDayOfTesting[0]],
            [FPR[0], ps[0], testInterval[1], firstDayOfTesting[0]], 
            [FPR[0], ps[1], testInterval[1], firstDayOfTesting[0]],
            [FPR[1], ps[0], testInterval[0], firstDayOfTesting[0]], 
            [FPR[1], ps[1], testInterval[0], firstDayOfTesting[0]],
            [FPR[1], ps[0], testInterval[1], firstDayOfTesting[0]],
            [FPR[1], ps[1], testInterval[1], firstDayOfTesting[0]]]
    scenario_key = pd.DataFrame([list(a) + list(b) for a in vax for b in scenarios],columns=['propVax','vaxAvail','test','ps','testFreq','firstTest'])           
    # Compare vaccines/test count for across scenarios 
    for p, v in vax: 
        fig_fq, ax_fq = plt.subplots(1,1, figsize=(4,3)) 
        fig_tests, ax_tests = plt.subplots(1,1, figsize=(4,3)) 
        fig_inf, ax_inf = plt.subplots(1,1, figsize=(4,3)) 

        summary = {'test cost per person per day':[], 'total infections':[], 'false isolation':[]}
        summary_labels = []
        for s in scenarios: 
            ids = get_ids(setup_information, s[0], p, s[1], s[2], v, s[3])
            s_label = scenario_key[scenario_key['propVax'].eq(p) & 
                            scenario_key['vaxAvail'].eq(v) &
                            scenario_key['test'].eq(s[0]) & 
                            scenario_key['ps'].eq(s[1]) & 
                            scenario_key['testFreq'].eq(s[2]) &
                            scenario_key['firstTest'].eq(s[3])].index[0] + 1
            if s[0] == 0.014: 
                test_id = 'A'
                test_cost = 100
            else:
                test_id = 'B' 
                test_cost = 50
            s_runs = runs[runs['id'].isin(ids)]
            avg_run = s_runs.groupby('tick').mean()
            avg_run['testCost'] = avg_run.testCount.cumsum() * test_cost / (10000 * 120)
            avg_run['infections'] = avg_run.newInfections.cumsum()
            avg_run['totalFI'] = avg_run.newFI.cumsum()
            avg_run.plot(y=['testCost'],
                    ax=ax_tests,
                    legend=True,
                    #color=['tab:blue','tab:orange','tab:brown','tab:red','tab:purple','tab:green'], 
                    label = [s_label])#[f'test {test_id}, pool size {s[1]}, testing interval {s[2]} days'])        
            avg_run.plot(y=['infections'],
                    ax=ax_inf,
                    legend=True,
                    #color=['tab:blue','tab:orange','tab:brown','tab:red','tab:purple','tab:green'], 
                    label = [s_label]) #[f'test {test_id}, pool size {s[1]}, testing interval {s[2]} days'])          
            avg_run.plot(y=['totalFI'],
                    ax=ax_fq,
                    legend=True,
                    #color=['tab:blue','tab:orange','tab:brown','tab:red','tab:purple','tab:green'], 
                    label = [s_label]) #[f'test {test_id}, pool size {s[1]}, testing interval {s[2]} days'])   
            
            if s[3]==120: 
                text_label = 'No\ntesting'
            else: 
                text_label = f'Test {test_id}\nPS {s[1]}\n{s[2]} days'

            summary['test cost per person per day'].append(avg_run['testCost'].iloc[-1])
            summary['total infections'].append(avg_run['infections'].iloc[-1])
            summary['false isolation'].append(avg_run['totalFI'].iloc[-1])
            summary_labels.append(text_label)

        ax_inf.set_ylabel('Total infections')
        ax_inf.set_xlabel('Simulation days')
        fig_inf.tight_layout()
        plt.rcParams['figure.dpi'] = 500
        plt.rcParams['savefig.dpi'] = 500
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.savefig(os.path.join(plotsFld, f"vax_{str(p).replace('.','')}_{v}_infection_comparisons.png"))
        plt.close()

        ax_tests.set_ylabel('Test cost per person per day')
        ax_tests.set_xlabel('Simulation days')
        fig_tests.tight_layout()
        plt.rcParams['figure.dpi'] = 500
        plt.rcParams['savefig.dpi'] = 500
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.savefig(os.path.join(plotsFld, f"vax_{str(p).replace('.','')}_{v}_test_comparisons.png"))
        plt.close()

        ax_fq.set_ylabel('Total falsely quarantined')
        ax_fq.set_xlabel('Simulation days')
        plt.rcParams['figure.dpi'] = 500
        plt.rcParams['savefig.dpi'] = 500
        plt.rcParams['savefig.bbox'] = 'tight'
        fig_fq.tight_layout()
        plt.savefig(os.path.join(plotsFld, f"vax_{str(p).replace('.','')}_{v}_fq_comparisons.png"))
        plt.close()

        ## bar chart 
        x = np.arange(len(scenarios))  # the label locations
        width = 0.2  # the width of the bars
        multiplier = 0
        fig, ax = plt.subplots(layout='constrained', figsize=(8,5))
        ax2 = ax.twinx()
        fmt = ['${:,.2f}', '{:.0f}', '{:.0f}']

        offset = width * multiplier
        rects_iso = ax.bar(x + offset, summary['false isolation'], width, label='false isolations', alpha=0.5)

        multiplier += 1
        offset = width * multiplier
        rects_inf = ax.bar(x + offset, summary['total infections'], width, label='total infections', alpha=0.5)

        multiplier += 1
        offset = width * multiplier
        rects_test = ax2.bar(x + offset, summary['test cost per person per day'], width, label='test cost', color='green', alpha=0.5)
        
        ax.bar_label(rects_inf, fmt=fmt[1], padding=3)
        ax.bar_label(rects_iso, fmt=fmt[2], padding=3)
        ax2.bar_label(rects_test, fmt=fmt[0], padding=3)
        

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Quantity')
        ax.set_ylim(0, max(summary['total infections'])+2500)
        ax2.set_ylabel('Cost per person per day')
        ax.set_title('Scenario comparison')
        ax.set_xticks(x + width, summary_labels)
        ax.legend(loc='upper left', ncols=2)      
        ax2.legend(loc='upper right', ncols=1) 
        plt.savefig(os.path.join(plotsFld, f"vax_{str(p).replace('.','')}_{v}_bar_comparison.png"), dpi=300)


