#!/usr/bin/env python
#
# Import libraries
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
import mne

# Script to pull demographic and data-related information for participants

homeDir = os.path.expanduser("~")
dataDir = homeDir + '/camcan/'
behaviouralDir = dataDir + 'behaviouralData/'
megDir = dataDir + 'megData_moveComp/'
outDir = dataDir + 'proc_data/TaskSensorAnalysis_transdef/'
plotOK = 0

####################################################################
# Pull demographic information from the CamCAN supplied spreadsheet

demographicFile = behaviouralDir + 'participant_data.csv'
csvfile = open(demographicFile )
reader = csv.DictReader (csvfile)
subjectData = []
subjectIDs = []
for row in reader:

    # Read demographic data
    subjectID = row['Observations']
    age = int(row['age'])
    hand = float(row['hand'])
    gender = int(row['gender_code'])

    subjectIDs.append(int(subjectID[2:]))
    subjectData.append({'SubjectID':subjectID, 'Age':age, 'Hand':hand, 'Gender':gender})

# Put all this information into a pandas dataframe 
subjectData_pd = pd.DataFrame(subjectData, index=subjectIDs)

###########################################################################
# Check who has data for the motor task

rawExists = []
for subjectID in subjectData_pd.index:

    rawFif = megDir + '/CC' + str(subjectID) + '/task/task_raw.fif'
    if os.path.exists(rawFif):
        rawExists.append(1)
    else:
        rawExists.append(0)

# Add these findings to the pandas dataframe
subjectData_pd['RawExists'] = pd.Series(rawExists, index=subjectData_pd.index)


###########################################################################
# Get some info for reporting

withRawData = subjectData_pd.loc[subjectData_pd['RawExists'] == 1]
print('There is raw data for ' + str(len(withRawData)) + ' participants')

###########################################################################
## Get event descriptive stats

goodSubjectData = subjectData_pd.copy()

dataReads = []
ITIMean = []
ITI_SD = []
RTMean = []
RT_SD = []
numberStimulus = []
numberGoodResponses = []
for subjectID in goodSubjectData.index:

    if subjectData_pd['RawExists'][subjectID]:
        print(str(subjectID))
        # Get events
        rawFif = megDir + '/CC' + str(subjectID) + '/task/task_raw.fif'
        try:
            raw = mne.io.Raw(rawFif)
        except ValueError:
            dataReads.append(0)
        else:
            dataReads.append(1)
            evs = mne.find_events(raw, 'STI101', shortest_event=1)
            # Get stimuli latencies
            evtID = evs[:,2]
            stim = np.where(evtID<10)
            stimLatency = evs[stim,0]/raw.info['sfreq']
            # Find inter-trial intervals
            ITIs = np.diff(stimLatency)
            ITIMean.append(np.mean(ITIs))
            ITI_SD.append(np.std(ITIs))
            # Get response latencies
            response = np.where(evtID>10)
            responseLatency = evs[response,0]/raw.info['sfreq']
            # Find delay to response for each stimulus
            RTs = []
            for onset in stimLatency[0]:
                allRTs = responseLatency-onset
                positiveRTIndex = np.where(allRTs>0)
                if len(positiveRTIndex[0])>0:
                    positiveRTs = allRTs[positiveRTIndex]
                    firstRT = positiveRTs[0]
                    if firstRT < 2:
                        RTs.append(firstRT)
            RTMean.append(np.mean(RTs))
            RT_SD.append(np.std(RTs))
            numberStimulus.append(len(stim[0]))
            numberGoodResponses.append(len(RTs))

# Add results to pandas dataframe - drop subjects with no task data file
goodSubjectData2 = goodSubjectData.copy()
goodSubjectData2 = goodSubjectData2.loc[goodSubjectData2['RawExists']==1]

# Drop subjects with unreadable task data files
goodSubjectData2['DataReads'] = pd.Series(dataReads, index=goodSubjectData2.index)
goodSubjectData2 = goodSubjectData2.loc[goodSubjectData2['DataReads']==1]

# Add other information to the pandas dataframe
goodSubjectData2['ITIMean'] = pd.Series(ITIMean, index=goodSubjectData2.index)
goodSubjectData2['ITI_SD'] = pd.Series(ITI_SD, index=goodSubjectData2.index)
goodSubjectData2['RTMean'] = pd.Series(RTMean, index=goodSubjectData2.index)
goodSubjectData2['RT_SD'] = pd.Series(RT_SD, index=goodSubjectData2.index)
goodSubjectData2['numStimuli'] = pd.Series(numberStimulus, index=goodSubjectData2.index)
goodSubjectData2['numGoodResponses'] = pd.Series(numberGoodResponses, index=goodSubjectData2.index)

goodSubjectData3 = goodSubjectData2.copy()
goodSubjectData3 = goodSubjectData3.loc[goodSubjectData3['numStimuli']>0]

#######################################################################
# Write files and make standard output

subjectData_pd.to_csv(outDir + 'demographics_allSubjects.csv')
goodSubjectData3.to_csv(outDir + 'demographics_goodSubjects.csv')

print('There is still good data for ' + str(len(goodSubjectData3)) + ' participants')
print('Mean Inter-Trial Interval Across Subjects is ' + str(np.nanmean(goodSubjectData3['ITIMean'].tolist())) + ' seconds.')
print('Minimum Number of Stimuli Across Subjects is ' + str(np.min(goodSubjectData3['numStimuli'].tolist())) + '.')
print('Maximum Number of Stimuli Across Subjects is ' + str(np.max(goodSubjectData3['numStimuli'].tolist())) + '.')
print('Mean Number of Stimuli Across Subjects is ' + str(np.nanmean(goodSubjectData3['numStimuli'].tolist())) + '.')
print('StdDev = ' + str(np.nanstd(goodSubjectData3['numStimuli'].tolist())))
print('Mean Response Time Across Subjects is ' + str(np.nanmean(goodSubjectData3['RTMean'].tolist())) + ' seconds.')
print('Minimum Number of Responses Across Subjects is ' + str(np.min(goodSubjectData3['numGoodResponses'].tolist())) + '.')
print('Maximum Number of Responses Across Subjects is ' + str(np.max(goodSubjectData3['numGoodResponses'].tolist())) + '.')
print('Mean Number of Responses Across Subjects is ' + str(np.nanmean(goodSubjectData3['numGoodResponses'].tolist())) + '.')
print('StdDev = ' + str(np.nanstd(goodSubjectData3['numGoodResponses'].tolist())))


