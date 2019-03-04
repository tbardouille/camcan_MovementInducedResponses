#!/usr/bin/env python

# Import libraries
import os, sys
import numpy as np
import pandas as pd
import mne

# Script to read PMBR beamformer data, grand-average, and write a functional ROI

topPercent = 5  # percent of vertices to keep for each participant
aggregateThreshold = 0.5 # fraction of participants that must have a vertex active to include it in the functional ROI

# Find subjects to be analysed
homeDir = os.path.expanduser("~")
dataDir = homeDir + '/camcan/'
evokedStatsCSV = dataDir + 'proc_data/evoked_process_stats.csv'
ROIStatsCSV = dataDir + 'source_data/PMBR_stats.csv'
subjectData = pd.read_csv(evokedStatsCSV)

# Take only subjects with more than 60 epochs
subjectData2 = subjectData.copy()
subjectData2 = subjectData2.loc[subjectData2['numEpochs'] > 58]
numSubjects = len(subjectData2)
subjectIDs = subjectData2['SubjectID'].tolist()

# Paths
subjectsDir = dataDir + 'subjects/'
groupSourceDir = dataDir + 'source_data/TaskSensorAnalysis_transdef/fsaverage/'

# Make source path if it does not exist
if not os.path.exists(groupSourceDir):
    os.makedirs(groupSourceDir)

# Files to make
dsPrefix = 'transdef_transrest_mf2pt2_task_raw'
gAvgStcFile = groupSourceDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo_PMBR_DICS_gAvg'
funStcFile = groupSourceDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo_PMBR_DICS_top' + \
             str(topPercent) + 'percent'
funcLabelFile = groupSourceDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo_PMBR_DICS_funcLabel'

# Make a dataframe with info about each subject, and read in all morphed PMBR stcs to
#   a list
labels = ['EpochFileExists', 'TransFileExists', 'SurfSrcFileExists', 'BemExists',
          'PMBRstcExists', 'PMBRstcMorphExists']
epochFifs = []
transFifs = []
srcFifs = []
bemFifs = []
subjectDirs = []
stcFiles = []
stcMorphFiles = []
stcData = []
for subjectID in subjectIDs:

    # Define paths
    megDir = dataDir + 'proc_data/TaskSensorAnalysis_transdef/' + subjectID + '/'
    sourceDir = dataDir + 'source_data/TaskSensorAnalysis_transdef/' + subjectID + '/'

    # DICS Settings
    tmins = [0.5, -1.0]
    tstep = 0.75
    fmin = 15
    fmax = 30
    numFreqBins = 10  # linear spacing
    DICS_regularizaion = 0.5
    data_decimation = 5

    # Debug for existence of needed files
    inFilesOK = True

    # Files that exist
    epochFif = megDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo.fif'
    epochFifs.append(os.path.exists(epochFif))
    transFif = subjectsDir + 'coreg/sub-' + subjectID + '-trans.fif'
    transFifs.append(os.path.exists(transFif))
    srcFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5-src.fif'
    srcFifs.append(os.path.exists(srcFif))
    bemFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5120-bem-sol.fif'
    bemFifs.append(os.path.exists(bemFif))
    stcFile = sourceDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo_PMBR_DICS'
    stcMorphFile = sourceDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo_PMBR_DICS_fsaverage'

    try:
        thisSTC = mne.read_source_estimate(stcFile)
        stcFiles.append(True)
    except:
        stcFiles.append(False)
    try:
        thisSTC = mne.read_source_estimate(stcMorphFile)
        stcMorphFiles.append(True)
        stcData.append(thisSTC.data)
    except:
        stcMorphFiles.append(False)

# Make dataframe
allData = [('SubjectID', subjectIDs), ('EpochFileExists', epochFifs), ('TransFileExists', transFifs),
            ('SurfSrcFileExists', srcFifs), ('BemExists', bemFifs), ('PMBRstcExists', stcFiles),
            ('PMBRstcMorphExists', stcMorphFiles)]
df = pd.DataFrame.from_items(allData)

df.to_csv(ROIStatsCSV)

# Take only subjects with a morphed STC file
subjectData3 = df.copy()
subjectData3 = subjectData3.loc[subjectData3['PMBRstcMorphExists'] == True]
stcSubjectIDs = subjectData3['SubjectID'].tolist()

