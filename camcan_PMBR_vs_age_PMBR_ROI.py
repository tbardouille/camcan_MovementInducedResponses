#!/usr/bin/env python

# Import libraries
import os
import numpy as np
import pandas as pd
import copy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import seaborn as sbn
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Script to find the characteristics of the induced response, and write interesting findings to a panda frame
#   with demographic information

homeDir = os.path.expanduser("~")
dataDir = homeDir + '/camcan/'
megDir = dataDir + 'proc_data/TaskSensorAnalysis_transdef/'
dsPrefix = 'transdef_transrest_mf2pt2_task_raw'

# Constants
TFRfmin = 10
TFRfmax = 35
TFRfstep = 1
TFRFirstSample = 1500 + 500
TFRLastSample = 1500 + 1250
plotOK = False
figureDir = os.path.join(dataDir, 'figures', 'sourceTFR', 'PMBR_ROI')
if not os.path.exists(figureDir):
    os.makedirs(figureDir)

# Files to read
evokedStatsCSV = dataDir + 'source_data/PMBR_stats.csv'
demographicCSV = dataDir + 'proc_data/demographics_allSubjects.csv'

# Files to make
gAvgTFRFile = megDir + '/gAvg/' + dsPrefix \
        + '_buttonPress_duration=3.4s_cleaned-epo_frange=' \
        + str(TFRfmin) + '-' + str(TFRfmax) + 'Hz_fstep=' + str(TFRfstep) + 'Hz-tfr.h5'
tfrStatsCSV = dataDir + 'proc_data/PMBR_process_stats.csv'


####################################################################
# Read information about analysis of evoked field
subjectData = pd.read_csv(evokedStatsCSV)

# Take only subjects with morphed STC from PMBR DICS process
subjectData2 = subjectData.copy()
subjectData2 = subjectData2.loc[subjectData2['PMBRstcMorphExists']]
numSubjects = len(subjectData2)
subjectIDs = subjectData2['SubjectID'].tolist()

freqs = np.arange(TFRfmin, TFRfmax + TFRfstep, TFRfstep)
times = np.arange(1500)/1000

# Load TFR maps
tfrData = []
tfrSubjects = []
for subjectID in subjectIDs:

    outDir = dataDir + 'source_data/TaskSensorAnalysis_transdef/' + subjectID + '/'

    # Files containing TFR data
    tfrFile1 = outDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo' + \
               '_PMBR_CofM_noEvoked_TFR_' + str(TFRfmin) + '-' + str(TFRfmax) + 'Hz.npy'

    if os.path.exists(tfrFile1):
        leftMI = np.load(tfrFile1)
        tfrData.append(leftMI)
        tfrSubjects.append(subjectID)

# Change list to array
tfrData_np = np.array(tfrData)

numSubjects = len(tfrSubjects)

# For each subject, find the strongest local peak in the TFR
betaPeakFrequency = []
betaPeakAmplitude = []
betaPeakTime = []
for pdsIndex in np.arange(numSubjects):

    # Isolate ERS 3D matrix
    thisTFR = copy.copy(tfrData_np[pdsIndex])
    dat = thisTFR[:,TFRFirstSample:TFRLastSample]

    # Use maximum_filter to find local peak (if any)
    footprintFreq = 10
    footprintTime = 80
    threshold = 0.05

    neighbourhood_size = (footprintFreq,footprintTime)
    data_max = filters.maximum_filter(dat, size=neighbourhood_size)
    maxima = (dat == data_max)
    data_min = filters.minimum_filter(dat, size=neighbourhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(int(x_center))
        y_center = (dy.start + dy.stop - 1)/2
        y.append(int(y_center))

    # Order local peaks based on magnitude, and take the largest as the PMBR
    peakOrder = np.argsort(dat[y,x])
    globalPeak = (x[peakOrder[-1]],y[peakOrder[-1]])

    # Log the PMBR frequency, amplitude and latency
    betaPeakFrequency.append (freqs[globalPeak[1]])
    betaPeakAmplitude.append (dat[[globalPeak[1]],[globalPeak[0]]][0])
    betaPeakTime.append (times[globalPeak[0]])


subjectDataList = [('SubjectID', tfrSubjects),
                   ('PMBRPeakFrequency', betaPeakFrequency),
                   ('PMBRPeakAmplitude', betaPeakAmplitude),
                   ('PMBRPeakTime', betaPeakTime)]
subjectData3 = pd.DataFrame.from_items(subjectDataList)

# Write out the pandas dataframe
subjectData3.to_csv(tfrStatsCSV)

# Read demographic data
demographicsData = pd.read_csv(demographicCSV)

# Attach both dataframes
allData = pd.merge(demographicsData, subjectData3, on='SubjectID')



