#!/usr/bin/env python

# Import libraries
import os
import sys
import numpy as np
import pandas as pd
import mne
import multiprocessing as mp
import logging
from mne.time_frequency import tfr_morlet

# Script to calculate the TFR for each participant's MEG data

class StreamToLogger(object):
   """
   Fake file-like stream object that redirects writes to a logger instance.
   """
   def __init__(self, logger, log_level=logging.INFO):
      self.logger = logger
      self.log_level = log_level
      self.linebuf = ''

   def write(self, buf):
      for line in buf.rstrip().splitlines():
         self.logger.log(self.log_level, line.rstrip())


def MEG_preproc(subjectID):

    # Script to create processed data for Motor blocks
    homeDir = os.path.expanduser("~")
    dataDir = homeDir + '/camcan/'
    outDir = dataDir + 'proc_data/TaskSensorAnalysis_transdef/'
    dsPrefix = 'transdef_transrest_mf2pt2_task_raw'

    # Analysis parameters
    TFRfmin = 5
    TFRfmax = 90
    TFRfstep = 5

    ####################################################################
    # Files that exist
    subjectOutDir = outDir + str(subjectID) + '/'
    epochFif = subjectOutDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo.fif'

    # Files that get made
    tfrFile = subjectOutDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo_frange=' + str(TFRfmin) + '-' + str(
        TFRfmax) + 'Hz_fstep=' + str(TFRfstep) + 'Hz-tfr.h5'

    # Setup log file for standarda output and error
    logFile = subjectOutDir + dsPrefix + '_TFR_processing_notes.txt'
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(message)s',
        filename=logFile,
        filemode='a'
    )
    stdout_logger = logging.getLogger('STDOUT')
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl
    stderr_logger = logging.getLogger('STDERR')
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    logging.info(str(subjectID))

    # Epoch data based on buttone press
    epochs_clean = mne.read_epochs(epochFif)

    # Calculate TFR on MEG data and save results
    logging.info('Starting TFR Calculation on Magnetometers Only')
    magPicks = mne.pick_types(epochs_clean.info, meg='mag', eeg=False, eog=False, stim=False, exclude='bads')
    epochs_no_evoked = epochs_clean.copy().subtract_evoked()
    freqs = np.arange(TFRfmin, TFRfmax, TFRfstep)
    n_cycles = freqs / 2.0
    power, _ = tfr_morlet(epochs_no_evoked, freqs=freqs, n_cycles=n_cycles, picks=magPicks,
                          use_fft=False, return_itc=True, decim=3, n_jobs=1)
    power.save(tfrFile, overwrite=True)

    return 0

if __name__ == '__main__':

    # Check number of subjects to be worked with
    homeDir = os.path.expanduser("~")
    csvFile = homeDir + '/camcan/proc_data/evoked_process_stats.csv'
    subjectsDF = pd.read_csv(csvFile)
    subjectsDF = subjectsDF.loc[subjectsDF['numEpochs'] > 60]
    numSubjects = len(subjectsDF)
    subjectIDs = subjectsDF['SubjectID'].tolist()
    print(subjectIDs)

    #Set up the parallel task pool to use all available processors
    count = int(np.round(mp.cpu_count()*1/2))
    pool = mp.Pool(processes=count)

    #Run the jobs
    pool.map(MEG_preproc, subjectIDs)

