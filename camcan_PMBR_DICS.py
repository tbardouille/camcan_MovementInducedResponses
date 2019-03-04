#!/usr/bin/env python

# Import libraries
import os, sys
import numpy as np
import pandas as pd
import mne
import logging
import multiprocessing as mp
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd

# Script to read calculate the DICS beamformer for the induced response of interest

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


def PMBR_DICS(subjectID):

    # Define paths
    homeDir = os.path.expanduser("~")
    dataDir = homeDir + '/camcan/'
    megDir = dataDir + 'proc_data/TaskSensorAnalysis_transdef/' + subjectID + '/'
    outDir = dataDir + 'source_data/TaskSensorAnalysis_transdef/' + subjectID + '/'
    dsPrefix = 'transdef_transrest_mf2pt2_task_raw'
    subjectsDir = dataDir + 'subjects/'

    # Make source path if it does not exist
    if not os.path.exists(outDir):
        os.makedirs(outDir)

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

    # Files that exits
    epochFif = megDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo.fif'
    if not os.path.exists(epochFif):
        inFilesOK = False
    transFif = subjectsDir + 'coreg/sub-' + subjectID + '-trans.fif'
    if not os.path.exists(transFif):
        inFilesOK = False
    srcFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5-src.fif'
    if not os.path.exists(srcFif):
        inFilesOK = False
    bemFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5120-bem-sol.fif'
    if not os.path.exists(bemFif):
        inFilesOK = False

    # Files to make
    stcFile = outDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo_PMBR_DICS'
    stcMorphFile = outDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo_PMBR_DICS_fsaverage'

    # Setup log file for standarda output and error
    logFile = outDir + dsPrefix + '_PMBR_DICS_processing_notes.txt'
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(message)s',
        filename=logFile,
        filemode='w'
    )
    stdout_logger = logging.getLogger('STDOUT')
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl
    stderr_logger = logging.getLogger('STDERR')
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    if inFilesOK:

        # Read epochs
        epochs = mne.read_epochs(epochFif)
        # Read source space
        src = mne.read_source_spaces(srcFif)
        # Make forward solution
        forward = mne.make_forward_solution(epochs.info,
                                            trans=transFif, src=src, bem=bemFif,
                                            meg=True, eeg=False)

        # DICS Source Power example
        # https://martinos.org/mne/stable/auto_examples/inverse/plot_dics_source_power.html#sphx-glr-auto-examples-inverse-plot-dics-source-power-py

        # Compute DICS spatial filter and estimate source power.
        stcs = []
        for tmin in tmins:
            csd = csd_morlet(epochs, tmin=tmin, tmax=tmin + tstep, decim=data_decimation,
                             frequencies=np.linspace(fmin, fmax, numFreqBins))
            filters = make_dics(epochs.info, forward, csd, reg=DICS_regularizaion)
            stc, freqs = apply_dics_csd(csd, filters)
            stcs.append(stc)

        # Take difference between active and baseline, and mean across frequencies
        ERS = np.log2(stcs[0].data / stcs[1].data)
        a = stcs[0]
        ERSstc = mne.SourceEstimate(ERS, vertices=a.vertices, tmin=a.tmin, tstep=a.tstep, subject=a.subject)
        ERSband = ERSstc.mean()
        ERSband.save(stcFile)

        ERSmorph = ERSband.morph(subject_to='fsaverage', subject_from='sub-' + subjectID, subjects_dir=subjectsDir)
        ERSmorph.save(stcMorphFile)

    else:

        print('Some or all input files missing')



if __name__ == '__main__':

    # Find subjects to be analysed
    homeDir = os.path.expanduser("~")
    dataDir = homeDir + '/camcan/'
    evokedStatsCSV = dataDir + 'proc_data/evoked_process_stats.csv'
    subjectData = pd.read_csv(evokedStatsCSV)

    # Take only subjects with more than 60 epochs
    subjectData2 = subjectData.copy()
    subjectData2 = subjectData2.loc[subjectData2['numEpochs'] > 60]
    numSubjects = len(subjectData2)
    subjectIDs = subjectData2['SubjectID'].tolist()

    # Set up the parallel task pool to use all available processors
    count = int(np.round(mp.cpu_count()*3/4))
    pool = mp.Pool(processes=count)

    # Run the jobs
    pool.map(PMBR_DICS, subjectIDs)


