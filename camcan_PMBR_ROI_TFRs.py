#!/usr/bin/env python

# Import libraries
import os, sys
import numpy as np
import pandas as pd
import mne
import logging
import multiprocessing as mp

# Script to make TFR data for a functional region of interest


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


def PMBR_ROI_TFR(subjectID):

    # Settings
    fmin = 10
    fmax = 35
    fstep = 1
    #fmin = 5
    #fmax = 90
    #fstep = 5
    LCMV_regularization = 0.5

    # Define paths
    homeDir = os.path.expanduser("~")
    dataDir = homeDir + '/camcan/'
    megDir = dataDir + 'proc_data/TaskSensorAnalysis_transdef/' + subjectID + '/'
    outDir = dataDir + 'source_data/TaskSensorAnalysis_transdef/' + subjectID + '/'
    dsPrefix = 'transdef_transrest_mf2pt2_task_raw'
    subjectsDir = dataDir + 'subjects/'
    groupSourceDir = dataDir + 'source_data/TaskSensorAnalysis_transdef/fsaverage/'

    # Make source path if it does not exist
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    # Files that exits
    epochFif = megDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo.fif'
    transFif = subjectsDir + 'coreg/sub-' + subjectID + '-trans.fif'
    srcFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5-src.fif'
    bemFif = subjectsDir + 'sub-' + subjectID + '/bem/sub-' + subjectID + '-5120-bem-sol.fif'
    funcLabelFile = groupSourceDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo_PMBR_DICS_funcLabel-lh.label'

    # Files to make
    tfrFile1 = outDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo_PMBR_ROI_TFR_' + str(fmin) + '-' + str(fmax) + 'Hz'
    tfrFile2 = outDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo_PMBR_CofM_TFR_' + str(fmin) + '-' + str(fmax) + 'Hz'
    tfrFile3 = outDir + dsPrefix + '_buttonPress_duration=3.4s_cleaned-epo_PMBR_CofM_noEvoked_TFR_' + str(fmin) + '-' + str(fmax) + 'Hz'


    # Setup log file for standarda output and error
    logFile = outDir + dsPrefix + '_PMBR_DICS_ROI_TFR_processing_notes.txt'
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


    if not os.path.exists(tfrFile3):
        # Read epochs
        epochs = mne.read_epochs(epochFif)
        # Read source space
        src = mne.read_source_spaces(srcFif)
        # Make forward solution
        forward = mne.make_forward_solution(epochs.info,
                                            trans=transFif, src=src, bem=bemFif,
                                            meg=True, eeg=False)

        # Read functional ROI label, morph to subject's MRI and take centre or mass for source estimation
        label = mne.read_label(funcLabelFile)
        label.morph(subject_from='fsaverage', subject_to='sub-' + subjectID, subjects_dir=subjectsDir)

        # Compute LCMV time-frequency response at ROI
        noise_cov = mne.compute_covariance(epochs, tmin=-1.7, tmax=-0.2, method='shrunk')
        data_cov = mne.compute_covariance(epochs, tmin=0.0, tmax=1.5, method='shrunk')
        filters = mne.beamformer.make_lcmv(epochs.info, forward, data_cov, reg=LCMV_regularization,
                                           noise_cov=noise_cov, pick_ori='max-power', weight_norm='unit-noise-gain',
                                           label=label)
        stc = mne.beamformer.apply_lcmv_epochs(epochs, filters, max_ori_out='signed')

        # Make a label based on the LCMV, then pull the center of mass
        stcLabel = mne.Label(stc[0].vertices[0], hemi='lh', subject='sub-' + subjectID)
        a = stcLabel.center_of_mass(subject='sub-' + subjectID, subjects_dir=subjectsDir, restrict_vertices=True)
        comLabel = mne.Label([a], hemi='lh', subject='sub-' + subjectID)

        # Calculate LCMV beamformer at centre of mass
        filters2 = mne.beamformer.make_lcmv(epochs.info, forward, data_cov, reg=LCMV_regularization,
                                            noise_cov=noise_cov, pick_ori='max-power', weight_norm='unit-noise-gain',
                                            label=comLabel)
        stc2 = mne.beamformer.apply_lcmv_epochs(epochs, filters2, max_ori_out='signed')

        # TFR Analysis Starts Here
        epochTimes = epochs.copy()
        times = epochTimes.crop(tmin=-1.5, tmax=1.5).times
        sfreq = epochs.info['sfreq']
        freqs = np.arange(fmin, fmax+fstep, fstep)
        n_cycles = freqs / 3.  # different number of cycle per frequency

        # Now put ROI beamformer data into an array and make the TFR
        roiData = []
        for thisSTC in stc:
            roiData.append(thisSTC.data)
        roiData = np.asarray(roiData)

        # And calculate TFR
        a = mne.time_frequency.tfr_array_morlet(roiData, sfreq, freqs, n_cycles=n_cycles,
                                                zero_mean=False, use_fft=True, decim=1, output='complex',
                                                n_jobs=1, verbose=True)
        b = np.mean(np.squeeze(np.mean(np.abs(a), axis=0)), axis=0)
        c = b[:, 200:-200]
        roiTFR = mne.baseline.rescale(c, times, (-1.5, -1.0), mode='logratio', copy=True)

        # Save result as npy file
        np.save(tfrFile1, roiTFR)

        # Now put centre of mass beamformer data into an array and make the TFR
        comData = []
        for thisSTC in stc2:
            comData.append(thisSTC.data)
        comData = np.asarray(comData)

        # And calculate TFR
        a = mne.time_frequency.tfr_array_morlet(comData, sfreq, freqs, n_cycles=n_cycles,
                                                zero_mean=False, use_fft=True, decim=1, output='complex',
                                                n_jobs=1, verbose=True)
        b = np.squeeze(np.mean(np.abs(a), axis=0))
        c = b[:, 200:-200]
        comTFR = mne.baseline.rescale(c, times, (-1.5, -1.0), mode='logratio', copy=True)

        # Save results as npy file
        np.save(tfrFile2, comTFR)

        # And calculate TFR with evoked removed
        numEpochs = roiData.shape[0]
        comEvoked = np.mean(comData, axis=0)
        comData_noEvoked = comData - np.tile(comEvoked, [numEpochs, 1, 1])

        a = mne.time_frequency.tfr_array_morlet(comData_noEvoked, sfreq, freqs, n_cycles=7.0,
                                                zero_mean=False, use_fft=True, decim=1, output='complex',
                                                n_jobs=1, verbose=True)
        b = np.squeeze(np.mean(np.abs(a), axis=0))
        c = b[:, 200:-200]
        comTFR_noEvoked = mne.baseline.rescale(c, times, (-1.5, -1.0), mode='logratio', copy=True)
        np.save(tfrFile3, comTFR_noEvoked)



if __name__ == '__main__':

    # Find subjects to be analysed
    homeDir = os.path.expanduser("~")
    dataDir = homeDir + '/camcan/'
    evokedStatsCSV = dataDir + 'source_data/PMBR_stats.csv'
    subjectData = pd.read_csv(evokedStatsCSV)

    # Take only subjects with more than 60 epochs
    subjectData2 = subjectData.copy()
    subjectData2 = subjectData2.loc[subjectData2['PMBRstcMorphExists']]
    numSubjects = len(subjectData2)
    subjectIDs = subjectData2['SubjectID'].tolist()
    print(subjectIDs)

    # Set up the parallel task pool to use all available processors
    count = int(np.round(mp.cpu_count()*1/3))
    print(count)
    pool = mp.Pool(processes=count)

    # Run the jobs
    pool.map(PMBR_ROI_TFR, subjectIDs)


