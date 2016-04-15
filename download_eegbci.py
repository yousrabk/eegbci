from mne.datasets import eegbci

runs = range(1, 15)
for subject in range(1, 110):
    raw_fnames = eegbci.load_data(subject, runs)
