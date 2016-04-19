
import numpy as np
import csv
import pandas as pd

from mne import Epochs, pick_types, find_events
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

# #############################################################################
# # Set parameters and read data

# tmin, tmax = -1., 4.
# event_id = dict(hands=2, feet=3)
n_subjs = 110
runs = [6, 10, 14]  # motor imagery: hands vs feet

for subject in range(1, n_subjs + 1):
    if subject == 88 or subject == 89 or subject == 92 or subject == 100 \
        or subject == 110:
        continue
    raw_fnames = eegbci.load_data(subject, runs)
    raw_files = [read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raw_files)
    index = np.round(raw.times * 1000.).astype(int)

    # Concatenate subjects as series. We obtain at the end 7 subjects
    # each one having 15 series. We learn on 12 subjects and try to 
    # generalize on the 3 left. Do this 7 times.
    if subject >= 90 and subject < 92:
        subject -= 2
    elif subject >=92 and subject < 100:
        subject -= 3
    elif subject >= 100 and subject < 110:
        subject -= 4
    new_subj, serie = (subject - 1) // 15 + 1, (subject - 1) % 15 + 1
    subj_name = 'subj%s_series%s_' % (new_subj, serie)

    # Events from STI 
    ev = raw.copy().pick_channels(['STI 014'])[0][0][0]
    # Keep only eeg from the raw data
    raw.pick_types(eeg=True)

    #####################
    # Make data csv files 
    df_data = raw.to_data_frame()
    df_data.index = [subj_name + '%s' %ind for ind in index]
    df_data.keys().insert(0, 'id')

    #####################
    # Make events csv files: we keep only events 2 and 3 for hands/feet
    ev[ev == 1] = 0.
    events = np.zeros((len(ev), 2))
    ev1_indices = np.where(ev == 2)[0]
    ev2_indices = np.where(ev == 3)[0]
    events[ev1_indices, 0] = 1.
    events[ev2_indices, 1] = 1.

    d = {'hands': pd.Series(events[:, 0], index=index),
         'feet':  pd.Series(events[:, 1], index=index)}

    d['hands'].index = [subj_name + '%s' %ind for ind in index]
    d['feet'].index = [subj_name + '%s' %ind for ind in index]
    df_events = pd.DataFrame(d)

    tmp_data, tmp_events = 'tmp_data.csv', 'tmp_events.csv'
    df_data.to_csv(tmp_data)
    df_events.to_csv(tmp_events)

    # Save the data/events
    if serie <= 12:
        data_name = 'data/train/' + subj_name + 'data.csv'
        events_name = 'data/train/' + subj_name + 'events.csv'
        with open(tmp_events, 'rb') as csvfile, open(events_name, 'wb') as outfile:
            r = csv.reader(csvfile, delimiter=',', quotechar='"')
            headers=r.next()
            headers[0] = 'id'

            w = csv.writer(outfile, delimiter=',', quotechar='"')
            w.writerow(headers)
            for line in r:
                w.writerow(line)
        csvfile.close()
        outfile.close()
    else:
        data_name = 'data/test/' + subj_name + 'data.csv'

    with open(tmp_data, 'rb') as csvfile, open(data_name, 'wb') as outfile:
        r = csv.reader(csvfile, delimiter=',', quotechar='"')
        headers=r.next()
        headers[0] = 'id'

        w = csv.writer(outfile, delimiter=',', quotechar='"')
        w.writerow(headers)
        for line in r:
            w.writerow(line)
    csvfile.close()
    outfile.close()
