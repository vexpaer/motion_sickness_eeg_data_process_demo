# Purpose:
# Crop six lap-based static segments from data.bdf using evt.bdf markers,
# then export each segment as middle1.edf ... middle6.edf in the same folder.
import os
import mne
import numpy as np
from File_io import read_bdf_by_path

# Crop static segments

def crop(save_p) :

    data_p = os.path.join(save_p,f"data.bdf")
    event_p = os.path.join(save_p,f"evt.bdf")

    eeg_info = read_bdf_by_path((data_p, event_p))
    data = eeg_info['data']  # signal data
    events = eeg_info['events']  # event markers

    event_list = events[:, 0]

    # 6 static laps; each lap uses 9 events. Segment range is from the first to the last event of each lap.
    for i in [1,2,3,4,5,6]:
        data_crop = data[:, event_list[9*i-9]:event_list[9*i-1]]

        raw = mne.io.read_raw_bdf(data_p, preload=True)

        info = raw.info

        raw_crop = mne.io.RawArray(data_crop, info)
        
        output_path = os.path.join(save_p, f'middle{i}.edf')
        raw_crop.export(output_path, fmt='edf', overwrite=True)


if __name__ == "__main__":
    # Example path: current directory /25009
    adress = os.path.join(os.getcwd(), "25009")
    crop(adress)

