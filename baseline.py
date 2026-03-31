import mne
import os
import glob
import numpy as np

# Purpose:
# Baseline correction demo script using the baseline file from ./25009/crop.edf,
# then applying channel-wise mean subtraction to crop files in the same folder.

def baseline_cor(adress1, adress2):
    """
    Baseline correction function.

    Parameters:
    adress1 -- Folder path containing the baseline file (crop.edf)
    adress2 -- Folder path containing target EDF files to process
    """
    # 1. Read and process the baseline file
    baseline_file = os.path.join(adress1, 'crop.edf')
    if not os.path.exists(baseline_file):
        raise FileNotFoundError(f"基线文件未找到: {baseline_file}")
    
    # Read baseline EDF file
    raw_baseline = mne.io.read_raw_edf(baseline_file, preload=True)
    
    # Compute channel-wise means (baseline reference)
    baseline_means = np.mean(raw_baseline.get_data(), axis=1)
    
    # Get baseline channel names
    baseline_channels = raw_baseline.ch_names
    
    # 2. Process files in the target folder
    crop_files = glob.glob(os.path.join(adress2, '*crop*.edf'))
    
    if not crop_files:
        print(f"在 {adress2} 中未找到包含'crop'的EDF文件")
        return
    
    for file_path in crop_files:
        # Read target file
        raw_target = mne.io.read_raw_edf(file_path, preload=True)
        
        # Ensure channel order matches baseline channels
        target_data = raw_target.get_data()
        ordered_data = np.zeros_like(target_data)
        
        for i, ch_name in enumerate(baseline_channels):
            if ch_name in raw_target.ch_names:
                idx = raw_target.ch_names.index(ch_name)
                # Baseline correction: subtract corresponding channel baseline mean
                ordered_data[i] = target_data[idx] - baseline_means[i]
            else:
                print(f"警告: 通道 {ch_name} 在文件 {os.path.basename(file_path)} 中不存在")
                ordered_data[i] = target_data[0]  # fallback to first channel
        
        # Create corrected Raw object
        info = mne.create_info(
            ch_names=baseline_channels,
            sfreq=raw_target.info['sfreq'],
            ch_types='eeg'  # assume all channels are EEG
        )
        raw_corrected = mne.io.RawArray(ordered_data, info)
        
        # Keep original metadata where possible
        raw_corrected.set_meas_date(raw_target.info['meas_date'])
        raw_corrected.set_montage(raw_target.get_montage())
        
        # 3. Save corrected file
        new_filename = os.path.basename(file_path).replace('crop', 'final')
        save_path = os.path.join(adress2, new_filename)
        
        raw_corrected.export(save_path,fmt='edf', overwrite=True)
        print(f"已保存校准后的文件: {save_path}")

# Example usage
if __name__ == "__main__":
    # Example path: current directory /25009
    subject_dir = os.path.join(os.getcwd(), "25009")
    baseline_cor(subject_dir, subject_dir)

