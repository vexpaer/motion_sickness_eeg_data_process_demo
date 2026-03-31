import os
import numpy as np
import mne
import matplotlib.pyplot as plt

# Purpose:
# EEG preprocessing demo pipeline for a single subject folder (25009) in the current directory.
# Steps include montage setup, bad-channel interpolation, re-reference, filtering, ICA, and EDF export.

# Set electrode montage
def mont(raw):
    mymontage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(mymontage, on_missing='warn')
    return raw

# Interpolate bad channels
def interpolate(raw):
    raw.plot(start=20, duration=1, n_channels=33, block=True, title='请检查并选中坏导')
    plt.show()
    if raw.info['bads']:
        print('已选择坏导: ', raw.info['bads'], '开始进行插值')
        raw.load_data()
        raw.interpolate_bads()
        raw.plot(start=20, duration=1, n_channels=33, block=True, title='坏导插值完成，如无误请关闭窗口')
        plt.show()
    else:
        print('无坏导，跳过插值')
    return raw


def refe(raw):
    # 加载数据
    raw.load_data()
    # 设置EEG参考电极，修正语法错误
    raw.set_eeg_reference(ref_channels=['A1', 'A2'])
    # 绘制原始数据的时间片段以查看效果
    raw.plot(start=20, duration=1, block=True, title='重参考完成，无误请关闭窗口')
    # 展示图像
    plt.show()
    # 返回修改后的数据
    return raw


# Filter settings
def filt(raw, filter_type='fir', notch_width=None, notch_trans_bandwidth=1.0,
         l_trans_bandwidth='auto', h_trans_bandwidth='auto'):
    raw_filter = raw.copy()
    raw_filter.filter(
        l_freq=1, h_freq=40,
        method=filter_type,
        l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth
    )
    raw_filter.notch_filter(
        freqs=50,
        method=filter_type,
        notch_widths=notch_width,
        trans_bandwidth=notch_trans_bandwidth
    )
    # raw_filter.plot_psd(fmax=60)
    # plt.show(block=False)
    raw_filter.plot(start=20, duration=1, block=True, title=f'滤波完成（{filter_type.upper()}),准备ICA')
    return raw_filter

# ICA artifact removal
def run_ICA(raw, method='picard'):
    ica = mne.preprocessing.ICA(n_components=10, method=method, max_iter=800, random_state=97)
    ica.fit(raw)
    ica.plot_components()
    ica.plot_sources(raw, show_scrollbars=False, title=f'{method} ICA 成分选择')
    plt.show(block=True)
    raw_recons = ica.apply(raw.copy())
    raw.plot(start=20, duration=1, n_channels=33, title=f'{method} ICA处理前')
    raw_recons.plot(start=20, duration=1, n_channels=33, title=f'{method} ICA处理后')
    plt.show()
    return raw_recons

# Baseline correction
def baseline_correction(raw, num=500):
    data = raw.get_data()[:, 760000:880000]  # 120000点 ≈ 120s
    info = raw.info
    for i in range(num):
        mean = np.mean(data[:, :num], axis=1)
        data[:, i] -= mean
    for i in range(data.shape[1] - num):
        mean = np.mean(data[:, i:num + i], axis=1)
        data[:, num + i] -= mean
    raw_crop = mne.io.RawArray(data, info)
    return raw_crop

# Example main pipeline
def preprocess_pipeline(adres):
    # edf_path = f"C:\\Users\\vexpaer\\Desktop\\old eeg data\\EEG\\易感\\正常前静态脑电\\25014\data.bdf"
    # edf_path = f"C:\\Users\\vexpaer\\Desktop\\old eeg data\\EEG\\易感\\正常前静态脑电\\25009\\data_crop.edf"
    #edf_path = f"C:\\Users\\黄仕源\\Desktop\\old eeg data\\{i}.edf"
    edf_path = adres
    if not os.path.exists(edf_path):
        print(f"文件不存在: {edf_path}")

    raw = mne.io.read_raw_bdf(edf_path, preload=True)
    raw.plot()
    plt.show()
    raw = mont(raw)
    raw = interpolate(raw)
    raw = refe(raw)
    raw = filt(raw)  # 默认 FIR
    raw = run_ICA(raw, method='picard')  # 仅执行一次 ICA

    # Save the preprocessed file
    dir_path = os.path.dirname(edf_path)  # 获取目录路径
    output_path = os.path.join(dir_path, "data_clean.edf")  # 组合新文件名
    
    raw.export(output_path, fmt='edf', overwrite=True)
    print(f" 预处理完成并保存到: {output_path}")

if __name__ == "__main__":
    # Example path: current directory /25009/data.bdf
    adress = os.path.join(os.getcwd(), "25009", "data.bdf")
    preprocess_pipeline(adress)

