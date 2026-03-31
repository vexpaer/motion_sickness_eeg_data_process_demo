
import numpy as np
import os
from readbdfdata import readbdfdata
from tkinter import filedialog
from tkinter import *

def check_files_format(path):
    filename = []
    pathname = []
    if len(path) == 0:
        raise TypeError('please select valid file')

    elif len(path) == 1:
        (temppathname, tempfilename) = os.path.split(path[0])
        if 'edf' in tempfilename:
            filename.append(tempfilename)
            pathname.append(temppathname)
            return filename, pathname
        elif 'bdf' in tempfilename:
            raise TypeError('unsupport only one neuracle-bdf file')
        else:
            raise TypeError('not support such file format')

    else:
        temp = []
        temppathname = r''
        evtfile = []
        idx = np.zeros((len(path) - 1,))
        for i, ele in enumerate(path):
            (temppathname, tempfilename) = os.path.split(ele)
            if 'data' in tempfilename:
                temp.append(tempfilename)
                if len(tempfilename.split('.')) > 2:
                    try:
                        idx[i] = (int(tempfilename.split('.')[1]))
                    except:
                        raise TypeError('no such kind file')
                else:
                    idx[i] = 0
            elif 'evt' in tempfilename:
                evtfile.append(tempfilename)

        pathname.append(temppathname)
        datafile = [temp[i] for i in np.argsort(idx)]

        if len(evtfile) == 0:
            raise TypeError('not found evt.bdf file')

        if len(datafile) == 0:
            raise TypeError('not found data11.bdf file')
        elif len(datafile) > 1:
            print('current readbdfdata() only support continue one data11.bdf ')
            return filename, pathname
        else:
            filename.append(datafile[0])
            filename.append(evtfile[0])
            return filename, pathname

def Read_bdf():
    """
    读取博瑞康数据，同时选择两个bdf文件
    Returns:
        data:字典类型，存储数据信息（包含原始数据、打标信息、采样率、通道名称、通道数）

        示例：
            data = data['data']
            events = data['events']
            fs = data['srate']
            ch_names = data['ch_names']
            nchan = data['nchan']
    """
    root = Tk()
    root.withdraw()
    ## select bdf or edf file
    # path= QFileDialog.getOpenFileNames(None,
    #                                       "选取文件",
    #                                       "./",
    #                                       "BDF Files (*.bdf);;EDF Files (*.edf)")  # 设置文件扩展名过滤,注意用双分号间隔

    try:
        path = filedialog.askopenfilenames(initialdir='/', title='Select two bdf files',
                                       filetypes=(("two bdf files", "*.bdf"), ("one edf files", "*.edf")))
        print(path)
    ## check files format
        filename, pathname = check_files_format(path)
    ## parse data
        data = readbdfdata(filename, pathname)
        print(data)
        return data
    except Exception as e:
        pass

def Read_bdf_by_file():
    """
    读取博瑞康数据，同时选择两个bdf文件
    Returns:
        data:字典类型，存储数据信息（包含原始数据、打标信息、采样率、通道名称、通道数）

        示例：
            data = data['data']
            events = data['events']
            fs = data['srate']
            ch_names = data['ch_names']
            nchan = data['nchan']
    """
    root = Tk()
    root.withdraw()
    ## select bdf or edf file

    try:
        path = filedialog.askopenfilenames(initialdir='/', title='Select bdf file',
                                       filetypes=(("two bdf files", "*.bdf"), ("one edf files", "*.edf")))
        # print(path)
    ## check files format
        filename, pathname = check_files_format(path)
    ## parse data
        data = readbdfdata(filename, pathname)
        return data, path[0]
    except Exception as e:
        return None

def read_bdf_by_path(path):
    """
    
    :param path: ('J:/数据/博瑞康/无负重/0001王冲/肌电数据/右上/20210818 16.37/data11.bdf', 'J:/数据/博瑞康/无负重/0001王冲/肌电数据/右上/20210818 16.37/evt.bdf')
    :return: 
    """
## check files format

    filename, pathname = check_files_format(path)
    print(filename)
## parse data
    eeg = readbdfdata(filename, pathname)
    print(eeg)
    return eeg

