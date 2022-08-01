import numpy as np
import os
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
import json
from six import string_types

import sys
import os
import time
import csv
from datetime import datetime

def extract_timestamps(filename):
    file = open(filename, 'r')
    data = file.readlines()

    ts = []
    for i in data:
        time_obj = datetime.strptime(i.rstrip(), '%Y-%m-%d %H:%M:%S.%f')
        ts.append(time_obj.timestamp())
    
    return np.array(ts)

def extract_data(input_filepath, vital_sign):
    file = open(input_filepath, 'r', encoding='utf-8-sig')
    csvreader = csv.reader(file)
    stamp_list = []
    idx = 3
    
    path = os.path.dirname(input_filepath)
    filename_ext = os.path.basename(input_filepath)
    filename = os.path.splitext(filename_ext)[0]

    if(filename == "MPDataExport"):
        # This skips the first row of the CSV file.
        next(csvreader)
        if(vital_sign == "hr_ppg"):
            idx = 3
        elif(vital_sign == "resp_rate"):
            idx = 4
        elif(vital_sign == "spO2"):
            idx = 13
        elif(vital_sign == "hr_ppg"):
            idx = 14
    
    for i in csvreader:
        stamp_list.append(i)

    mx_stamps = []
    sys_stamps = []
    data = []


    for j in range(len(stamp_list)):
        time_obj_mx = datetime.strptime(stamp_list[j][0], '%d-%m-%Y %H:%M:%S.%f')
        stamp_mx = time_obj_mx.timestamp()
        mx_stamps.append(stamp_mx)

        time_obj_sys = datetime.strptime(stamp_list[j][2], '%d-%m-%Y %H:%M:%S.%f')
        stamp_sys = time_obj_sys.timestamp()
        sys_stamps.append(stamp_sys)

        data.append(int(stamp_list[j][idx]))

    return mx_stamps, sys_stamps, data
        
def find_deltas(sys_stamps, mx_stamps):
    const_num = sys_stamps[0]
    deltas = []
    for i, stamp in enumerate(sys_stamps):
        if(stamp != const_num):
            const_num = stamp
            delta = sys_stamps[i-1] - mx_stamps[i-1]
            if(sys_stamps[i] - sys_stamps[i-1] > 0.3): # comparison for blips
                deltas.append(delta)
    return np.array(deltas)

def find_deltas2(sys_stamps, mx_stamps):
    sys_stamps = sys_stamps[0:200*32]
    mx_stamps  = mx_stamps[0:200*32]

    const_num = sys_stamps[0]
    deltas = []
    for i, stamp in enumerate(sys_stamps):
        if(stamp != const_num):
            const_num = stamp
            delta = sys_stamps[i-1] - mx_stamps[i-1]
            if(sys_stamps[i] - sys_stamps[i-1] > 0.3): # comparison for blips
                deltas.append(delta)
    return np.array(deltas)

def unroll_stamps(mx_stamps, batch_size = int(32), time_diff = 0.256):

    unrolled_stamps = []

    for i in range(int(len(mx_stamps)/batch_size)):
        current_stamp = mx_stamps[i * batch_size]
        # print(current_stamp)
        for j in range(batch_size):
            unrolled_val = current_stamp - time_diff + time_diff*(j+1)/batch_size
            # print(unrolled_val)
            unrolled_stamps.append(unrolled_val)
    
    return np.array(unrolled_stamps)

def unroll_stamps2(mx_stamps, batch_size = int(32), time_diff = 0.256):

    unrolled_stamps = []

    current_stamp = mx_stamps[0] - time_diff
    for i in range(int(len(mx_stamps)/batch_size)):
        current_stamp += time_diff
        # print(current_stamp)
        for j in range(batch_size):
            unrolled_val = current_stamp - time_diff + time_diff*(j+1)/batch_size
            # print(unrolled_val)
            unrolled_stamps.append(unrolled_val)
    
    return np.array(unrolled_stamps)

def apply_delta(mx_stamps, sys_mx_time_delta):
    return mx_stamps + sys_mx_time_delta

def timestamp_process(ts):
        f = ((float(ts)/1e6)-int(float(ts)/1e6))*1e6

        ts = int(float(ts)/1e6)
        s = ((float(ts)/1e2)-int(float(ts)/1e2))*1e2
        ts = int(float(ts)/1e2)
        m = ((float(ts)/1e2)-int(float(ts)/1e2))*1e2
        ts = int(float(ts)/1e2)
        h = ((float(ts)/1e2)-int(float(ts)/1e2))*1e2


        temp = (3600*h)+(60*m)+s+(f*1e-6)
        temp = float(int(temp*1e6))/1e6

        return temp

def interpolate_timestamp(sensor, vital_sign, path):
    if (sensor == "rgbd" or sensor == "rgb" or sensor == "nir" or sensor == "polarized" or sensor == "thermal" or sensor == "uv" ): # or "audio"
        filepath_vid = os.path.join(path, sensor + "_local.txt")

        if (vital_sign == "ppg"):
            filepath_vs = os.path.join(path, "NOM_PLETHWaveExport.csv")
        elif(vital_sign == "ecg"):
            filepath_vs = os.path.join(path, "NOM_ECG_ELEC_POTL_IIWaveExport.csv")
        elif(vital_sign == "resp"):
            filepath_vs = os.path.join(path, "NOM_RESPWaveExport.csv")
        else: # hr_ppg, hr_ecg, resp_rate (#TODO: hr_ECG, Blood Pressure, etc..)
            filepath_vs = os.path.join(path, "MPDataExport.csv")

        print("Interpolating {} signal using {} timestamps".format(vital_sign, sensor) )

        #constucting arrays for the data
        mx_stamps, sys_stamps, data = extract_data(filepath_vs, vital_sign)
        delta_array = find_deltas2(sys_stamps, mx_stamps)
        sys_mx_time_delta = np.mean(delta_array)
        ts_data = apply_delta(mx_stamps, sys_mx_time_delta)
        ts_vid = extract_timestamps(filepath_vid)

        ##CHECK FOR Data AND TS LENGTHS AND CORRECT
        l1 = len(ts_data)
        l2 = len(data)
        if l1<l2:
            data = data[0:l1]
        elif l2<l1:
            ts_data = ts_data[0:l2]

        ts_data_sec = ts_data 
        ts_vid_sec = ts_vid

        #interpolation function
        f = interpolate.interp1d(ts_data_sec,data,kind='linear')

        reinterp_data = []

        for t_temp in ts_vid_sec:
            if t_temp<ts_data_sec[0]:
                reinterp_data.append(data[0])
            elif t_temp>ts_data_sec[-1]:
                reinterp_data.append(data[-1])
            else:
                reinterp_data.append(f(t_temp))
        output = np.array(reinterp_data)

        plt.plot(reinterp_data)
        plt.show()
        return output

def vital_matrix(sensors_list, vital_sign_list, path):
    num_sensors = len(sensors_list)
    num_vitals = len(vital_sign_list)
    num_datapoints = 900

    vital_matrix = np.empty((num_sensors, num_datapoints, num_vitals))

    for vital in vital_sign_list:
        v_idx = vital_sign_list.index(vital)
        for sensor in sensors_list:
            s_idx = sensors_list.index(sensor)
            vital_list = interpolate_timestamp(sensor, vital, path)
            vital_arr = np.array(vital_list)
            vital_matrix[s_idx,:,v_idx] = vital_arr

    #save numpy vital matrix
    filepath_output = os.path.join(path, "vital_matrix.npy")
    np.save(filepath_output, vital_matrix)

def aslist_cronly(value):
    if isinstance(value, string_types):
        value = filter(None, [x.strip() for x in value.splitlines()])
    return list(value)

def aslist(value, flatten=True):
    """ Return a list of strings, separating the input based on newlines
    and, if flatten=True (the default), also split on spaces within
    each line."""
    values = aslist_cronly(value)
    if not flatten:
        return values
    result = []
    for value in values:
        subvalues = value.split()
        result.extend(subvalues)
    return result