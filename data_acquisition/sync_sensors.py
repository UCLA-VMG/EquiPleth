import multiprocessing as mp
import time
import pickle
import shutil
from natsort import natsorted, ns
from progressbar import progressbar

from sensors.config import *
from sensors.rgb_sensor import *
from sensors.rgbd_sensor import *
from sensors.mx800_sensor import *
from sensors.rf_sensor import *
from postproc.data_interpolation import *
from postproc.tiff_to_avi import *
import sensors.rf_UDP.organizer_copy as org

def cleanup_mx800(folder_name):
    file_list = ['MPrawoutput.txt','NOM_ECG_ELEC_POTL_IIWaveExport.csv','NOM_PLETHWaveExport.csv', 'MPDataExport.csv', 'NOM_RESPWaveExport.csv']
    for file in file_list:
        if (os.path.isfile(file)): # if file exists
            shutil.move(file, os.path.join(folder_name, file)) #os.replace cannot copy files across drives

def cleanup_rf():
    rf_dump_path = r"path/to/dump/folder"
    file_list = os.listdir(rf_dump_path)
    file_list_sorted = natsorted(file_list, key=lambda y: y.lower())
    file_list_sorted = file_list_sorted[1:] # remove adc_data_LogFile.txt from list 
    file_list_sorted = file_list_sorted[:-1] # remove adc_data_Raw_LogFile.csv from list
    file_time = []

    for file in file_list_sorted:
        file_path = os.path.join(rf_dump_path, file)
        file_time.append(os.path.getctime(file_path) )

    file_time_arr = np.array(file_time)
    file_time_sorted = np.sort(file_time_arr)
    value = file_time_sorted[-1]
    idx, = np.where(file_time_arr == value)
    idx = idx[0]
    file_list_sorted = file_list_sorted[:idx] + file_list_sorted[idx+1:]

    for file in file_list_sorted:
        os.remove(os.path.join(rf_dump_path, file))

def read_pickle_rf(folder_name):
    file_list = os.listdir(folder_name)
    for file in file_list:
        filename_ext = os.path.basename(file)
        filename, ext = os.path.splitext(filename_ext)
        if (ext == ".pkl"):
            f = open(os.path.join(folder_name , filename_ext),'rb')
            s = pickle.load(f)
            o = org.Organizer(s, 1, 4, 3, 512)
            frames = o.organize()
            print("Shape of RF pickle file: {}".format(frames.shape) )
            to_save = {'frames':frames, 'start_time':s[3], 'end_time':s[4], 'num_frames':len(frames)}
            with open(os.path.join(folder_name , filename + '_read.pkl'), 'wb') as f:
                pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    

mx800_init_time = 3.9
mx800_buffer = 1
wait_time = mx800_init_time + mx800_buffer #MX800 wait time to ensure that it begins recording before any sensor does.
calibrate_mode = config.getint("mmhealth", "calibration_mode") 

def rgb_main(acquisition_time, folder_name, synchronizer):
    rgb_cam = RGB_Sensor(filename="rgb", foldername=folder_name)
    print("Ready rgb cam")
    synchronizer.wait()
    time.sleep(wait_time)
    rgb_cam.acquire(acquisition_time = acquisition_time)

def rgbd_main(acquisition_time, folder_name, synchronizer):
    rgbd_cam = RGBD_Sensor(filename="rgbd", foldername=folder_name)
    print("Ready rgbd cam")
    synchronizer.wait()
    time.sleep(wait_time)
    rgbd_cam.acquire(acquisition_time = acquisition_time)

def rf_main(acquisition_time, folder_name, synchronizer, sensor_on=True):
    rf_s = RF_Sensor(filename="rf", foldername=folder_name, sensor_on=sensor_on)
    if(not sensor_on):
        time.sleep(120) #two minutes warm up time of the RF
    print("Ready rf device")
    synchronizer.wait()
    time.sleep(wait_time)
    rf_s.acquire(acquisition_time = acquisition_time)

def mx800_main(acquisition_time, folder_name, synchronizer):
    mx800_instance = MX800_Sensor(filename="mx800", foldername=folder_name)
    print("Ready mx800 device")
    synchronizer.wait()
    mx800_instance.acquire(acquisition_time = acquisition_time + wait_time)

def progress_main(acquistion_time, folder_name, synchronizer):
    synchronizer.wait()
    # time.sleep(wait_time)
    print("\nProgress:")
    for i in progressbar(range(acquistion_time)):
        time.sleep(1)
    print("\n")

if __name__ == '__main__':

    #Start
    start = time.time()
    #-------------------- Sensor Config ---------------------------
    sensors_str = config.get("mmhealth", "sensors_list")
    sensors_list_str = aslist(sensors_str, flatten=True)

    sensors_list = []
    if(calibrate_mode == 1):
        folder_name = "calibration" + "_"
    else:
        sensors_list = [progress_main]
        folder_name = str(config.getint("mmhealth", "volunteer_id") ) + "_"

    for sensor in sensors_list_str:
        if(sensor == "rgbd"):
            sensors_list.append(rgbd_main)
        elif(sensor == "rgb"):
            sensors_list.append(rgb_main)
        elif(sensor == "rf"):
            sensors_list.append(rf_main)
        elif(sensor == "mx800"):
            sensors_list.append(mx800_main)
        else:
            continue
    
    jobs = []
    print(sensors_list_str)
    num_sensors = len(sensors_list_str) + 1 #RGB, NIR, Polarized, Webcam Audio, Mic Audio
    time_acquire = config.getint("mmhealth", "acquire_time") #seconds
    sync_barrior = mp.Barrier(num_sensors)
    #-------------------- Folder Config ---------------------------
    start_num = 1
    data_folder_name = os.path.join(config.get("mmhealth", "data_path"), folder_name)

    while(os.path.exists(data_folder_name + str(start_num))):
        start_num += 1
    data_folder_name += str(start_num)
    folder_name += str(start_num)
    os.makedirs(data_folder_name)
    #-------------------- Start Sensors ----------------------------
    for sensor in sensors_list:
        proc = mp.Process(target=sensor, args= (time_acquire,folder_name,sync_barrior))
        jobs.append(proc)
        proc.start()

    for job in jobs:
        job.join() 

    end = time.time()
    print("Time taken: {}".format(end-start))
    
    #--------------------- Post-Processing ---------------------------
    if(sensors_list.count(mx800_main) != 0):
        print("Cleaning up MX800 files")
        vital_sign_str = config.get("mmhealth", "vital_sign_list")
        vital_sign_list = aslist(vital_sign_str, flatten=True)
        cleanup_mx800(data_folder_name)
        vital_matrix(sensors_list, vital_sign_list, data_folder_name) # interpolate_ppg_timestamp(sensor_file_name="rgbd_local.txt", file_dir_mx800=data_folder_name)

    if(sensors_list.count(rf_main) != 0):
        print("Cleaning up RF dump files")
        cleanup_rf()
        if (config.getint("mmhealth", "read_rf_pkl") == 1):
            print("Reading RF pickle files")
            read_pickle_rf(data_folder_name)
    
    if (config.getint("mmhealth", "tiff_to_avi") == 1):
        print("Converting .tiff files to .avi")
        file_list = os.listdir(data_folder_name)
        for file in file_list:
            filename_ext = os.path.basename(file)
            ext = os.path.splitext(filename_ext)[1]
            if (ext == ".tiff"):
                tiff_to_avi(os.path.join(data_folder_name, file))