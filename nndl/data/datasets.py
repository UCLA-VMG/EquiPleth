import os
import pickle
import numpy as np 
import imageio
import scipy.signal as sig
from torch.utils.data import Dataset

import rf.organizer as org
from rf.proc import create_fast_slow_matrix, find_range

class RGBData(Dataset):
    def __init__(self, datapath, datapaths, recording_str="rgbd_rgb", ppg_str="rgbd",
                 video_length = 900, frame_length = 64) -> None:
        
        # There is an offset in capturing the signals w.r.t the ground truth.
        self.ppg_offset = 25
        # Number of samples to be created by oversampling one trial.
        self.num_samps = 30
        # Name of the files being read. Name depends on how the file was save. We have saved the file as rgbd_rgb
        self.id_str = recording_str
        self.ppg_str = ppg_str
        # Number of frames in the input video. (Requires all data-samples to have the same number of frames).
        self.video_length = video_length
        # Number of frames in the output tensor sample.
        self.frame_length = frame_length
        
        # Data structure for videos.
        self.datapath = datapath
        # Load videos and signals.
        self.video_list = datapaths
        # The PPG files for the RGB are stored as rgbd_ppg and not rgbd_rgb_ppg.

        self.signal_list = []
        # Load signals
        remove_folders = []
        for folder in self.video_list:
            file_path = os.path.join(datapath, folder)
            # Make a list of the folder that do not have the PPG signal.
            if(os.path.exists(file_path)):
                if(os.path.exists(os.path.join(file_path, f"{self.ppg_str}_ppg.npy"))):
                    signal = np.load(os.path.join(file_path, f"{self.ppg_str}_ppg.npy"))
                    self.signal_list.append(signal[self.ppg_offset:])
                else:
                    print(folder, "ppg doesn't exist.")
                    remove_folders.append(folder)
            else:
                print(folder, " doesn't exist.")
                remove_folders.append(folder)
        # Remove the PPGs
        for i in remove_folders:
            self.video_list.remove(i)    
            print("Removed", i)

        # Extract the stats for the vital signs.
        self.signal_list = np.array(self.signal_list)
        self.vital_mean = np.mean(self.signal_list)
        self.vital_std = np.std(self.signal_list)
        self.signal_list = (self.signal_list - self.vital_mean)/self.vital_std

        # Create a list of video number and valid frame number to extract the data from.
        self.video_nums = np.arange(0, len(self.video_list))
        self.frame_nums = np.arange(0, self.video_length - frame_length - self.ppg_offset)
        
        # Create all possible sampling combinations.
        self.all_idxs = []
        for num in self.video_nums:
            # Generate the start index.
            cur_frame_nums = np.random.randint(low=0, 
                                               high = self.video_length - frame_length - self.ppg_offset, 
                                               size = self.num_samps)
            # Append all the start indices.
            for cur_frame_num in cur_frame_nums:
                self.all_idxs.append((num,cur_frame_num))
            
            
    def __len__(self):
        return int(len(self.all_idxs))
    def __getitem__(self, idx):
        # Get the video number and the starting frame index.
        video_number, frame_start = self.all_idxs[idx]
        # Get video frames for the output video tensor.
        # (Expects each sample to be stored in a folder with the sample name. Each frame is stored as a png)
        item = []
        for img_idx in range(self.frame_length):
            image_path = os.path.join(self.datapath, 
                                str(self.video_list[video_number]), 
                                f"{self.id_str}_{frame_start+img_idx}.png")
            item.append(imageio.imread(image_path))
        item = np.array(item)

        # Add channel dim if no channels in image.
        if(len(item.shape) < 4): 
            item = np.expand_dims(item, axis=3)
        item = np.transpose(item, axes=(3,0,1,2))
        # Get signal.
        item_sig = self.signal_list[int(video_number)][int(frame_start):int(frame_start+self.frame_length)]
        
        # Patch for the torch constructor. uint16 is a not an acceptable data-type.
        if(item.dtype == np.uint16):
            item = item.astype(np.int32)
        return np.array(item), np.array(item_sig)


# class RFRppgData_RAM(Dataset):
class RFDataRAMVersion(Dataset):
    def __init__(self, datapath, datapaths, ppg_signal_length=900, frame_length_ppg=512, sampling_ratio=4, \
                 window_size=5, samples=256, samp_f=5e6, freq_slope=60.012e12, static_dataset_samples = 30) -> None:
        
        # There is an offset in capturing the signals w.r.t the ground truth.
        self.ppg_offset = 25
        # Number of samples to be created by oversampling one trial.
        self.num_samps = static_dataset_samples

        # Data structure for videos.
        self.datapath = datapath
        # Load videos and signals.
        self.rf_file_list = datapaths
        self.signal_list = []

        # Load signals.
        remove_list_folder = []
        for folder in self.rf_file_list:
            file_path = os.path.join(datapath, folder)
            if(os.path.exists(os.path.join(file_path,"vital_dict.npy"))):
                signal = np.load(f"{file_path}/vital_dict.npy", allow_pickle=True).item()['rgbd']['NOM_PLETHWaveExport']
                self.signal_list.append(signal[self.ppg_offset:])
            else:
                remove_list_folder.append(folder)
     
        # Remove unwanted folders from the list.
        # NOTE: This is done in-place. So the folders will be removed from the list passed into this class.
        for folder in remove_list_folder:
            self.rf_file_list.remove(folder)
            print(f"Removed {folder} from the RF file list (In-place execution).")

        # Normalize the GT
        self.signal_list = np.array(self.signal_list)
        self.vital_mean = np.mean(self.signal_list)
        self.vital_std = np.std(self.signal_list)
        self.signal_list = (self.signal_list - self.vital_mean)/self.vital_std

        # The ratio of the sampling frequency of the RF signal and the PPG signal.
        self.sampling_ratio = sampling_ratio
        
        # Save the RF config parameters.
        self.window_size = window_size
        self.samples = samples
        self.samp_f = samp_f
        self.freq_slope = freq_slope

        # Window the PPG and the RF samples.
        self.ppg_signal_length = ppg_signal_length
        self.frame_length_ppg = frame_length_ppg
        self.frame_nums_rf = np.arange(0, sampling_ratio \
                                       * (self.ppg_signal_length - frame_length_ppg \
                                       - self.ppg_offset), step=sampling_ratio)
        self.frame_nums_ppg = np.arange(0, self.ppg_signal_length \
                                        - frame_length_ppg - self.ppg_offset)
        self.frame_nums = [(i,j) for i,j in zip(self.frame_nums_rf, self.frame_nums_ppg) ]
        self.rf_file_nums = np.arange(len(self.rf_file_list))

        self.all_idxs = []
        for num in self.rf_file_nums:
            cur_frame_nums = np.random.randint(
              low=0, high = self.ppg_signal_length - frame_length_ppg - self.ppg_offset, size = self.num_samps)
            rf_cur_frame_nums = cur_frame_nums*4
            
            for rf_frame_num, cur_frame_num in zip(rf_cur_frame_nums, cur_frame_nums):
                self.all_idxs.append((num,(rf_frame_num, cur_frame_num)))

        # High-ram, compute FFTs before starting training.
        self.rf_data_list = []
        for rf_file in self.rf_file_list:
            # Read the raw RF data
            rf_fptr = open(os.path.join(self.datapath, rf_file, "rf.pkl"),'rb')
            s = pickle.load(rf_fptr)

            # Organize the raw data from the RF.
            # Number of samples is set ot 256 for our experiments.
            rf_organizer = org.Organizer(s, 1, 1, 1, 2*self.samples)
            frames = rf_organizer.organize()
            # The RF read adds zero alternatively to the samples. Remove these zeros.
            frames = frames[:,:,:,0::2]

            # Process the organized RF data
            data_f = create_fast_slow_matrix(frames)
            range_index = find_range(data_f, self.samp_f, self.freq_slope, self.samples)
            # Get the windowed raw data for the network
            raw_data = data_f[:, range_index-self.window_size//2:range_index+self.window_size//2 + 1]
            # Note that item is a complex number due to the nature of the algorithm used to extract and process the pickle file.
            # Hence for simplicity we separate the real and imaginary parts into 2 separate channels.
            raw_data = np.array([np.real(raw_data),  np.imag(raw_data)])
            raw_data = np.transpose(raw_data, axes=(0,2,1))

            self.rf_data_list.append(raw_data)

    def __len__(self):
        return int(len(self.all_idxs))

    def __getitem__(self, idx):
        # This part is hard-coded for our settings. TX and RX = 1.
        file_num, (rf_start, ppg_start) = self.all_idxs[idx]
        
        # Get the RF data.
        data_f = self.rf_data_list[file_num]
        data_f = data_f[:,:,rf_start : rf_start + (self.sampling_ratio * self.frame_length_ppg)]
        item = data_f

        # Get the PPG signal.
        item_sig = self.signal_list[file_num][ppg_start:ppg_start+self.frame_length_ppg]
        assert len(item_sig) == self.frame_length_ppg, f"Expected signal of length {self.frame_length_ppg}, but got signal of length {len(item_sig)}"

        return item, np.array(item_sig)

class FusionDatasetObject(Dataset):
    def __init__(self, datapath, datafiles, \
                    compute_fft=True, fs=30, l_freq_bpm=45, u_freq_bpm=180, \
                        desired_ppg_len=None, fft_resolution = 1, num_static_samples=1, window_rf=False, rf_window_size=5) -> None:
        # There is an offset in the dataset between the captured video and GT
        self.ppg_offset = 25
        #Data structure for videos
        self.datapath = datapath
        self.datafiles = datafiles
        self.desired_ppg_len = desired_ppg_len
        self.compute_fft = compute_fft
        self.fs = fs
        self.l_freq_bpm = l_freq_bpm
        self.u_freq_bpm = u_freq_bpm

        self.window_rf = window_rf
        self.fft_resolution = fft_resolution
        self.rf_window_size = rf_window_size

        # Load the data from the pickle file
        with open(datapath, 'rb') as f:
            pickle_data = pickle.load(f)
        
        # Is any of the 4 keys (video path, estimated ppg from rgb, ground truth ppg, ppg from rf) is missing, we drop that point
        self.usable_data = []
        for data_pt in pickle_data:
            if data_pt['video_path'] in self.datafiles:
                if len(data_pt) != 4:
                    # self.usable_data.remove(data_pt)
                    print(f"{data_pt['video_path']} is dropped")
                    continue
                self.usable_data.append(data_pt)

        
        # If we want to use smaller window of the signals rather than the whole signal itself
        self.all_combs = []
        if self.desired_ppg_len is not None:
            self.num_static_samples = num_static_samples
            for data_pt in self.usable_data:
                static_idxs = np.random.randint(0, len(data_pt['gt_ppgs']) - self.desired_ppg_len - self.ppg_offset, size=self.num_static_samples)

                for idx in static_idxs:
                    self.all_combs.append((data_pt, idx))
            seq_len = self.desired_ppg_len*self.fft_resolution##HERE
        else:
            for data_pt in self.usable_data:
                self.all_combs.append((data_pt, None))
                seq_len = len(data_pt['gt_ppgs'])*self.fft_resolution
        print(f"Dataset Ready. There are {self.__len__()} samples")

        freqs_bpm = np.fft.fftfreq(seq_len, d=1/self.fs) * 60
        self.l_freq_idx = np.argmin(np.abs(freqs_bpm - self.l_freq_bpm))
        self.u_freq_idx = np.argmin(np.abs(freqs_bpm - self.u_freq_bpm))
        print(self.l_freq_idx, self.u_freq_idx)
        print(freqs_bpm[self.l_freq_idx], freqs_bpm[self.u_freq_idx])
        assert self.l_freq_idx < self.u_freq_idx
        
            
    def __len__(self):
        return len(self.all_combs)

    def __getitem__(self, idx):

        dict_item, start_idx = self.all_combs[idx]
        # dict_keys(['video_path', 'est_ppgs', 'gt_ppgs', 'rf_ppg'])
        # Get the ppg data of the rgb, gt and rf
        item = {'est_ppgs':dict_item['est_ppgs'], 'rf_ppg':dict_item['rf_ppg']}
        item_sig = dict_item['gt_ppgs']
        if self.desired_ppg_len is not None:
            assert start_idx is not None
            item_sig = item_sig[start_idx+self.ppg_offset:start_idx+self.ppg_offset+self.desired_ppg_len]
            item['est_ppgs'] = item['est_ppgs'][start_idx:start_idx+self.desired_ppg_len]
            item['rf_ppg'] = item['rf_ppg'][start_idx:start_idx+self.desired_ppg_len]
        
        item_sig = (item_sig - np.mean(item_sig)) / np.std(item_sig)
        item['est_ppgs'] = (item['est_ppgs'] - np.mean(item['est_ppgs'])) / np.std(item['est_ppgs'])
        item['rf_ppg'] = (item['rf_ppg'] - np.mean(item['rf_ppg'], axis = 0)) / np.std(item['rf_ppg'], axis = 0)

        if self.compute_fft:
            n_curr = len(item_sig) * self.fft_resolution
            fft_gt  = np.abs(np.fft.fft(item_sig, n=int(n_curr), axis=0))
            fft_gt = fft_gt / np.max(fft_gt, axis=0)
            
            fft_est = np.abs(np.fft.fft(item['est_ppgs'], n=int(n_curr), axis=0))
            fft_est = fft_est / np.max(fft_est, axis=0)
            fft_est = fft_est[self.l_freq_idx : self.u_freq_idx + 1]

            fft_rf  = np.abs(np.fft.fft(item['rf_ppg'], n=int(n_curr), axis=0))
            fft_rf = fft_rf[self.l_freq_idx : self.u_freq_idx + 1]
            if(self.window_rf):
                center_idx = np.argmax(fft_est)
                window_size = self.rf_window_size
                if(center_idx - window_size <= 0):
                    center_idx = window_size + 1
                elif(center_idx + window_size + 1 >= len(fft_est)):
                    center_idx = len(fft_est) - window_size - 1
                mask = np.zeros_like(fft_rf)
                mask[center_idx-window_size:center_idx+window_size+1,:] = 1
                fft_rf = np.multiply(fft_rf, mask)
                fft_rf = fft_rf / np.max(fft_rf)
            else:
                fft_rf = fft_rf / np.max(fft_rf, axis=0)
            return {'est_ppgs':fft_est, 'rf_ppg':fft_rf}, fft_gt[self.l_freq_idx : self.u_freq_idx + 1]
        else:
            item_sig         = self.lowPassFilter(item_sig)
            item['est_ppgs'] = self.lowPassFilter(item['est_ppgs'])
            for i in range(item['rf_ppg'].shape[1]):
                item['rf_ppg'][:,i]   = self.lowPassFilter(item['rf_ppg'][:,i])

            item_sig = (item_sig - np.mean(item_sig)) / np.std(item_sig)
            item['est_ppgs'] = (item['est_ppgs'] - np.mean(item['est_ppgs'])) / np.std(item['est_ppgs'])
            item['rf_ppg'] = (item['rf_ppg'] - np.mean(item['rf_ppg'], axis = 0)) / np.std(item['rf_ppg'], axis = 0)

            return item, np.array(item_sig)

    def lowPassFilter(self, BVP, butter_order=4):
        [b, a] = sig.butter(butter_order, [self.l_freq_bpm/60, self.u_freq_bpm/60], btype='bandpass', fs = self.fs)
        filtered_BVP = sig.filtfilt(b, a, np.double(BVP))
        return filtered_BVP

class FusionEvalDatasetObject(Dataset):
    def __init__(self, datapath, datafiles, \
                    compute_fft=True, fs=30, l_freq_bpm=45, u_freq_bpm=180, \
                        desired_ppg_len=None, fft_resolution = 1, num_static_samples=7, window_rf=False, rf_window_size=5) -> None:        
        # There is an offset in the dataset between the captured video and GT
        self.ppg_offset = 25
        #Data structure for videos
        self.datapath = datapath
        self.datafiles = datafiles
        
        self.desired_ppg_len = desired_ppg_len
        self.compute_fft = compute_fft
        
        self.fs = fs
        self.l_freq_bpm = l_freq_bpm
        self.u_freq_bpm = u_freq_bpm

        self.window_rf = window_rf
        self.fft_resolution = fft_resolution
        self.rf_window_size = rf_window_size

        # Load the data from the pickle file
        with open(datapath, 'rb') as f:
            pickle_data = pickle.load(f)
        
        # Is any of the 4 keys (video path, estimated ppg from rgb, ground truth ppg, ppg from rf) is missing, we drop that point
        self.usable_data = []
        for data_pt in pickle_data:
            if data_pt['video_path'] in self.datafiles:
                if len(data_pt) != 4:
                    # self.usable_data.remove(data_pt)
                    print(f"{data_pt['video_path']} is dropped")
                    continue
                self.usable_data.append(data_pt)

        
        # If we want to use smaller window of the signals rather than the whole signal itself
        self.all_combs = []
        if self.desired_ppg_len is not None:
            self.num_static_samples = num_static_samples
            for data_pt in self.usable_data:
                # TODO crosscheck this and pass as a param
                static_idxs = np.array([0,128,256,384,512])

                for idx in static_idxs:
                    self.all_combs.append((data_pt, idx))
            seq_len = self.desired_ppg_len*self.fft_resolution
        else:
            for data_pt in self.usable_data:
                self.all_combs.append((data_pt, None))
                seq_len = len(data_pt['gt_ppgs'])*self.fft_resolution
        print(f"Dataset Ready. There are {self.__len__()} samples")

        freqs_bpm = np.fft.fftfreq(seq_len, d=1/self.fs) * 60
        self.l_freq_idx = np.argmin(np.abs(freqs_bpm - self.l_freq_bpm))
        self.u_freq_idx = np.argmin(np.abs(freqs_bpm - self.u_freq_bpm))
        print(self.l_freq_idx, self.u_freq_idx)
        print(freqs_bpm[self.l_freq_idx], freqs_bpm[self.u_freq_idx])
        assert self.l_freq_idx < self.u_freq_idx
        
            
    def __len__(self):
        return len(self.all_combs)

    def __getitem__(self, idx):

        dict_item, start_idx = self.all_combs[idx]
        # dict_keys(['video_path', 'est_ppgs', 'gt_ppgs', 'rf_ppg'])
        # Get the ppg data of the rgb, gt and rf
        item = {'est_ppgs':dict_item['est_ppgs'], 'rf_ppg':dict_item['rf_ppg']}
        item_sig = dict_item['gt_ppgs']
        if self.desired_ppg_len is not None:
            assert start_idx is not None
            item_sig = item_sig[start_idx+self.ppg_offset:start_idx+self.ppg_offset+self.desired_ppg_len]
            item['est_ppgs'] = item['est_ppgs'][start_idx:start_idx+self.desired_ppg_len]
            item['rf_ppg'] = item['rf_ppg'][start_idx:start_idx+self.desired_ppg_len]
        
        item_sig = (item_sig - np.mean(item_sig)) / np.std(item_sig)
        item['est_ppgs'] = (item['est_ppgs'] - np.mean(item['est_ppgs'])) / np.std(item['est_ppgs'])
        item['rf_ppg'] = (item['rf_ppg'] - np.mean(item['rf_ppg'], axis = 0)) / np.std(item['rf_ppg'], axis = 0)

        if self.compute_fft:
            n_curr = len(item_sig) * self.fft_resolution
            fft_gt  = np.abs(np.fft.fft(item_sig, n=int(n_curr), axis=0))
            fft_gt = fft_gt / np.max(fft_gt, axis=0)
            
            fft_est = np.abs(np.fft.fft(item['est_ppgs'], n=int(n_curr), axis=0))
            fft_est = fft_est / np.max(fft_est, axis=0)
            fft_est = fft_est[self.l_freq_idx : self.u_freq_idx + 1]

            fft_rf  = np.abs(np.fft.fft(item['rf_ppg'], n=int(n_curr), axis=0))
            fft_rf = fft_rf[self.l_freq_idx : self.u_freq_idx + 1]

            #Get full ffts
            rppg_fft = np.fft.fft(item['est_ppgs'], n=int(n_curr), axis=0)
            rf_fft = np.fft.fft(item['rf_ppg'], n=int(n_curr), axis=0)
            gt_fft = np.fft.fft(item_sig, n=int(n_curr), axis=0)

            if(self.window_rf):
                center_idx = np.argmax(fft_est)
                window_size = self.rf_window_size
                if(center_idx - window_size <= 0):
                    center_idx = window_size + 1
                elif(center_idx + window_size + 1 >= len(fft_est)):
                    center_idx = len(fft_est) - window_size - 1
                mask = np.zeros_like(fft_rf)
                mask[center_idx-window_size:center_idx+window_size+1,:] = 1
                fft_rf = np.multiply(fft_rf, mask)
                fft_rf = fft_rf / np.max(fft_rf)
            else:
                fft_rf = fft_rf / np.max(fft_rf, axis=0)
            return {'est_ppgs':fft_est, 'rf_ppg':fft_rf, 'rppg_fft':rppg_fft, 'rf_fft':rf_fft, 'gt_fft':gt_fft, 'rgb_true': item['est_ppgs'], 'rf_true': item['rf_ppg'], 'start_idx': start_idx}, fft_gt[self.l_freq_idx : self.u_freq_idx + 1]
        else:
            item_sig         = self.lowPassFilter(item_sig)
            item['est_ppgs'] = self.lowPassFilter(item['est_ppgs'])
            for i in range(item['rf_ppg'].shape[1]):
                item['rf_ppg'][:,i]   = self.lowPassFilter(item['rf_ppg'][:,i])

            item_sig = (item_sig - np.mean(item_sig)) / np.std(item_sig)
            item['est_ppgs'] = (item['est_ppgs'] - np.mean(item['est_ppgs'])) / np.std(item['est_ppgs'])
            item['rf_ppg'] = (item['rf_ppg'] - np.mean(item['rf_ppg'], axis = 0)) / np.std(item['rf_ppg'], axis = 0)

            return item, np.array(item_sig)

    def lowPassFilter(self, BVP, butter_order=4):
        [b, a] = sig.butter(butter_order, [self.l_freq_bpm/60, self.u_freq_bpm/60], btype='bandpass', fs = self.fs)
        filtered_BVP = sig.filtfilt(b, a, np.double(BVP))
        return filtered_BVP
