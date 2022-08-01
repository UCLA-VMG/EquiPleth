import os
import pickle
import numpy as np
import scipy.stats
import sklearn.metrics
import torch

from tqdm import tqdm

from rf import organizer as org
from rf.proc import create_fast_slow_matrix, find_range
from .errors import getErrors
from .utils import extract_video, pulse_rate_from_power_spectral_density

def eval_rgb_model(root_dir, session_names, model, sequence_length = 64, 
                   file_name = "rgbd_rgb", ppg_file_name = "rgbd_ppg.npy", device=torch.device('cpu')):
    model.eval()
    video_samples = []
    for cur_session in session_names:
        video_sample = {"video_path" : os.path.join(root_dir, cur_session)}
        video_samples.append(video_sample)

    for cur_video_sample in tqdm(video_samples):
        cur_video_path = cur_video_sample["video_path"]
        cur_est_ppgs = None

        frames = extract_video(path=cur_video_path, file_str=file_name)
        target = np.load(os.path.join(cur_video_path, ppg_file_name))

        for cur_frame_num in range(frames.shape[0]):
            # Preprocess
            cur_frame = frames[cur_frame_num, :, :, :]
            cur_frame_cropped = torch.from_numpy(cur_frame.astype(np.uint8)).permute(2, 0, 1).float()
            cur_frame_cropped = cur_frame_cropped / 255
            # Add the T dim
            cur_frame_cropped = cur_frame_cropped.unsqueeze(0).to(device) 

            # Concat
            if cur_frame_num % sequence_length == 0:
                cur_cat_frames = cur_frame_cropped
            else:
                cur_cat_frames = torch.cat((cur_cat_frames, cur_frame_cropped), 0)

            # Test the performance
            if cur_cat_frames.shape[0] == sequence_length:
                
                # DL
                with torch.no_grad():
                    # Add the B dim
                    cur_cat_frames = cur_cat_frames.unsqueeze(0) 
                    cur_cat_frames = torch.transpose(cur_cat_frames, 1, 2)
                    # Get the estimated PPG signal
                    cur_est_ppg, _, _, _ = model(cur_cat_frames)
                    cur_est_ppg = cur_est_ppg.squeeze().cpu().numpy()

                # First sequence
                if cur_est_ppgs is None: 
                    cur_est_ppgs = cur_est_ppg
                else:
                    cur_est_ppgs = np.concatenate((cur_est_ppgs, cur_est_ppg), -1)
    
        # Save
        cur_video_sample['est_ppgs'] = cur_est_ppgs
        cur_video_sample['gt_ppgs'] = target[25:]
    print('All finished!')

    # Estimate using waveforms

    hr_window_size = 300
    stride = 128
    mae_list = []
    all_hr_est = []
    all_hr_gt = []
    for index, cur_video_sample in enumerate(video_samples):
        cur_video_path = cur_video_sample['video_path']
        cur_est_ppgs = cur_video_sample['est_ppgs']

        # Load GT
        cur_gt_ppgs = cur_video_sample['gt_ppgs']
        cur_est_ppgs = (cur_est_ppgs - np.mean(cur_est_ppgs)) / np.std(cur_est_ppgs)
        cur_gt_ppgs = (cur_gt_ppgs - np.mean(cur_gt_ppgs)) / np.std(cur_gt_ppgs)

        # Get est HR for each window
        hr_est_temp = []
        hr_gt_temp = []
        for start in range(0, len(cur_est_ppgs) - hr_window_size, stride):
            ppg_est_window = cur_est_ppgs[start:start + hr_window_size]

            ppg_gt_window = cur_gt_ppgs[start:start + hr_window_size]

            # Normalize PPG
            ppg_est_window = (ppg_est_window - np.mean(ppg_est_window)) / np.std(ppg_est_window)
            ppg_gt_window = (ppg_gt_window - np.mean(ppg_gt_window)) / np.std(ppg_gt_window)
            hr_est_temp.append(pulse_rate_from_power_spectral_density(
                ppg_est_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))
            hr_gt_temp.append(pulse_rate_from_power_spectral_density(
                ppg_gt_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))

        hr_est_windowed = np.array([hr_est_temp])
        hr_gt_windowed = np.array(hr_gt_temp)
        all_hr_est.append(hr_est_temp)
        all_hr_gt.append(hr_gt_temp)

        # Errors
        _, MAE, _, _ = getErrors(hr_est_windowed, hr_gt_windowed)

        mae_list.append(MAE)
    print('Mean MAE:', np.mean(np.array(mae_list)))
    return np.array(mae_list), (all_hr_est, all_hr_gt)


def eval_rf_model(root_path, test_files, model, sequence_length = 128, 
                  adc_samples = 256, rf_window_size = 5, freq_slope=60.012e12, 
                  samp_f=5e6, sampling_ratio = 4, device=torch.device('cpu')):
    model.eval()
    video_samples = []
    for rf_folder in tqdm(test_files, total=len(test_files)):

        signal = np.load(f"{root_path}/{rf_folder}/vital_dict.npy", allow_pickle=True).item()['rgbd']['NOM_PLETHWaveExport']

        rf_fptr = open(os.path.join(root_path, rf_folder, "rf.pkl"),'rb')
        s = pickle.load(rf_fptr)

        # Number of samples is set ot 256 for our experiments
        rf_organizer = org.Organizer(s, 1, 1, 1, 2*adc_samples) 
        frames = rf_organizer.organize()
        # The RF read adds zero alternatively to the samples. Remove these zeros.
        frames = frames[:,:,:,0::2] 

        data_f = create_fast_slow_matrix(frames)
        range_index = find_range(data_f, samp_f, freq_slope, adc_samples)
        temp_window = np.blackman(rf_window_size)
        raw_data = data_f[:, range_index-len(temp_window)//2:range_index+len(temp_window)//2 + 1]
        circ_buffer = raw_data[0:800]
        
        # Concatenate extra to generate ppgs of size 3600
        raw_data = np.concatenate((raw_data, circ_buffer))
        raw_data = np.array([np.real(raw_data),  np.imag(raw_data)])
        raw_data = np.transpose(raw_data, axes=(0,2,1))
        rf_data = raw_data

        rf_data = np.transpose(rf_data, axes=(2,0,1))
        cur_video_sample = {}

        cur_est_ppgs = None

        for cur_frame_num in range(rf_data.shape[0]):
            # Preprocess
            cur_frame = rf_data[cur_frame_num, :, :]
            cur_frame = torch.tensor(cur_frame).type(torch.float32)/1.255e5
            # Add the T dim
            cur_frame = cur_frame.unsqueeze(0).to(device)

            # Concat
            if cur_frame_num % (sequence_length*sampling_ratio) == 0:
                cur_cat_frames = cur_frame
            else:
                cur_cat_frames = torch.cat((cur_cat_frames, cur_frame), 0)

            # Test the performance
            if cur_cat_frames.shape[0] == sequence_length*sampling_ratio:
                
                # DL
                with torch.no_grad():
                    # Add the B dim
                    cur_cat_frames = cur_cat_frames.unsqueeze(0)
                    cur_cat_frames = torch.transpose(cur_cat_frames, 1, 2)
                    cur_cat_frames = torch.transpose(cur_cat_frames, 2, 3)
                    IQ_frames = torch.reshape(cur_cat_frames, (cur_cat_frames.shape[0], -1, cur_cat_frames.shape[3]))
                    cur_est_ppg, _ = model(IQ_frames)
                    cur_est_ppg = cur_est_ppg.squeeze().cpu().numpy()

                # First seq
                if cur_est_ppgs is None: 
                    cur_est_ppgs = cur_est_ppg
                else:
                    cur_est_ppgs = np.concatenate((cur_est_ppgs, cur_est_ppg), -1)
    
        # Save
        cur_video_sample['est_ppgs'] = cur_est_ppgs[0:900]
        cur_video_sample['gt_ppgs'] = signal[25:]
        video_samples.append(cur_video_sample)
    print('All finished!')

    # Estimate using waveforms

    hr_window_size = 300
    stride = 128
    mae_list = []
    all_hr_est = []
    all_hr_gt = []
    for index, cur_video_sample in enumerate(video_samples):
        cur_est_ppgs = cur_video_sample['est_ppgs']
        # Load GT
        cur_gt_ppgs = cur_video_sample['gt_ppgs']
        cur_est_ppgs = (cur_est_ppgs - np.mean(cur_est_ppgs)) / np.std(cur_est_ppgs)
        cur_gt_ppgs = (cur_gt_ppgs - np.mean(cur_gt_ppgs)) / np.std(cur_gt_ppgs)

        # Get est HR for each window
        hr_est_temp = []
        hr_gt_temp = []
        for start in range(0, len(cur_est_ppgs) - hr_window_size, stride):
            ppg_est_window = cur_est_ppgs[start:start + hr_window_size]
            ppg_gt_window = cur_gt_ppgs[start:start + hr_window_size]

            # Normalize PPG
            ppg_est_window = (ppg_est_window - np.mean(ppg_est_window)) / np.std(ppg_est_window)
            ppg_gt_window = (ppg_gt_window - np.mean(ppg_gt_window)) / np.std(ppg_gt_window)
            hr_est_temp.append(pulse_rate_from_power_spectral_density(
                ppg_est_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))
            hr_gt_temp.append(pulse_rate_from_power_spectral_density(
                ppg_gt_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))

        hr_est_windowed = np.array([hr_est_temp])
        hr_gt_windowed = np.array(hr_gt_temp)

        all_hr_est.append(hr_est_temp)
        all_hr_gt.append(hr_gt_temp)

        # Errors
        _, MAE, _, _ = getErrors(hr_est_windowed, hr_gt_windowed)
        mae_list.append(MAE)

    print('Mean MAE:', np.mean(np.array(mae_list)))
    return np.array(mae_list), (all_hr_est, all_hr_gt), video_samples


# Test Functions
def get_mapped_fitz_labels(fitz_labels_path, session_names):
    with open(fitz_labels_path, "rb") as fpf:
        out = pickle.load(fpf)

    #mae_list
    #session_names
    sess_w_fitz = []
    fitz_dict = dict(out)
    l_m_d_arr = []
    for i, sess in enumerate(session_names):
        pid = sess.split("_")
        pid = pid[0] + "_" + pid[1]
        fitz_id = fitz_dict[pid]
        if(fitz_id < 3):
            l_m_d_arr.append(1)
        elif(fitz_id < 5):
            l_m_d_arr.append(-1)
        else:
            l_m_d_arr.append(2)
    return l_m_d_arr

def eval_clinical_performance(hr_est, hr_gt, fitz_labels_path, session_names):
    l_m_d_arr = get_mapped_fitz_labels(fitz_labels_path , session_names)
    l_m_d_arr = np.array(l_m_d_arr)
    #absolute percentage error
    # print(hr_gt.shape, hr_est.shape)
    apes = np.abs(hr_gt - hr_est)/hr_gt*100
    # print(apes)
    l_apes = np.reshape(apes[np.where(l_m_d_arr==1)], (-1))
    d_apes = np.reshape(apes[np.where(l_m_d_arr==2)], (-1))

    l_5 = len(l_apes[l_apes <= 5])/len(l_apes)*100 
    d_5 = len(d_apes[d_apes <= 5])/len(d_apes)*100
    
    l_10 = len(l_apes[l_apes <= 10])/len(l_apes)*100
    d_10 = len(d_apes[d_apes <= 10])/len(d_apes)*100

    print("AAMI Standard - L,D")
    print(l_10, d_10)

def eval_performance(hr_est, hr_gt):
    hr_est = np.reshape(hr_est, (-1))
    hr_gt  = np.reshape(hr_gt, (-1))
    r = scipy.stats.pearsonr(hr_est, hr_gt)
    mae = np.sum(np.abs(hr_est - hr_gt))/len(hr_est)
    hr_std = np.std(hr_est - hr_gt)
    hr_rmse = np.sqrt(np.sum(np.square(hr_est-hr_gt))/len(hr_est))
    hr_mape = sklearn.metrics.mean_absolute_percentage_error(hr_est, hr_gt)

    return mae, hr_mape, hr_rmse, hr_std, r[0]

def eval_performance_bias(hr_est, hr_gt, fitz_labels_path, session_names):
    l_m_d_arr = get_mapped_fitz_labels(fitz_labels_path , session_names)
    l_m_d_arr = np.array(l_m_d_arr)

    general_performance = eval_performance(hr_est, hr_gt)
    l_p = np.array(eval_performance(hr_est[np.where(l_m_d_arr == 1)], hr_gt[np.where(l_m_d_arr == 1)]))
    d_p = np.array(eval_performance(hr_est[np.where(l_m_d_arr == 2)], hr_gt[np.where(l_m_d_arr == 2)]))

    performance_diffs = np.array([l_p-d_p])
    performance_diffs = np.abs(performance_diffs)
    performance_max_diffs = performance_diffs.max(axis=0)

    print("General Performance")
    print(general_performance)
    print("Performance Max Differences")
    print(performance_max_diffs)
    print("Performance By Skin Tone")
    print("Light - ", l_p)
    print("Dark - ", d_p)

    return general_performance, performance_max_diffs

def get_discriminator_accuracy(y_prob, y_true):
    '''
    Accuracy function for Discriminator
    '''
    assert y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)

def eval_fusion_model(dataset_test, model, device = torch.device('cpu'), method = 'both'):
    model.eval()
    print(f"Method : {method}")
    mae_list = []
    session_names = []
    hr_est_arr = []
    hr_gt_arr = []
    hr_rgb_arr = []
    hr_rf_arr = []
    est_wv_arr = []
    gt_wv_arr = []
    rgb_wv_arr = []
    rf_wv_arr = []
    for i in range(len(dataset_test)):
        pred_ffts = []
        targ_ffts = []
        pred_rgbs = []
        pred_rfs  = []
        train_sig, gt_sig = dataset_test[i]
        sess_name = dataset_test.all_combs[i][0]["video_path"]
        session_names.append(sess_name)

        train_sig['est_ppgs'] = torch.tensor(train_sig['est_ppgs']).type(torch.float32).to(device)
        train_sig['est_ppgs'] = torch.unsqueeze(train_sig['est_ppgs'], 0)
        train_sig['rf_ppg'] = torch.tensor(train_sig['rf_ppg']).type(torch.float32).to(device)
        train_sig['rf_ppg'] = torch.unsqueeze(train_sig['rf_ppg'], 0)

        gt_sig = torch.tensor(gt_sig).type(torch.float32).to(device)

        with torch.no_grad():
            if method.lower()  == 'rf':
                # Only RF, RGB is noise
                fft_ppg = model(torch.rand(torch.unsqueeze(train_sig['est_ppgs'], axis=0).shape).to(device), torch.unsqueeze(train_sig['rf_ppg'], axis=0))
            elif method.lower() == 'rgb':
                # Only RGB, RF is randn
                fft_ppg = model(torch.unsqueeze(train_sig['est_ppgs'], axis=0), torch.rand(torch.unsqueeze(train_sig['rf_ppg'], axis=0).shape).to(device))
            else:
                # Both RGB and RF
                fft_ppg = model(torch.unsqueeze(train_sig['est_ppgs'], axis=0), torch.unsqueeze(train_sig['rf_ppg'], axis=0))
        # Reduce the dims
        fft_ppg = torch.squeeze(fft_ppg, 1)


        temp_fft = fft_ppg[0].detach().cpu().numpy()
        temp_fft = temp_fft-np.min(temp_fft)
        temp_fft = temp_fft/np.max(temp_fft)

        # Calculate iffts of original signals
        rppg_fft = train_sig['rppg_fft']
        rppg_mag = np.abs(rppg_fft)
        rppg_ang = np.angle(rppg_fft)
        # Replace magnitude with new spectrum
        lix = dataset_test.l_freq_idx 
        rix = dataset_test.u_freq_idx + 1
        roi = rppg_mag[lix:rix]
        temp_fft = temp_fft*np.max(roi)
        rppg_mag[lix:rix] = temp_fft
        rppg_mag[-rix+1:-lix+1] = np.flip(temp_fft)
        rppg_fft_est = rppg_mag*np.exp(1j*rppg_ang)

        rppg_est = np.real(np.fft.ifft(rppg_fft_est))
        rppg_est = rppg_est[0:300] # The 300 is the same as desired_ppg_length given in the dataloader
        gt_est = np.real(np.fft.ifft(train_sig['gt_fft']))[0:300] #The 300 is the same as desired_ppg_length given in the dataloader

        # Re-normalize
        rppg_est = (rppg_est - np.mean(rppg_est)) / np.std(rppg_est)
        gt_est = (gt_est - np.mean(gt_est)) / np.std(gt_est)

        pred_ffts.append(pulse_rate_from_power_spectral_density(rppg_est, 30, 45, 150))
        targ_ffts.append(pulse_rate_from_power_spectral_density(gt_est, 30, 45, 150))
        pred_rgbs.append(pulse_rate_from_power_spectral_density(train_sig['rgb_true'], 30, 45, 150))
        pred_rfs.append(pulse_rate_from_power_spectral_density(train_sig['rf_true'], 30, 45, 150))

        pred_ffts = np.array(pred_ffts)[:,np.newaxis]
        targ_ffts = np.array(targ_ffts)[:,np.newaxis]
        pred_rgbs = np.array(pred_rgbs)[:,np.newaxis]
        pred_rfs = np.array(pred_rfs)[:,np.newaxis]

        hr_est_arr.append(pred_ffts)
        hr_gt_arr.append(targ_ffts)
        hr_rgb_arr.append(pred_rgbs)
        hr_rf_arr.append(pred_rfs)

        _, MAE, _, _ = getErrors(pred_ffts, targ_ffts, PCC=False)

        mae_list.append(MAE)
        est_wv_arr.append(rppg_est)
        gt_wv_arr.append(gt_est)
        rgb_wv_arr.append(train_sig['rgb_true'])
        rf_wv_arr.append(train_sig['rf_true'])
    return np.array(mae_list), session_names, (hr_est_arr, hr_gt_arr), (est_wv_arr,gt_wv_arr, rgb_wv_arr, rf_wv_arr)
