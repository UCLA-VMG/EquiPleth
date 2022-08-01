import argparse
import numpy as np 
import matplotlib.pyplot as plt 
import os
import torch
from tqdm import tqdm
import pickle

from rgb.model import CNN3D
from rf.model import RF_conv_decoder
from rf.proc import create_fast_slow_matrix, find_range
import rf.organizer as org
from utils.utils import extract_video

def parseArgs():
    parser = argparse.ArgumentParser(description='Configs for thr RGB train script')

    parser.add_argument('-rgbdir', '--rgb-data-dir', type=str, default="./dataset/rgb_files",
                        help="Parent directory containing the folders with the Pickle files and the Vital signs.")

    parser.add_argument('-rfdir', '--rf-data-dir', default="./dataset/rf_files", type=str,
                        help="Parent directory containing the folders with the PNG images and the PPG npy.")

    parser.add_argument('-save', '--save-dir', type=str, default="./dataset",
                        help="Directory to save the pickle file with the generated PPG waveforms.")

    parser.add_argument('--folds-path', type=str,
                        default="./dataset/train_val_test_folds_ld_max.pkl",
                        help='Pickle file containing the folds.')
                        
    parser.add_argument('--fold', type=int, default=3,
                        help='Fold Number')

    parser.add_argument('--device', type=str, default=None,
                        help="Device on which the model needs to run (input to torch.device). \
                              Don't specify for automatic selection. Will be modified inplace.")

    parser.add_argument('--rgb-ckpt', type=str,
                        default="./ckpt/RGB_CNN3D/best.pth",
                        help='Checkpoint path to the best RGB model')

    parser.add_argument('--rf-ckpt', type=str,
                        default="./ckpt/RF_IQ_net/best.pth",
                        help='Checkpoint path to the best RF model')

    parser.add_argument('--verbose', action='store_true', help="Verbosity.")

    return parser.parse_args()

def gen_rgb_preds(root_dir, session_names, model, sequence_length = 64, max_ppg_length = 900,
                   file_name = "rgbd_rgb", ppg_file_name = "rgbd_ppg.npy", device=torch.device('cpu')):
    model.eval()
    video_samples = []
    for cur_session in session_names:
        single_video_sample = {"video_path" : os.path.join(root_dir, cur_session)}
        video_samples.append(single_video_sample)

    for cur_video_sample in tqdm(video_samples):
        try:
            cur_video_path = cur_video_sample["video_path"]
            cur_est_ppgs = None

            frames = extract_video(path=cur_video_path, file_str=file_name)
            circ_buff = frames[0:100]
            frames = np.concatenate((frames, circ_buff))

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
            cur_video_sample['video_path'] = os.path.basename(cur_video_sample['video_path'])
            cur_video_sample['est_ppgs'] = cur_est_ppgs[0:max_ppg_length]
            cur_video_sample['gt_ppgs'] = target[0:max_ppg_length]
        except:
            if args.verbose:
                print("RGB folder does not exist : ", cur_video_path)
    if args.verbose:
        print('All finished!')
    return video_samples


def gen_rf_preds(root_path, test_files, model, sequence_length = 128, max_ppg_length = 900, 
                  adc_samples = 256, rf_window_size = 5, freq_slope=60.012e12, 
                  samp_f=5e6, sampling_ratio = 4, device=torch.device('cpu')):
    model.eval()
    video_samples = []
    for rf_folder in tqdm(test_files, total=len(test_files)):
        try:
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
            cur_video_sample['video_path'] = rf_folder
            cur_video_sample['rf_ppg'] = cur_est_ppgs[0:max_ppg_length]
            cur_video_sample['gt_ppgs'] = signal[0:max_ppg_length]
            video_samples.append(cur_video_sample)
        except:
            if args.verbose:
                print("RF folder does not exist : ", rf_folder)
    if args.verbose:
        print('All finished!')
    return video_samples

def main(args):
    # Read the file list for the fold
    with open(args.folds_path, "rb") as fpf:
        out = pickle.load(fpf)
    train = out[args.fold]["train"]
    val = out[args.fold]["val"]
    test = out[args.fold]["test"]
    # Select the device
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    if args.verbose:
        print('Running on device: {}'.format(args.device))

    # RF
    rf_train = [i[2:] for i in train]
    rf_val = [i[2:] for i in val]
    rf_test = [i[2:] for i in test]

    model_rf = RF_conv_decoder().to(args.device)
    model_rf.load_state_dict(torch.load(args.rf_ckpt))

    rf_train_vids = gen_rf_preds(root_path=args.rf_data_dir, test_files=rf_train, model=model_rf, device=args.device)
    rf_val_vids = gen_rf_preds(root_path=args.rf_data_dir, test_files=rf_val, model=model_rf, device=args.device)
    rf_test_vids = gen_rf_preds(root_path=args.rf_data_dir, test_files=rf_test, model=model_rf, device=args.device)

    # RGB
    model_rgb = CNN3D().to(args.device)
    model_rgb.load_state_dict(torch.load(args.rgb_ckpt))

    rgb_train_vids = gen_rgb_preds(root_dir=args.rgb_data_dir, session_names=train, model=model_rgb, device=args.device)
    rgb_val_vids = gen_rgb_preds(root_dir=args.rgb_data_dir, session_names=val, model=model_rgb, device=args.device)
    rgb_test_vids = gen_rgb_preds(root_dir=args.rgb_data_dir, session_names=test, model=model_rgb, device=args.device)

    print(len(rf_val_vids))
    print(len(rgb_val_vids))

    # Combine the dictionaries
    total_ppg_list = []
    for i in range(len(rf_train_vids)):
        for j in range(len(rgb_train_vids)):
            if(rf_train_vids[i]['video_path'] == rgb_train_vids[j]['video_path'][2:] \
                                and 'est_ppgs' in rgb_train_vids[j]) and 'rf_ppg' in rf_train_vids[i]:
                rgb_train_vids[j]['rf_ppg'] = rf_train_vids[i]['rf_ppg']
                total_ppg_list.append(rgb_train_vids[j])
    for i in range(len(rf_val_vids)):
        for j in range(len(rgb_val_vids)):
            if(rf_val_vids[i]['video_path'] == rgb_val_vids[j]['video_path'][2:] \
                                and 'est_ppgs' in rgb_val_vids[j]) and 'rf_ppg' in rf_val_vids[i]:
                rgb_val_vids[j]['rf_ppg'] = rf_val_vids[i]['rf_ppg']
                total_ppg_list.append(rgb_val_vids[j])
    for i in range(len(rf_test_vids)):
        for j in range(len(rgb_test_vids)):
            if(rf_test_vids[i]['video_path'] == rgb_test_vids[j]['video_path'][2:] \
                                and 'est_ppgs' in rgb_test_vids[j]) and 'rf_ppg' in rf_test_vids[i]:
                rgb_test_vids[j]['rf_ppg'] = rf_test_vids[i]['rf_ppg']
                total_ppg_list.append(rgb_test_vids[j])

    print()
    print(len(total_ppg_list))
    print()

    # Save the final data structure with the PPG waveforms
    with open(f'{args.save_dir}/fold_{args.fold}.pkl', 'wb') as handle:
        pickle.dump(total_ppg_list, handle)

if __name__ == '__main__':
    args = parseArgs()
    main(args)
