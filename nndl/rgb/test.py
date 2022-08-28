import numpy as np 
import pickle
import os
import argparse
import matplotlib.pyplot as plt

import torch

from rgb.model import CNN3D
from utils.eval import eval_performance_bias, eval_clinical_performance, eval_rgb_model

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Argparser.
def parseArgs():
    parser = argparse.ArgumentParser(description='Configs for thr RGB test script')

    parser.add_argument('-dir', '--data-dir', default="./dataset/rgb_files", type=str,
                        help="Parent directory containing the folders with the PNG images and the PPG npy.")
    
    parser.add_argument('-fp', '--fitzpatrick-path', type=str,
                        default="./dataset/fitzpatrick_labels.pkl",
                        help='Pickle file containing the fitzpatrick labels.')

    parser.add_argument('--folds-path', type=str,
                        default="./dataset/demo_fold.pkl",
                        help='Pickle file containing the folds.')
                        
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold Number')

    parser.add_argument('--device', type=str, default=None,
                        help="Device on which the model needs to run (input to torch.device). \
                              Don't specify for automatic selection. Will be modified inplace.")
    
    parser.add_argument('--ppg-str', type=str, default='rgbd',
                        help='')

    parser.add_argument('-ckpt','--checkpoint-path', type=str,
                        default="./ckpt/RGB_CNN3D/best.pth",
                        help='Checkpoint Folder.')

    parser.add_argument('--verbose', action='store_true', help="Verbosity.")

    parser.add_argument('--viz', action='store_true', help="Visualize.")

    return parser.parse_args()

def main(args):
    # Import essential info, i.e. destination folder and fitzpatrick label path
    destination_folder = args.data_dir
    fitz_labels_path = args.fitzpatrick_path
    ckpt_path = args.checkpoint_path
    

    with open(args.folds_path, "rb") as fp:
        files_in_fold = pickle.load(fp)

    v_test = files_in_fold[args.fold]["test"]
    remove_folders = []
    for folder in v_test:
        file_path = os.path.join(destination_folder, folder)
        # Make a list of the folder that do not have the PPG signal.
        if(os.path.exists(file_path)):
            if(os.path.exists(os.path.join(file_path, f"{args.ppg_str}_ppg.npy"))):
                pass
            else:
                if args.verbose:    
                    print(folder, "ppg doesn't exist.")
                remove_folders.append(folder)
        else:
            if args.verbose:    
                print(folder, " doesn't exist.")
            remove_folders.append(folder)
    # Remove the PPGs
    for i in remove_folders:
        v_test.remove(i)    
        if args.verbose:    
            print("Removed", i)


    # Select the device
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    if args.verbose:
        print('Running on device: {}'.format(args.device))

    model = CNN3D().to(args.device)
    model.load_state_dict(torch.load(ckpt_path))

    maes_test, hr_test = eval_rgb_model(destination_folder, v_test, model, device=args.device)

    eval_clinical_performance(hr_est=np.array(hr_test[0]), hr_gt=np.array(hr_test[1]), \
        fitz_labels_path=fitz_labels_path, session_names=v_test)
    print(100*"-")
    eval_performance_bias(hr_est=np.array(hr_test[0]), hr_gt=np.array(hr_test[1]), \
        fitz_labels_path=fitz_labels_path, session_names=v_test)
    print(100*"-")

if __name__ == '__main__':
    args = parseArgs()
    main(args)