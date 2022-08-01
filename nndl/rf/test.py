import numpy as np 
import pickle
import os
import argparse
import matplotlib.pyplot as plt

import torch

from rf.model import RF_conv_decoder
from utils.eval import eval_rf_model, eval_performance_bias, eval_clinical_performance

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Argparser.
def parseArgs():
    parser = argparse.ArgumentParser(description='Configs for thr RF test script')

    parser.add_argument('-dir', '--data-dir', default="./dataset/rf_files", type=str,
                        help="Parent directory containing the folders with the PNG images and the PPG npy.")
    
    parser.add_argument('-fp', '--fitzpatrick-path', type=str,
                        default="./dataset/ID_only_fitzpatrick_labels_full_volunteers.pickle",
                        help='Pickle file containing the fitzpatrick labels.')

    parser.add_argument('--folds-path', type=str,
                        default="./dataset/train_val_test_folds_ld_max.pkl",
                        help='Pickle file containing the folds.')
                        
    parser.add_argument('--fold', type=int, default=3,
                        help='Fold Number')

    parser.add_argument('--device', type=str, default=None,
                        help="Device on which the model needs to run (input to torch.device). \
                              Don't specify for automatic selection. Will be modified inplace.")

    parser.add_argument('-ckpt','--checkpoint-path', type=str,
                        default="./ckpt/RF_IQ_Net/best.pth",
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
    test_files = [i[2:] for i in v_test]

    # Select the device
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    if args.verbose:
        print('Running on device: {}'.format(args.device))

    model = RF_conv_decoder().to(args.device)
    model.load_state_dict(torch.load(ckpt_path))

    maes_test, hr_test, video_samples = eval_rf_model(destination_folder, test_files, model, device=args.device)

    eval_clinical_performance(hr_est=np.array(hr_test[0]), hr_gt=np.array(hr_test[1]), \
        fitz_labels_path=fitz_labels_path, session_names=v_test)
    print(100*"-")
    eval_performance_bias(hr_est=np.array(hr_test[0]), hr_gt=np.array(hr_test[1]), \
        fitz_labels_path=fitz_labels_path, session_names=v_test)
    print(100*"-")

if __name__ == '__main__':
    args = parseArgs()
    main(args)