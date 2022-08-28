from calendar import EPOCH
import numpy as np 
import pickle
import os
import argparse
import matplotlib.pyplot as plt

import torch

from fusion.model import Discriminator, FusionModel
from data.datasets import FusionEvalDatasetObject
from utils.eval import eval_fusion_model, eval_performance_bias, eval_clinical_performance

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Argparser.
def parseArgs():
    parser = argparse.ArgumentParser(description='Configs for thr RF test script')

    parser.add_argument('-dir', '--pickle-file-dir', default="./dataset", type=str,
                        help="Parent directory containing the folders with the pickle file.")
    
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

    parser.add_argument('-ckpt','--checkpoint-folder', type=str,
                        default="./ckpt/Fusion",
                        help='Checkpoint Folder.')

    parser.add_argument('--verbose', action='store_true', help="Verbosity.")

    parser.add_argument('--epochs', type=int, default=300, help="Number of Epochs.")

    return parser.parse_args()

def find_best_ckpt(model, dataset, args):
    best_epoch = 1
    best_mae = np.inf
    for epoch in range(1, args.epochs+1):
        ckpt_path = f'{args.checkpoint_folder}/Gen_{epoch}_epochs.pth'
        model.load_state_dict(torch.load(ckpt_path))
        with torch.no_grad():
            maes, _, _, _ = eval_fusion_model(dataset, model, method='both', device=args.device)
            mean_mae = np.mean(maes)
            if args.verbose:
                print(f"Epoch: {epoch}  ;  Mean MAE: {mean_mae}")
                print(f'{args.checkpoint_folder}/Gen_{epoch}_epochs.pth')
                print("-"*50)
            
            if(mean_mae < best_mae):
                best_mae = mean_mae
                best_epoch = epoch
    return best_epoch

def main(args):
    # Import essential info, i.e. destination folder and fitzpatrick label path
    pickle_file = f'{args.pickle_file_dir}/fold_{args.fold}.pkl'
    fitz_labels_path = args.fitzpatrick_path

    with open(args.folds_path, "rb") as fp:
        files_in_fold = pickle.load(fp)

    train = files_in_fold[args.fold]["train"]
    val = files_in_fold[args.fold]["val"]
    test = files_in_fold[args.fold]["test"]

    # Select the device
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    if args.verbose:
        print('Running on device: {}'.format(args.device))

    dataset = FusionEvalDatasetObject(datapath=pickle_file, datafiles=train, fft_resolution=48, desired_ppg_len=300, compute_fft=True)
    dataset_val = FusionEvalDatasetObject(datapath=pickle_file, datafiles=val, fft_resolution=48, desired_ppg_len=300, compute_fft=True)
    dataset_test = FusionEvalDatasetObject(datapath=pickle_file, datafiles=test, fft_resolution=48, desired_ppg_len=300, compute_fft=True)

    model = FusionModel(base_ppg_est_len=1024, rf_ppg_est_len=1024*5, out_len=1024).to(args.device)

    best_epoch = find_best_ckpt(model, dataset_val, args)
    print(f"The best epoch was found to be {best_epoch}")
    best_ckpt_path = f'{args.checkpoint_folder}/Gen_{best_epoch}_epochs.pth'
    model.load_state_dict(torch.load(best_ckpt_path))

    mae_list, session_names, hr_test, _ = eval_fusion_model(dataset_test, model, method='both', device=args.device)
    if args.verbose:
        print('Mean MAE:', np.mean(np.array(mae_list)))

    eval_clinical_performance(hr_est=np.array(hr_test[0]), hr_gt=np.array(hr_test[1]), \
        fitz_labels_path=fitz_labels_path, session_names=session_names)
    print(100*"-")
    eval_performance_bias(hr_est=np.array(hr_test[0]), hr_gt=np.array(hr_test[1]), \
        fitz_labels_path=fitz_labels_path, session_names=session_names)
    print(100*"-")

if __name__ == '__main__':
    args = parseArgs()
    main(args)