import numpy as np 
import matplotlib.pyplot as plt 
import scipy.signal as sig
import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import argparse
from tqdm import tqdm
import imageio
import pickle
from itertools import cycle

from fusion.model import Discriminator, FusionModel
from data.datasets import FusionDatasetObject
from utils.utils import pulse_rate_from_power_spectral_density, distribute_l_m_d
from utils.errors import getErrors
from utils.eval import get_discriminator_accuracy
from losses.NegPearsonLoss import Neg_Pearson, Neg_Pearson2
from losses.SNRLoss import SNRLossOnPreComputedAndWindowedFFT_base 

def parseArgs():
    parser = argparse.ArgumentParser(description='Configs for thr RGB train script')

    parser.add_argument('-dir', '--pickle-file-dir', default="./dataset/", type=str,
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

    parser.add_argument('-ckpts', '--checkpoints-path', type=str,
                        default="./ckpt/Fusion",
                        help='Checkpoint Folder.')

    parser.add_argument('--verbose', action='store_true', help="Verbosity.")

    parser.add_argument('--viz', action='store_true', help="Visualize.")

    # Train args
    parser.add_argument('--batch-size', type=int, default=8,
                        help="Batch Size for the dataloaders.")

    parser.add_argument('--num-workers', type=int, default=2,
                        help="Number of Workers for the dataloaders.")

    parser.add_argument('--shuffle', action='store_true', help="Shuffle the data loader.")
    parser.add_argument('--drop', action='store_true', help="Drop the final sample of the daat loader.")

    parser.add_argument('-lr1', '--learning-rate-1', type=float, default=1e-3,
                        help="Learning Rate for the Generator's optimizer.")

    parser.add_argument('-lr2', '--learning-rate-2', type=float, default=1e-4,
                        help="Learning Rate for the Discriminator's optimizer.")
    
    parser.add_argument('--epochs', type=int, default=300, help="Number of Epochs.")

    parser.add_argument('--checkpoint-period', type=int, default=1, 
                        help="Checkpoint save period.")

    parser.add_argument('--epoch-start', type=int, default=1, 
                        help="Starting epoch number.")

    parser.add_argument('--gan-epoch_thresh', type=int, default=4, 
                        help="Until this epoch number only RPPG-Net is trained.")

    parser.add_argument('--disc-epoch-thresh', type=int, default=4, 
                        help="Until this epoch number only the discriminator is trained (after gan-epoch-thresh).")

    return parser.parse_args()

def train_model(args, model, datasets):
    # Instantiate the dataloaders
    train_dataloader_l = DataLoader(datasets["l"], batch_size=args.batch_size,
                                  shuffle=args.shuffle, drop_last=args.drop,
                                  num_workers=args.num_workers)
    train_dataloader_m = DataLoader(datasets["m"], batch_size=args.batch_size,
                                  shuffle=args.shuffle, drop_last=args.drop,
                                  num_workers=args.num_workers)
    train_dataloader_d  = DataLoader(datasets["d"], batch_size=args.batch_size,
                                  shuffle=args.shuffle, drop_last=args.drop,
                                  num_workers=args.num_workers)

    if args.verbose:
        print(f"Number of iterations : {len(train_dataloader_l)}")
        print(f"Number of iterations : {len(train_dataloader_m)}")
        print(f"Number of iterations : {len(train_dataloader_d)}")

    ckpt_path = args.checkpoints_path
    latest_ckpt_path = os.path.join(os.getcwd(), f"{ckpt_path}/latest_context.pth")

    # Train Essentials
    loss_fn1 = Neg_Pearson2()
    loss_fn2 = SNRLossOnPreComputedAndWindowedFFT_base(datasets["l"].l_freq_idx, device=args.device)
    loss_fn3 = torch.nn.BCELoss() #For the discriminator
    lam = 0.2/20
    optimizerGen = torch.optim.Adam(model['generator'].parameters(), lr=args.learning_rate_1)
    optimizerDisc = torch.optim.Adam(model['discriminator'].parameters(), lr=args.learning_rate_2)

    # Train configurations
    epochs = args.epochs
    checkpoint_period = args.checkpoint_period
    epoch_start = args.epoch_start
    
    if os.path.exists(latest_ckpt_path):
        print('Context checkpoint exists. Loading state dictionaries.')
        checkpoint = torch.load(latest_ckpt_path)
        model['generator'].load_state_dict(checkpoint['generator_state_dict'])
        model['discriminator'].load_state_dict(checkpoint['discriminator_state_dict'])
        optimizerGen.load_state_dict(checkpoint['optimizerGen_state_dict'])
        optimizerDisc.load_state_dict(checkpoint['optimizerDisc_state_dict'])
        epoch_start = checkpoint['epoch']
        epoch_start+=1

    mae_best_loss = np.inf
    flg = 0
    flg_ctr = 0
    for epoch in range(epoch_start, epochs+1):
        # Training Phase
        loss_rppg_train = 0
        loss_disc_train = 0
        loss_gen_train = 0
        acc_disc_train = 0
        acc_gen_train = 0
        no_batches = 0
        print("Starting Epoch: {}".format(epoch))
        for item1, item2, item3 in zip(train_dataloader_l, cycle(train_dataloader_m), cycle(train_dataloader_d)):
            # if no_batches%100==0:
            #     print(no_batches)
            ppg_dict_l, signal_l = item1
            ppg_dict_m, signal_m = item2
            ppg_dict_d, signal_d = item3

            est_ppgs_l = ppg_dict_l['est_ppgs']
            rf_ppg_l = ppg_dict_l['rf_ppg']

            est_ppgs_m = ppg_dict_m['est_ppgs']
            rf_ppg_m = ppg_dict_m['rf_ppg']

            est_ppgs_d = ppg_dict_d['est_ppgs']
            rf_ppg_d = ppg_dict_d['rf_ppg']


            est_ppgs_l = est_ppgs_l.type(torch.float32).to(args.device)
            est_ppgs_l = torch.unsqueeze(est_ppgs_l, 1)
            rf_ppg_l = rf_ppg_l.type(torch.float32).to(args.device)
            # rf_ppg_l = torch.transpose(rf_ppg_l, 1, 2)
            rf_ppg_l = torch.unsqueeze(rf_ppg_l, 1)
            signal_l = signal_l.to(args.device)

            est_ppgs_m = est_ppgs_m.type(torch.float32).to(args.device)
            est_ppgs_m = torch.unsqueeze(est_ppgs_m, 1)
            rf_ppg_m = rf_ppg_m.type(torch.float32).to(args.device)
            # rf_ppg_m = torch.transpose(rf_ppg_m, 1, 2)
            rf_ppg_m = torch.unsqueeze(rf_ppg_m, 1)
            signal_m = signal_m.to(args.device)

            est_ppgs_d = est_ppgs_d.type(torch.float32).to(args.device)
            est_ppgs_d = torch.unsqueeze(est_ppgs_d, 1)
            rf_ppg_d = rf_ppg_d.type(torch.float32).to(args.device)
            # rf_ppg_d = torch.transpose(rf_ppg_d, 1, 2)
            rf_ppg_d = torch.unsqueeze(rf_ppg_d, 1)
            signal_d = signal_d.to(args.device)

            #generate skin tone labels
            labels_l = torch.zeros((est_ppgs_l.shape[0],1)).to(args.device)
            labels_d = torch.ones((est_ppgs_d.shape[0],1)).to(args.device)
            labels_gen = torch.zeros((est_ppgs_l.shape[0]+est_ppgs_d.shape[0],1)).to(args.device)

            
            ##Normalize the waveforms?

            if epoch<=args.gan_epoch_thresh or epoch>args.disc_epoch_thresh:
                #train step on the rppg network- with the standard rppg losses
                model['generator'].train()
                # Predict the PPG signal and find ther loss
                est_ppgs =torch.cat((est_ppgs_l,est_ppgs_m,est_ppgs_d),dim=0)
                rf_ppg =torch.cat((rf_ppg_l,rf_ppg_m,rf_ppg_d),dim=0)
                signal = torch.cat((signal_l,signal_m,signal_d),dim=0)

                pred_signal = model['generator'](est_ppgs,rf_ppg)

                loss1 = loss_fn1(pred_signal, signal)
                # loss2 = loss_fn2(pred_signal, signal)
                # loss3 = loss_fn3(tone, label)
                loss_rppg = loss1#+lam*loss2#-loss3
                # Backprop
                optimizerGen.zero_grad()
                loss_rppg.backward()
                optimizerGen.step()
                loss_rppg_train += loss_rppg.item()

            if epoch>args.disc_epoch_thresh and flg == 0: # and epoch < args.disc_epoch_thresh:
                # ADVERSARIAL PART TRAINING
                # Optimize the discriminator
                model['discriminator'].train()
                est_ppgs_adv =torch.cat((est_ppgs_l,est_ppgs_d),dim=0)
                rf_ppg_adv =torch.cat((rf_ppg_l,rf_ppg_d),dim=0)
                signal_adv = torch.cat((signal_l,signal_d),dim=0)
                labels_adv = torch.cat((labels_l,labels_d),dim=0)

                
                pred_signal_adv = model['generator'](est_ppgs_adv,rf_ppg_adv)
                tones_adv = model['discriminator'](pred_signal_adv)
                loss3 = loss_fn3(tones_adv, labels_adv)
                loss_disc = loss3
                acc_disc = get_discriminator_accuracy(tones_adv, labels_adv)
                #Backprop
                optimizerDisc.zero_grad()
                loss_disc.backward()
                optimizerDisc.step()

                loss_disc_train += loss_disc.item()
                acc_disc_train +=acc_disc

            if epoch>args.gan_epoch_thresh and flg == 1: # epoch > args.disc_epoch_thresh:
                # Optimize the generator
                model['generator'].train()
                est_ppgs_adv =torch.cat((est_ppgs_l,est_ppgs_d),dim=0)
                rf_ppg_adv =torch.cat((rf_ppg_l,rf_ppg_d),dim=0)

                pred_signal_adv = model['generator'](est_ppgs_adv,rf_ppg_adv)

                tones_adv = model['discriminator'](pred_signal_adv)
                loss4 = loss_fn3(tones_adv, labels_gen)
                loss_gen = 4*loss4
                acc_gen = get_discriminator_accuracy(tones_adv, labels_gen)
                #Backprop
                optimizerGen.zero_grad()
                loss_gen.backward()
                optimizerGen.step()

                loss_gen_train += loss_gen.item()
                acc_gen_train += acc_gen
                

            no_batches+=1

        if epoch>args.gan_epoch_thresh:
            flg_ctr +=1
            if flg_ctr%4==0:
                flg = (flg+1)%2
                flg_ctr = 0

        torch.save(model['generator'].state_dict(), os.path.join(os.getcwd(), f"{ckpt_path}/Gen_{epoch}_epochs.pth"))
        torch.save(model['discriminator'].state_dict(), os.path.join(os.getcwd(), f"{ckpt_path}/Disc_{epoch}_epochs.pth"))

        print(f"Epoch: {epoch} ; R-PPG Loss: {loss_rppg_train/no_batches:>7f}; Gen. Loss: {loss_gen_train/no_batches:>7f}; Gen. Acc: {acc_gen_train/no_batches:>7f}; Disc. Loss: {loss_disc_train/no_batches:>7f}; Disc. Acc: {acc_disc_train/no_batches:>7f}")
        # Save latest_context after epoch
        torch.save({
                'epoch': epoch,
                'generator_state_dict': model['generator'].state_dict(),
                'discriminator_state_dict': model['discriminator'].state_dict(),
                'optimizerGen_state_dict': optimizerGen.state_dict(),
                'optimizerDisc_state_dict': optimizerDisc.state_dict(),
                'mae_best_loss': mae_best_loss,
                }, latest_ckpt_path)

def main(args):
    # Select the device
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    if args.verbose:
        print('Running on device: {}'.format(args.device))


    pickle_file = f'{args.pickle_file_dir}/fold_{args.fold}.pkl'
    ckpt_path = args.checkpoints_path

    fold = args.fold
    with open(args.folds_path, "rb") as fpf:
        out = pickle.load(fpf)

    train = out[fold]["train"]
    val = out[fold]["val"]
    test = out[fold]["test"]

    l_m_d_arr = distribute_l_m_d(args.fitzpatrick_path, train)
    train_l = l_m_d_arr[0]
    train_m = l_m_d_arr[1]
    train_d = l_m_d_arr[2]

    dataset_l = FusionDatasetObject(datapath=pickle_file, datafiles=train_l, fft_resolution=48, desired_ppg_len=300, compute_fft=True)
    dataset_m = FusionDatasetObject(datapath=pickle_file, datafiles=train_m, fft_resolution=48, desired_ppg_len=300, compute_fft=True)
    dataset_d = FusionDatasetObject(datapath=pickle_file, datafiles=train_d, fft_resolution=48, desired_ppg_len=300, compute_fft=True)

    # Visualize some examples
    if args.viz:
        l_batch, l_batch_sig = dataset_l[0]
        m_batch, m_batch_sig = dataset_m[0]
        d_batch, d_batch_sig = dataset_d[0]

        if args.verbose:
            print(f"Data and signal shapes : {l_batch.shape}, {l_batch_sig.shape}")
            print(f"Data and signal shapes : {m_batch.shape}, {m_batch_sig.shape}")
            print(f"Data and signal shapes : {d_batch.shape}, {d_batch_sig.shape}")

        plt.figure(); plt.plot(l_batch['est_ppgs'])
        plt.figure(); plt.plot(l_batch['rf_ppg'])
        plt.figure(); plt.plot(l_batch_sig)

        plt.figure(); plt.plot(m_batch['est_ppgs'])
        plt.figure(); plt.plot(m_batch['rf_ppg'])
        plt.figure(); plt.plot(m_batch_sig)

        plt.figure(); plt.plot(d_batch['est_ppgs'])
        plt.figure(); plt.plot(d_batch['rf_ppg'])
        plt.figure(); plt.plot(d_batch_sig)

        plt.show()

    # Create the checkpoints folder if it does not exist
    os.makedirs(ckpt_path, exist_ok=True)

    #Check if Checkpoints exist
    all_ckpts = os.listdir(ckpt_path)
    if(len(all_ckpts) > 0):
        all_ckpts.sort() 
        print(f"Checkpoints already exists at : {all_ckpts}")
    else:
        print("No checkpoints found, starting from scratch!")
        
    datasets = {"l" : dataset_l, "m" : dataset_m, "d" : dataset_d}
    model = {
        'generator':FusionModel(base_ppg_est_len=128, rf_ppg_est_len=128, out_len=128).to(args.device), 
        'discriminator':Discriminator(frames=1081).to(args.device)
        }
    train_model(args, model, datasets)

if __name__ == '__main__':
    args = parseArgs()
    main(args)