import numpy as np 
import pickle
import os
import argparse
import matplotlib.pyplot as plt 

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from rgb.model import CNN3D
from data.datasets import RGBData
from losses.NegPearsonLoss import Neg_Pearson
from utils.eval import eval_rgb_model

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Argparser.
def parseArgs():
    parser = argparse.ArgumentParser(description='Configs for thr RGB train script')

    parser.add_argument('-dir', '--data-dir', default="./dataset/rgb_files", type=str,
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

    parser.add_argument('-ckpts','--checkpoints-path', type=str,
                        default="./ckpt/RGB_CNN3D",
                        help='Checkpoint Folder.')

    parser.add_argument('--verbose', action='store_true', help="Verbosity.")

    parser.add_argument('--viz', action='store_true', help="Visualize.")

    # Train args
    parser.add_argument('--batch-size', type=int, default=8,
                        help="Batch Size for the dataloaders.")

    parser.add_argument('--num-workers', type=int, default=2,
                        help="Number of Workers for the dataloaders.")

    parser.add_argument('--train-shuffle', action='store_true', help="Shuffle the train loader.")
    parser.add_argument('--val-shuffle', action='store_true', help="Shuffle the val loader.")
    parser.add_argument('--test-shuffle', action='store_true', help="Shuffle the test loader.")

    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4,
                        help="Learning Rate for the optimizer.")

    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6,
                        help="Weight Decay for the optimizer.")
    
    parser.add_argument('--epochs', type=int, default=30, help="Number of Epochs.")

    parser.add_argument('--checkpoint-period', type=int, default=1, 
                        help="Checkpoint save period.")

    parser.add_argument('--epoch-start', type=int, default=1, 
                        help="Starting epoch number.")

    return parser.parse_args()

def train_model(args, model, datasets):
    # Instantiate the dataloaders
    train_dataloader = DataLoader(datasets["train"], batch_size=args.batch_size, 
                                  shuffle=args.train_shuffle, num_workers=args.num_workers)
    val_dataloader = DataLoader(datasets["val"], batch_size=args.batch_size, 
                                  shuffle=args.val_shuffle, num_workers=args.num_workers)
    test_dataloader  = DataLoader(datasets["test"], batch_size=args.batch_size, 
                                  shuffle=args.test_shuffle, num_workers=args.num_workers)
    
    if args.verbose:
        print(f"Number of train iterations : {len(train_dataloader)}")
        print(f"Number of val iterations : {len(val_dataloader)}")
        print(f"Number of test iterations : {len(test_dataloader)}")

    ckpt_path = args.checkpoints_path
    latest_ckpt_path = os.path.join(os.getcwd(), f"{ckpt_path}/latest_context.pth")

    # Train Essentials
    loss_fn  = Neg_Pearson()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # A high number to remember the best Loss.
    best_loss = 1e7
    
    # Train configurations
    epochs = args.epochs
    checkpoint_period = args.checkpoint_period
    epoch_start = args.epoch_start
    
    if os.path.exists(latest_ckpt_path):
        print('Context checkpoint exists. Loading state dictionaries.')
        checkpoint = torch.load(latest_ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        epoch_start+=1

    best_loss = np.inf
    for epoch in range(epoch_start, epochs+1):
        # Training Phase
        loss_train = 0
        no_batches = 0
        for batch, (imgs, signal) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            model.train()
            # Convert to the appropriate format and mount on the specified device
            # Normalize image pixel range to [0,1]
            imgs = imgs.type(torch.float32)/(255)
            imgs = imgs.to(args.device)
            signal = signal.to(args.device)

            # Predict the PPG signal and find ther loss
            pred_signal, _, _, _ = model(imgs)
            loss = loss_fn(pred_signal, signal)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate the total loss
            loss_train += loss.item()
            no_batches+=1

        # Save the model every few epochs
        if(epoch % checkpoint_period == 0):
            torch.save(model.state_dict(), os.path.join(os.getcwd(), f"{ckpt_path}/{epoch}.pth"))
            # See if best checkpoint
            mae_loss_list, _ = eval_rgb_model(root_dir=args.data_dir, session_names=datasets["val"].video_list, model=model, device=args.device)
            current_loss = np.mean(mae_loss_list) 
            if(current_loss < best_loss):
                best_loss = current_loss
                torch.save(model.state_dict(), os.path.join(os.getcwd(), f"{ckpt_path}/best.pth"))
                print("Best checkpoint saved!")
            print("Saved Checkpoint!")

        print(f"Epoch: {epoch} ; Loss: {loss_train/no_batches:>7f}")
        # Save context after each epoch
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, latest_ckpt_path)

def main(args):
    # Import essential info, i.e. destination folder and fitzpatrick label path
    destination_folder = args.data_dir
    fitz_labels_path = args.fitzpatrick_path
    ckpt_path = args.checkpoints_path
    

    with open(args.folds_path, "rb") as fp:
        files_in_fold = pickle.load(fp)

    train_files = files_in_fold[args.fold]["train"]
    val_files = files_in_fold[args.fold]["val"]
    test_files = files_in_fold[args.fold]["test"]

    if args.verbose:    
        print(f"There are {len(train_files)} train files. They are : {train_files}")
        print(f"There are {len(val_files)} val files. They are : {val_files}")
        print(f"There are {len(test_files)} test files. They are : {test_files}")

    # Dataset
    dataset_train = RGBData(datapath=destination_folder, datapaths=train_files)
    dataset_val = RGBData(datapath=destination_folder, datapaths=val_files)
    dataset_test = RGBData(datapath=destination_folder, datapaths=test_files)

    # Select the device
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    if args.verbose:
        print('Running on device: {}'.format(args.device))

    # Visualize some examples
    if args.viz:
        train_batch, train_batch_sig = dataset_train[0]
        val_batch, val_batch_sig = dataset_val[0]
        test_batch, test_batch_sig = dataset_test[0]

        if args.verbose:
            print(f"Train data and signal shapes : {train_batch.shape}, {train_batch_sig.shape}")
            print(f"Val data and signal shapes : {val_batch.shape}, {val_batch_sig.shape}")
            print(f"Test data and signal shapes : {test_batch.shape}, {test_batch_sig.shape}")

        plt.figure(); plt.imshow(np.transpose(train_batch[:,0], (1,2,0)))
        plt.figure(); plt.plot(train_batch_sig)

        plt.figure(); plt.imshow(np.transpose(val_batch[:,0], (1,2,0)))
        plt.figure(); plt.plot(val_batch_sig)


        plt.figure(); plt.imshow(np.transpose(test_batch[:,0], (1,2,0)))
        plt.figure(); plt.plot(test_batch_sig)

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
        
    datasets = {"train" : dataset_train, "val" : dataset_val, "test" : dataset_test}
    model = CNN3D(frames=64, channels=3).to(args.device)
    train_model(args, model, datasets)

if __name__ == '__main__':
    args = parseArgs()
    main(args)