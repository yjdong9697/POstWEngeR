import argparse
from util import *
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import Dataset
from tqdm import tqdm
from build_model import build_model

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=False, type=int, default=0)
    parser.add_argument('--path_to_train_input_data', required=False, type=str, default='./datasets/saved_x.npy')
    parser.add_argument('--path_to_train_label_data', required=False, type=str, default='./datasets/saved_y.npy')
    parser.add_argument('--epoch', required=False, type=int, default=1000)
    parser.add_argument('--batch_size', required=False, type=int, default=128)
    parser.add_argument('--initial_lr', required=False, type=float, default=0.001)



    args = parser.parse_args()

    set_seed(args.seed)

    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    #check_device()

    print("========== Loading Dataset =========")
    input = np.load(args.path_to_train_input_data, allow_pickle=True)
    label = np.load(args.path_to_train_label_data, allow_pickle=True)

    dataset = Dataset.FootBall_Dataset(
        input, label, device=device
    )

    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    val_size = int(dataset_size * 0.1)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    print(f"Training Data Size : {len(train_dataset)}")
    print(f"Validation Data Size : {len(val_dataset)}")
    print(f"Testing Data Size : {len(test_dataset)}")

    print("=========== Data Preprocessing =========== ")
    # normalization, scaling, etc.

    print("Putting data to loader...", end="")
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    print("completed")
    
    print("Loading model/optim/schedualar...", end="")
    model = build_model(src_vocab_size=16, tgt_vocab_size=1, device=device, max_len=3, d_embed=512, n_layer=6, d_model=512, h=8, d_ff=2048, dr_rate=0.1, norm_eps=1e-5)
    # test = model(torch.from_numpy(train_input[:2]).to(device)) 
    # print(test.shape)
    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.initial_lr, weight_decay=0)
    lr_step_size = int(len(train_dataset)/args.batch_size)
    # lr_sch = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_step_size, gamma=0.99)
    print("completed")

    print("########## Start Train ##########")

    for idx_epoch in tqdm(range(args.epoch)):
        start_time = time.time()

        model.train()
        train_loss = 0.
        for idx_batch, (x, y) in enumerate(train_loader):
            model.zero_grad()

            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            # lr_sch.step()
            
        train_loss /= idx_batch+1


        model.eval()
        val_loss = 0.
        for idx_batch, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            
            output = output[:,2]
            y = y[:,2]

            loss = loss_fn(output, y)

            val_loss += loss.item()

        val_loss /= idx_batch+1

        elapsed_time = time.time() - start_time

        print("\r %05d | Train Loss: %.7f | Valid Loss: %.7f | lr: %.7f | time: %.3f" % (idx_epoch+1, train_loss, val_loss, optimizer.param_groups[0]['lr'], elapsed_time))



if __name__ == '__main__':
    train()
