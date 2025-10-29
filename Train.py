
import os
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

from Dataset_CD import change_detection_dataset
from Model_gpu import UNETR_2D_CD
from Loss import DiceLoss, DiceBCELoss
from Utils import seeding

# def train(model, train_loader, val_loader, optimizer, loss_function, device, num_epochs, save_path):

#################### Train Function ####################
#################### Train Function ####################
def train(model, train_loader, optimizer, loss_fn, device):
    loss_list=[]
    
    model.train()
    for _, data in enumerate(train_loader):
        optimizer.zero_grad()
        pre_tensor, post_tensor, label_tensor, fname = data

        pre_tensor = pre_tensor.to(device, dtype=torch.float32)
        post_tensor = post_tensor.to(device, dtype=torch.float32)
        label_tensor = label_tensor.to(device, dtype=torch.float32)

        prediction = model(pre_tensor, post_tensor)

        total_loss = loss_fn(prediction,label_tensor)
        
        loss_list.append(total_loss.item())                    #only append the loss value and ignore the grad to save memory
        total_loss.backward()
        optimizer.step()
    
    loss_avg=sum(loss_list)/len(loss_list)
    return loss_avg


#################### Evaluate Function ####################
#################### Evaluate Function ####################
def evaluate(model, train_loader, loss_fn, device):
    loss_list=[]
    
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(train_loader):
            pre_tensor, post_tensor, label_tensor, fname = data

            pre_tensor = pre_tensor.to(device, dtype=torch.float32)
            post_tensor = post_tensor.to(device, dtype=torch.float32)
            label_tensor = label_tensor.to(device, dtype=torch.float32)

            prediction = model(pre_tensor, post_tensor)

            total_loss = loss_fn(prediction,label_tensor)

            loss_list.append(total_loss.item())                    #only append the loss value and ignore the grad to save memory
    
        loss_avg=sum(loss_list)/len(loss_list)
    return loss_avg


#################### Main Function ####################
#################### Main Function ####################
if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Config """
    config = {}
    config["image_size"] = 256
    config["num_layers"] = 12
    config["hidden_dim"] = 768
    config["mlp_dim"] = 3072
    config["num_heads"] = 12
    config["dropout_rate"] = 0.1
    config["num_patches"] = 256
    config["patch_size"] = 16
    config["num_channels"] = 3

    """ Device """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    """ Dataset """
    patch_size = 16
    train_path="D://Datasets//Levir_croped_256_10000//train"
    val_path="D://Datasets//Levir_croped_256_10000//val"
    test_path="D://Datasets//Levir_croped_256_10000//test"

    """ Dataloader """
    batch_size = 32
    train_loader=DataLoader(change_detection_dataset(root_path=train_path, patch_size=patch_size),batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=False)
    val_loader=DataLoader(change_detection_dataset(root_path=val_path, patch_size=patch_size),batch_size=batch_size, shuffle=False,num_workers=0,pin_memory=False)
    test_loader=DataLoader(change_detection_dataset(root_path=test_path, patch_size=patch_size),batch_size=batch_size, shuffle=False,num_workers=0,pin_memory=False)

    """ Model """
    model = UNETR_2D_CD(config, device=device)
    model = model.to(device)

    """ Loss Function """
    loss_fn = DiceBCELoss()

    """ Optimizer """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    """ Training Loop """
    num_epochs = 30
    # save_path="D://Datasets//Levir_croped_256//LEVIR_CD//model.pth"
    save_path = "E://VS Projects//test_2-2-2025 - UNETR_CD//model_new.pth"

    tr_loss = []
    v_loss = []

    for epoch in range(num_epochs):
        
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        print ("------------------------------------------------------------------------------------")
        print(f"Epoch: {epoch} Train Loss: {train_loss} Val Loss: {val_loss}")

        tr_loss.append(train_loss)
        v_loss.append(val_loss)

    """ Save Model """
    torch.save(model.state_dict(), save_path)





# With CPU: 
# Epoch = 2, took around 10 minutes with the following losses:
# Epoch: 0 Train Loss: 1.6277169696986675 Val Loss: 1.617289699614048
# Epoch: 1 Train Loss: 1.4909338541328907 Val Loss: 1.4281685426831245

# With GPU: 
# Epoch = 2, took less than 1 min with the following losses:
# Epoch: 0 Train Loss: 1.5227098390460014 Val Loss: 1.483910657465458
# Epoch: 1 Train Loss: 1.3995565734803677 Val Loss: 1.4362161681056023

#With GPU
# Epoch = 10, took around 3 mins with the following losses:
# ...............................................
# Epoch: 7 Train Loss: 1.2110683023929596 Val Loss: 1.3746152073144913
# Epoch: 8 Train Loss: 1.1902501955628395 Val Loss: 1.3291277438402176
# Epoch: 9 Train Loss: 1.169368326663971 Val Loss: 1.3471436351537704

#With GPU - Epoch = 30
# Epoch: 0 Train Loss: 1.5227098390460014 Val Loss: 1.483910657465458
# Epoch: 14 Train Loss: 1.0739740040153265 Val Loss: 1.2823707684874535
# Epoch: 16 Train Loss: 1.055117603391409 Val Loss: 1.1986955031752586

# Epoch: 27 Train Loss: 0.8655796553939581 Val Loss: 1.2110243737697601
# Epoch: 28 Train Loss: 0.8470488656312227 Val Loss: 1.2416155487298965
# Epoch: 29 Train Loss: 0.8058522287756205 Val Loss: 1.1824553310871124