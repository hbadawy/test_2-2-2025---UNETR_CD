
import os
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.utils import save_image


from Dataset_CD import change_detection_dataset
from Model_gpu import UNETR_2D_CD
# from Loss import DiceLoss, DiceBCELoss
from Utils import seeding

###################### Confusion Matrix ######################
###################### Confusion Matrix ######################
def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction/truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


#################### Test Function ####################
#################### Test Function ####################
# def test(model, loader, loss_fn, device):
  
    
 



#     return 

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
    test_path="D://Datasets//Levir_croped_256_10000//test"

    """ Dataloader """
    batch_size = 4
    test_loader = DataLoader(change_detection_dataset(root_path=test_path, patch_size=patch_size),batch_size=batch_size, shuffle=False,num_workers=0,pin_memory=False)

    """ Model """
    model = UNETR_2D_CD(config, device=device)
    model = model.to(device)
    model.load_state_dict(torch.load("E://VS Projects//test_2-2-2025 - UNETR_CD//model_new.pth"))#, map_location=device, weights_only=True))
    # model.eval()

    test_results_path = "E://VS Projects//test_2-2-2025 - UNETR_CD//test_results2"
    os.makedirs(test_results_path, exist_ok=True)

    TP=0
    TN=0
    FP=0
    FN=0

    
    # with torch.no_grad():
    for _, data in enumerate(test_loader):
            pre_tensor, post_tensor, label_tensor, fname = data

            pre_tensor = pre_tensor.to(device, dtype=torch.float32)
            post_tensor = post_tensor.to(device, dtype=torch.float32)
            label_tensor = label_tensor.to(device, dtype=torch.float32)

            probs = model(pre_tensor, post_tensor)
            print ("model ouput of test:   ", probs.shape)

            prediction = torch.where(probs>0.5, 1.0, 0.0)
            print ("prediction shape:   ", prediction.shape)

            true_positives, false_positives, true_negatives, false_negatives = confusion(prediction, label_tensor)
            TP += true_positives
            TN += true_negatives
            FP += false_positives
            FN += false_negatives
            for i in range(prediction.shape[0]):
                print ("fname:   ", fname[i])
                print ("prediction:   ", prediction[i,:,:,:].shape)
                save_image(prediction[i,:,:,:].cpu(), os.path.join(test_results_path, fname[i]))


    OA=(TP+TN)/(TP+TN+FP+FN)
    Precision=TP/(TP+FP)
    Recall=TP/(TP+FN)
    F1_score=2*Precision*Recall/(Precision+Recall)

    print(f'OA={OA:.3f}, Precision={Precision:.3f}, Recall={Recall:.3f}, F1-score={F1_score:.3f}')






# OA=0.951, Precision=0.170, Recall=0.045, F1-score=0.072