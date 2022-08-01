'''
Code of 'Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks' 
By Zitong Yu, 2019/05/05

If you use the code, please cite:
@inproceedings{yu2019remote,
    title={Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks},
    author={Yu, Zitong and Li, Xiaobai and Zhao, Guoying},
    booktitle= {British Machine Vision Conference (BMVC)},
    year = {2019}
}

Only for research purpose, and commercial use is not allowed.

MIT License
Copyright (c) 2019 
      How to use it
    #1. Inference the model
    rPPG, x_visual, x_visual3232, x_visual1616 = model(inputs)
    
    #2. Normalized the Predicted rPPG signal and GroundTruth BVP signal
    rPPG = (rPPG-torch.mean(rPPG)) /torch.std(rPPG)	 	# normalize
    BVP_label = (BVP_label-torch.mean(BVP_label)) /torch.std(BVP_label)	 	# normalize
    
    #3. Calculate the loss
    loss_ecg = Neg_Pearson(rPPG, BVP_label)

'''

import numpy as np
import torch

class Neg_Pearson(torch.nn.Module):    
    def __init__(self) -> None:
        super(Neg_Pearson,self).__init__()
        """ Negative Pearson Correlation Loss. Pearson range [-1, 1] so if < 0, abs|loss| ; if > 0, 1 - loss
        """
        self.epsilon = 1e-2

    def forward(self, preds: torch.tensor, labels: torch.tensor) -> torch.tensor:
        """ Forward Call

        Args:
            preds (torch.tensor): The tensor with the model predictions. [Batch, Temporal]
            labels (torch.tensor): The tensor with the ground truths. [Batch, Temporal]

        Returns:
            loss (torch.tensor): The loss tensor.
        """
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                         # x
            sum_y = torch.sum(labels[i])                        # y
            sum_xy = torch.sum(preds[i]*labels[i])              # xy
            sum_x2 = torch.sum(torch.pow(preds[i],2))           # x^2
            sum_y2 = torch.sum(torch.pow(labels[i],2))          # y^2
            # Temporal length
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y) \
                      / (torch.sqrt((N*sum_x2 - torch.pow(sum_x, 2) + self.epsilon) \
                      * (N*sum_y2 - torch.pow(sum_y, 2) + self.epsilon)))
            loss += 1 - pearson
        # Normalize w.r.t the batch size
        loss = loss/preds.shape[0]
        return loss

class Neg_Pearson2(torch.nn.Module):
    def __init__(self):
        super(Neg_Pearson2,self).__init__()
        """ Squared Negative Pearson Correlation Loss. Pearson range [-1, 1] so if < 0, abs|loss| ; if > 0, 1 - loss
        """
        self.epsilon = 1e-2
        
    def forward(self, preds: torch.tensor, labels: torch.tensor) -> torch.tensor:
        """ Forward Call

        Args:
            preds (torch.tensor): The tensor with the model predictions. [Batch, Temporal]
            labels (torch.tensor): The tensor with the ground truths. [Batch, Temporal]

        Returns:
            loss (torch.tensor): The loss tensor.
        """
        loss = 0
        for i in range(preds.shape[0]):
            x = normalize_signal(preds[i])
            y = normalize_signal(labels[i])

            sum_x = torch.sum(x)                                # x
            sum_y = torch.sum(y)                                # y
            sum_xy = torch.sum(x*y)                             # xy
            sum_x2 = torch.sum(torch.pow(x,2))                  # x^2
            sum_y2 = torch.sum(torch.pow(y,2))                  # y^2
            # Temporal length.
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y) \
                      /(torch.sqrt((N*sum_x2 - torch.pow(sum_x, 2) + self.epsilon) \
                      * (N*sum_y2 - torch.pow(sum_y, 2) + self.epsilon)))
            
            loss += (1 - pearson)**2
        # Normalize w.r.t the batch size.
        loss = loss/preds.shape[0]
        return loss


def normalize_signal(sig: torch.tensor) -> torch.tensor:
    """ Normalize the signal.

    Args:
        sig (torch.tensor): Input signal to normalize.

    Returns:
        torch.tensor: Normalized signal with zero mean and unit variance.
    """
    return (sig-torch.mean(sig)) / (torch.std(sig)+1.00e-6)

