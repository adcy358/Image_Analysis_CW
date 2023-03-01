import torch
import torch.nn as nn 
from utils import intersection_over_union

class YoloLoss(nn.Module): 

    '''
    Compute the loss for the yolo model 
          
    Parameters:  
        S: plit size of image (in paper 7),
        B: number of boxes (in paper 2),
        C: number of classes (for our dataset is 4)
    
    '''
    
    def __init__(self, S=7, B=2, C=4): 
        super(YoloLoss, self).__init__()
        self.S = S 
        self.B = B 
        self.C = C
        self.mse = nn.MSELoss()
        self.lambda_noobject = 0.5
        self.lambda_coord = 5
        
    def forward(self, predictions, target): 
        
        # output is (BATCH_SIZE, S*S(C+B*5))
        predictions = predictions.reshape(-1, self.S, self.S, (self.C + self.B * 5))
        
        # predictions[1] = [c1, c2, c3, c4, p1, x1, y1, w1, h1, p2, x2, y2, w2, h2]  
        iou_box1 = intersection_over_union(predictions[..., 5:9], target[..., 5:9])
        iou_box2 = intersection_over_union(predictions[..., 10:14], target[..., 5:9])
        # iou_box1 =[ [], [], ... ] -> iou_box1.unsqueeze(0): [[ [], [], ... ]]
        ious = torch.cat([iou_box1.unsqueeze(0), iou_box2.unsqueeze(0)], dim=0)
        
        # We assign one predictor to be 'responsible' for predicting an object 
        # Take the box with highest IoU out of the two predictions
        iou_maxes, bestbox = torch.max(ious, dim=0) # two output tensors (max, max_indices)
        exists_box = target[..., 4].unsqueeze(3) # Iobj_i: if object does not appear in the box, p1 will be 0
        

        # LOSS = series 1 + series 2 + series 3 + series 4 + series 5

        
        # SERIES 1 + SERIES 2: specify the coordinates for the midpoint, the width and the height of the object
        #    S1) x_i, y_i, x_hat_i, y_hat_i        
        #    we only compute the box if there is an object in that cell 
        #    we take the coordinates for the box with highest IOU (their indexes are stored in bestbox)
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 10:14]
                + (1 - bestbox) * predictions[..., 5:9]
            )
        )

        box_targets = exists_box * target[..., 5:9]   

        #   S2) w_i, w_i, h_hat_i, h_hat_i
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])


        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )  # end_dim = -2: (N, S, S, 4) -> (N*S*S, 4)


        # SERIES 3: specifies if there is an object 
        pred_box = (
            bestbox * predictions[..., 9:10] + (1 - bestbox) * predictions[..., 4:5]
        ) 

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 4:5]),
        )

        # SERIES 4: specifies if there is no object 
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 4:5], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 4:5], start_dim=1),
        ) # end_dim = 1: (N, S, S, 1) -> (N, S*S*1)

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 9:10], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 4:5], start_dim=1)
        ) 

        # SERIES 5: specifies the class that the object belongs to
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :4], end_dim=-2,),
            torch.flatten(exists_box * target[..., :4], end_dim=-2,),
        )

        # We compute the loss 
        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobject * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss
