import torch
import torch.nn.functional as F
from utils import cellboxes_to_boxes


def test(test_loader, model, DEVICE='cuda', filter_params=[]):
    model.eval()
    predictions_by_batch = []
    gt_by_batch = []
    S, B, C = model.S, model.B, model.C

    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)

        # reshaping and converting coordinates
        output = torch.tensor(cellboxes_to_boxes(output, S=S, B=B, C=C))
        ground_truths = torch.tensor(cellboxes_to_boxes(y, S=S, B=B, C=C))


        # sigmoid on probabilities
        output[..., 1] = torch.sigmoid(output[..., 1] * 2)
        predictions_by_batch.append(output)
        gt_by_batch.append(ground_truths) 


    y_pred = torch.cat(tuple(predictions_by_batch))
    y_pred = filter_predictions(y_pred, *filter_params)
    y_true = torch.cat(tuple(gt_by_batch))

    return y_pred, y_true


def filter_predictions(y, theta=0.5, remove_negative_preds=True):
    """
    Removes predictions where probability _p_ < theta and handles invalid predictions (negative coords).

    Parameters
    ----------
        y: Tensor(N, SxS, (c, p, x, y, w, h)),
            predicted bounding boxes (SxS for each image)
        theta: float,
            probability threshold
        remove_negative_preds: bool
            whether to remove invalid predictions or transform them
    """
    # setting entries with p < theta to 0, keeping shape (N, SxS, 6)
    y[y[..., 1] < theta] = 0

    # predictions with x, y, w, h < 0:
    if remove_negative_preds:
        # setting invalid predictions to 0
        mask = torch.any((y[..., 2:] < 0), dim=2)
        y[mask] = 0
    else:
        # setting invalid x, y to 0
        y[..., 2:4] *= (y[..., 2:4] > 0).int()
        # setting invalid w, h to abs(), at this point all other values are >= 0
        y[y < 0] *= -1

    return y

def non_max_suppression(y_pred, iou_threshold=0.7):
    """
    **Non-max Suppression**
    Reduces the number of predicted boxes for the same object.
    
    Parameters
    ----------
        y_pred: Tensor(N, SxS, (c, p, x, y, w, h)),
            predicted bounding boxes after filtering - contains some entries set to 0
        iou_threshold: float 0~1
            min overlapping for predictions to be considered the same object
    
    """
    for idx in range(y_pred.size()[0]):
        pred_by_img = {idx : y_pred[idx, y_pred[idx, ::, 1] > 0]}
        
    