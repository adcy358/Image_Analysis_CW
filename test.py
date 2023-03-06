import torch
from utils import cellboxes_to_boxes


def test(test_loader, model, DEVICE='cuda', prob_thresh=0.6):
    model.eval()
    predictions_by_batch = []
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)

        # reshaping and converting coordinates
        output = torch.tensor(cellboxes_to_boxes(output))

        # sigmoid on probabilities
        torch.sigmoid_(output[..., 1])
        predictions_by_batch.append(output)

    y_pred = torch.cat(tuple(predictions_by_batch))
    y_pred = filter_predictions(y_pred, prob_thresh)

    return y_pred


def filter_predictions(y, theta=0.6, remove_negative_preds=True):
    """
    Removes predictions where probability _p_ < theta and handles invalid predictions (negative coords).

    Attributes
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
        y[..., 2:4] *= (y[..., 2:4] > 0)
        # setting invalid w, h to abs(), at this point all other values are >= 0
        y[y < 0] *= -1

    return y