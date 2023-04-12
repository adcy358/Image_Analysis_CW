import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def intersection_over_union(boxes_preds, boxes_labels): 
  
    '''
    Calculates the intersection over union
    
    '''    
    
    # boxes_preds: (BATCH_SIZE, 4) -> (x, y, w, h)
    # boxes_labels (BATCH_SIZE, 4) -> (x, y, w, h)
    #     (0,0) placed on the upper left corner of the cell
    #     x, y: coordinates for the object midpoint in cell
    #     w, h: width and height relative to the cell
    
    # Box1: [box1_x1, box1_y1, box1_x2, box1_y2] 
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2 # x1 for the bounding box relative to the cell
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    
    # Box2: [box2_x1, box2_y1, box2_x2, box2_y2]
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    
    # Intersection box (of Box 1 and Box2): [x1, y1, x2, y2]
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    # Area of Intersection box
    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    # Area of union of Box1 and Box2
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = (box1_area + box2_area - intersection + 1e-6) # 1e-6 to stabilize in case its 0 
    
    # IOU: area of intersection / area of union
    iou = intersection / union
    
    return iou


def convert_cellboxes(predictions, S=7, B=2, C=4): 
    # SOURCE: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/utils.py
    """
    Converts bounding boxes output: 
       from: Yolo with an image split size of S relative to the cell
       to: relative the entire image
    """
    
    # predictions = (N, S, S, [c1, c2, c3, c4, p1, x1, y1, w1, h1, p2, x2, y2, w2, h2]) 
    
    predictions = predictions.to('cpu')
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, (C + B * 5))
    
    # bound_box: [x,y,w,h]
    bound_box1 = predictions[..., 5:9] # (N, S, S, 4)
    bound_box2 = predictions[..., 10:14] # (N, S, S, 4)
    
    # we choose the best_box, i.e., the one with highest probability (p1, p2)
    scores = torch.cat(
        (predictions[..., 4].unsqueeze(0), predictions[..., 9].unsqueeze(0)), dim=0 
    )
    best_box = scores.argmax(0).unsqueeze(-1) # max_indices (0 if box1, 1 if box2)
    best_boxes = (1 - best_box) * bound_box1 + best_box * bound_box2  # here we choose the best boxes from each prediction
    
    # torch.arange(S=7) -> tensor([0, 1, 2, 3, 4, 5, 6])
    # .repeat(batch_size, S, 1) -> (N, S, S)
    # .unsqueeze(-1): (N, S, S) -> (N, S, S, 1)
    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1) #(N, S, S, 1)
    
    # convert x from relative to cell to image
    x = 1 / S * (best_boxes[..., :1] + cell_indices) # (N, S, S, 1)
    # convert y
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3)) # (N, S, S, 1)
    # convert w and h 
    w_y = 1 / S * best_boxes[..., 2:4] # (N, S, S, 2)
    
    converted_bound_boxes = torch.cat((x, y, w_y), dim=-1) # (N, S, S, 4)
    
    # get the number of the predicted class (i.e., 0, 1, 2, 3)
    predicted_class = predictions[..., :4].argmax(-1).unsqueeze(-1) # (N, S, S, 1)
    
    # get the probability of the object being the bounding box
    best_confidence = torch.max(predictions[..., 4], predictions[..., 9]).unsqueeze(-1) # (N, S, S, 1)
    
    # converted_preds: (c, p, x, y, w, h)
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bound_boxes), dim=-1
    ) # (N, S, S, 1+1+4)  

    return converted_preds


def cellboxes_to_boxes(output, S=7, B=2, C=4): 
# SOURCE: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/utils.py    
    # output: (N, S, S, [c1, c2, c3, c4, p1, x1, y1, w1, h1, p2, x2, y2, w2, h2])  
    converted_pred = convert_cellboxes(output, S=S, B=B, C=C).reshape(output.shape[0], S*S, -1) # (N, S*S , 6)
    converted_pred[..., 0] = converted_pred[..., 0].long() 
    all_bound_boxes = []
    
    # for every sample in batch
    #  for every bound box in predictions 
    for sample_idx in range(output.shape[0]): 
        bound_boxes = []

        for bound_box_idx in range(S * S):
            bound_boxes.append([x.item() for x in converted_pred[sample_idx, bound_box_idx, :]])
        all_bound_boxes.append(bound_boxes)

    return all_bound_boxes


def non_max_suppression(predictions, iou_threshold=0.5):
    """
    Implements Non-Maximum Suppression
    
    Input: 
        predictions(tensor): bounding box predictions for an image
        iou_threshold(float): iou threshold
    """
    # predictions: (49, 6) 
    bboxes = predictions.tolist() 
    
    # We sort the bound_boxes in reverse order according to confidence score
    sorted_bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) 
    
    # We remove all the empty boxes
    candidates = [box for box in sorted_bboxes if box[1] != 0]
    
    num_candidates = len(candidates) 
    keep={idx:True for idx, box in enumerate(candidates)} 
    # True: if we keep it 
    # False: if we remove it  
    
    # we compare each image (that is set to True) with the rest
    # and remove the ones that overlap
    for i in range(num_candidates):
        for j in range(i+1, num_candidates): 
            if keep[i]:
                if candidates[i][0] == candidates[j][0]: # they have the same class
                    boxes_iou = intersection_over_union(torch.tensor(candidates[i][2:]), torch.tensor(candidates[j][2:]))
                    if boxes_iou > iou_threshold: 
                        keep[j] = False
        
    keep_boxes = [k for k, v in keep.items() if v]
    bound_boxes_after = [candidates[i] for i in keep_boxes]
    
    return bound_boxes_after


def get_boxes(y_pred, y_true, iou_threshold=0.5): 
    
    """
        Prepares the predictions and ground truths for mAP
        
        Input: 
            y_pred(tensor): torch.size([num_images, num_cells, bounding box])  
            y_true(tensor): torch.size([num_images, num_cells, bounding box]) 
            threshold(float): filter out the 0 pred
            
        Output: 
            all_pred_boxes(list): [ [image_idx, c, p, x, y, w, h], [], ...]
            all_true_boxes(list): [ [image_idx, c, p, x, y, w, h], [], ...]
    
    """

    batch_size = y_pred.shape[0]
    # bound_boxes = y_pred.tolist() # [ [], [], []] 
    true_boxes = y_true.tolist()
    image_idx = 0 # label that determines the image 
    all_pred_boxes = []
    all_true_boxes = []
    
    for idx in range(batch_size): 
        nms_boxes = non_max_suppression(y_pred[idx], iou_threshold=iou_threshold) 

        for nms_box in nms_boxes: 
            if nms_box[2:] != [0, 0, 0, 0]: # to remove the boxes with only zeroes
                all_pred_boxes.append([image_idx] + nms_box) 
        
        for box in true_boxes[idx]: 
            if box[1] > 0: # to remove the boxes with only zeroes
                all_true_boxes.append([image_idx] + box) 
            
        image_idx += 1
        
    return all_pred_boxes, all_true_boxes


def plot_bbox(img_idx, dataset, pred, figsize=448, ax=None):
    """
    Plots an image and its respective predictions

    Attributes
    ----------
    img_idx: int,
        index of image in dataset;
    dataset: torch.utils.data.Dataset
        dataset containing the image;
    pred: list(Tensor(Class, prob, x, y, w, h)),
        predicted bounding boxes to be plotted
    """
    img = dataset.__getitem__(img_idx)[0]
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img.permute(1, 2, 0))

    color_map = {0: "red", 1: "blue", 2: "yellow", 3: "green"}
    for c, p, x, y, w, h in pred:
        if type(c) == torch.Tensor:
            c = c.item()
        c = int(c)
        x0 = (x - w / 2) * figsize
        y0 = (y - h / 2) * figsize
        w *= figsize
        h *= figsize
        rect = Rectangle((x0, y0), w, h, lw=2, edgecolor=color_map[c], fill=False)
        ax.add_patch(rect)
     
