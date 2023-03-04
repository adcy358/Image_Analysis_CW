import torch 

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
    converted_pred = convert_cellboxes(output).reshape(output.shape[0], S*S, -1) # (N, S*S , 6)
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
     
