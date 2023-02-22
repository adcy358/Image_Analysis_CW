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
     