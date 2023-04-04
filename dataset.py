import torch
import os
from PIL import Image


class AfricanWildlifeDataset(torch.utils.data.Dataset):
    def __init__(
        self, train_dir, test_dir, val_dir, label_dir, S=7, B=2, C=4, transform=None, istesting=False, isvalidation=False
    ):
        self.annotations = label_dir
        self.val_dir = val_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.transform = transform
        self.istesting = istesting   
        self.isvalidation = isvalidation
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        
        if self.isvalidation: 
            return len(os.listdir(self.val_dir))
        else:
            return len(os.listdir(self.train_dir)) if not self.istesting else len(os.listdir(self.test_dir))   

        
    def __getitem__(self, index):
        #path to label
        
        if not self.istesting and not self.isvalidation: 
            file = os.listdir(self.train_dir)
            img_path = f'{self.train_dir}/{file[index]}'
            
        elif self.isvalidation:
            file = os.listdir(self.val_dir)
            img_path = f'{self.val_dir}/{file[index]}'
            
        else: 
            file = os.listdir(self.test_dir)
            img_path = f'{self.test_dir}/{file[index]}'
        
        label_name = file[index].split('.')[0]
        label_path = f'{self.annotations}/{label_name}.txt'
        
        boxes = []
        with open(label_path) as f: # open the image 
            for label in f.readlines():
                class_label, x, y, width, height = label.replace("\n", "").split()
                boxes.append([int(class_label), float(x), float(y), float(width), float(height)])
            
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)
        
        # if there are any transformations
        if self.transform:
            # image = self.transform(image)
            image = self.transform(image)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B)) # one bounding box per cell
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            if label_matrix[i, j, 4] == 0:
                # Set that there exists an object
                label_matrix[i, j, 4] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 5:9] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix