import torch 
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader
from dataset import AfricanWildlifeDataset
from utils import intersection_over_union, convert_cellboxes, cellboxes_to_boxes
from loss import YoloLoss
from tqdm import tqdm
from YOLOv1 import YOLOv1
import numpy as np 

def train(train_loader, model, optimizer, loss, epochs):

    """
        Implementation of the training loop.
    """
    
    for epoch in range(EPOCHS):
        # https://github.com/tqdm/tqdm.git
        loop = tqdm(train_loader, leave=True)
        mean_loss = []

        for batch_idx, (x, y) in enumerate(loop):
            x, y = x.to(DEVICE), y.to(DEVICE)
            #forward
            output = model(x)
            loss = criterion(output, y)
            mean_loss.append(loss.item())
            
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # updates the progress bar
            loop.set_postfix(loss=loss.item())

        print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

        #hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16
EPOCHS = 10
TRAIN_DIR = 'African_Wildlife/train'
TEST_DIR = 'African_Wildlife/test'
LABEL_DIR = 'African_Wildlife/annotations'


transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

train_set = AfricanWildlifeDataset(TRAIN_DIR, TEST_DIR, LABEL_DIR, transform=transform)
train_loader = DataLoader(
    dataset = train_set,
    batch_size = BATCH_SIZE, 
    shuffle = True
)

model = YOLOv1(input_channels=3, S=7, B=2, C=4).to(DEVICE)
optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE
    )
criterion = YoloLoss()


train(train_loader, model, optimizer, criterion, EPOCHS)