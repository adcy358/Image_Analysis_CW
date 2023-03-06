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

def save_checkpoint(model_state, filename='checkpoints.tar'): 
    print("-> Saving checkpoint") 
    torch.save(model_state, filename)      
    
def load_checkpoint(checkpoint): 
    print("-> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict']) 
    optimizer.load_state_dict(checkpoint['optimizer'])  

def train(train_loader, model, optimizer, criterion, epochs, DEVICE='cuda', load_model=False):
    
    # WARNING: everytime we set load_model=False, it overwrites the previously saved file.
    if load_model: 
        load_checkpoint(torch.load('checkpoints.tar')) 
    
    loss_history = []
    for epoch in range(epochs):
          
        # save checkpoint   
        if epoch % 10 == 0 and epoch != 0: 
            checkpoint = {
                'state_dict': model.state_dict(), 
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(checkpoint) 
          
        # https://github.com/tqdm/tqdm.git
        loop = tqdm(train_loader, leave=True)
        mean_loss = []

        for batch_idx, (x, y) in enumerate(loop):
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            loss = criterion(output, y)
            mean_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update progress bar
            loop.set_postfix(loss=loss.item())
        
        avg_loss = sum(mean_loss)/len(mean_loss)
        loss_history.append(avg_loss)
        print(f"\033[34m EPOCH {epoch + 1}: \033[0m Mean loss {avg_loss:.3f}")
        
        return loss_history
