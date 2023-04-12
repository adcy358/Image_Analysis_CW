import torch 
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader
from dataset import AfricanWildlifeDataset
from utils import intersection_over_union, convert_cellboxes, cellboxes_to_boxes
from loss import YoloLoss
from tqdm import tqdm
import numpy as np 

def save_checkpoint(model_state, ckpt_filename): 
    print("-> Saving checkpoint") 
    torch.save(model_state, ckpt_filename)      
    
def load_checkpoint(checkpoint, model, optimizer): 
    print("-> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict']) 
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss_history = checkpoint['loss_history']  

def train(train_loader, model, optimizer, criterion, epochs, DEVICE='cuda', ckpt_filename='checkpoints.tar',
          load_model=False, save_epochs=10):
    
    # WARNING: everytime we set load_model=False, it overwrites the previously saved file.
    if load_model: 
        load_checkpoint(torch.load(ckpt_filename), model, optimizer) 
    
    loss_history = []
    for epoch in range(epochs):
          
        # save checkpoint   
        if epoch % save_epochs == 0 and epoch != 0: 
            checkpoint = {
                'state_dict': model.state_dict(), 
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss_history': loss_history,
            }
            save_checkpoint(checkpoint, ckpt_filename) 
          
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
        print(f"\033[34m EPOCH {epoch + 1}: \033[0m Train loss {avg_loss:.3f}")
        
    if epochs != 0: 
        checkpoint = {
                    'state_dict': model.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch+1, 
                    'loss_history': loss_history,
                }
        save_checkpoint(checkpoint, ckpt_filename)    
        
    return loss_history


