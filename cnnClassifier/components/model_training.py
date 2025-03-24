import torch
import torch.nn as nn
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
import os
from tqdm import tqdm
import gc
import time
from collections import defaultdict
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics import MulticlassAUROC
import torch.nn.functional as F


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    """
    This function trains the model for one epoch.

    Args:
        model (nn.Module): The model to be trained
        optimizer (optim.Optimizer): The optimizer used to train the model
        scheduler (optim.lr_scheduler._LRScheduler): The scheduler used to update the learning rate
        dataloader (DataLoader): The dataloader that contains the training data
        device (torch.device): The device used to train the model
        epoch (int): The current epoch number

    Returns:
        tuple: A tuple containing the total loss and AUROC for the current epoch
    """
    model.train()
    dataset_size = 0
    running_loss = 0.0
    running_auroc  = 0.0
    num_batches=len(dataloader)
    bar = tqdm(enumerate(dataloader), total=len(dataloader),position=0, leave=True)
    for batch, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.long)
        
        batch_size = images.size(0)     
        outputs = model(images)    
        loss = criterion(outputs, targets)
        loss.backward()
    
        if (batch + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        multi_auroc = MulticlassAUROC(num_classes=4)
        multi_auroc.update(input=outputs, target=targets)
        auroc = multi_auroc.compute()            
        
        running_loss += (loss.item() * batch_size)
        running_auroc  += (auroc * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size
        if batch % 25 == 0:
            loss, current = epoch_loss, batch
            step = (batch // 25) + epoch*num_batches//25
            #mlflow.log_metric("loss", f"{loss:3f}", step=(step))
            #mlflow.log_metric("MulticlassAUROC", f"{epoch_auroc:3f}", step=(step))
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_Auroc=epoch_auroc,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    
    return epoch_loss, epoch_auroc




def run_training(model, optimizer, scheduler, device, num_epochs, train_loader, valid_loader, run_id="latest"):
    """
    Trains the model for a specified number of epochs, evaluates it on a validation set, 
    and saves the best performing model based on the AUROC metric.

    Args:
        model (nn.Module): The neural network model to be trained.
        optimizer (optim.Optimizer): The optimizer to use for training.
        scheduler (optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        device (torch.device): The device to run the training on (CPU or GPU).
        num_epochs (int): The total number of epochs to train the model.
        train_loader (DataLoader): The DataLoader for the training dataset.
        valid_loader (DataLoader): The DataLoader for the validation dataset.
        run_id (str, optional): The identifier for the current training run. Defaults to "latest".

    Returns:
        tuple: The best model after training and a history dictionary containing 
               training and validation loss and AUROC for each epoch.
    """

    os.makedirs("/home/vikram/Downloads/pop_os_backup/Kidney-Disease-Classification-Deep-Learning-Project-main/artifacts/runs/"+run_id)
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_auroc = -np.inf
    history = defaultdict(list)
    multi_auroc = MulticlassAUROC(num_classes=4,average=None)
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss, train_epoch_auroc = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CONFIG['device'], epoch=epoch)
        
        val_epoch_loss, val_epoch_auroc = valid_one_epoch(model,valid_loader, device=CONFIG['device'], 
                                         epoch=epoch)
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Train AUROC'].append(train_epoch_auroc)
        history['Valid AUROC'].append(val_epoch_auroc)
        history['lr'].append( scheduler.get_lr()[0] )
        
        # deep copy the model
        print(f"best_epoch_auroc {best_epoch_auroc}")
        print(f"val_epoch_auroc {val_epoch_auroc}")
        
        
        if best_epoch_auroc <= val_epoch_auroc:
            print(f"Validation AUROC Improved ({best_epoch_auroc} ---> {val_epoch_auroc})")
            best_epoch_auroc = val_epoch_auroc
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "/AUROC{:.4f}_Loss{:.4f}_epoch{:.0f}.bin".format(val_epoch_auroc, val_epoch_loss, epoch)
            torch.save(model.state_dict(), "/home/vikram/Downloads/pop_os_backup/Kidney-Disease-Classification-Deep-Learning-Project-main/artifacts/" + run_id + PATH)
            # Save a model file from the current directory
            print(f"Model Saved{PATH}")
            
        
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best AUROC: {:.4f}".format(best_epoch_auroc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history



optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], 
                       weight_decay=CONFIG['weight_decay'])
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])


def criterion(outputs, targets):
    """
    Computes the cross-entropy loss between the model's output and the target.

    Args:
        outputs (torch.Tensor): The output of the model
        targets (torch.Tensor): The target labels

    Returns:
        torch.Tensor: The cross-entropy loss
    """
    return nn.CrossEntropyLoss()(outputs, targets)



@torch.inference_mode()
def validate_model(model, dataloader, criterion, device, epoch=1):
    """
    Validates the model on the validation set.

    Args:
        model (nn.Module): The model to be validated
        dataloader (DataLoader): The validation dataloader
        criterion (nn.Module): The loss criterion
        device (torch.device): The device used to validate the model
        epoch (int, optional): The current epoch number. Defaults to 1.

    Returns:
        tuple: A tuple containing the total loss and AUROC for the current epoch
    """
    model.eval()
    dataset_size = 0
    running_loss = 0.0
    running_auroc = 0.0
    
   
    bar = tqdm(enumerate(dataloader), total=len(dataloader), position=0, leave=True)
    for batch, data in bar:
       
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.long)    
        batch_size = images.size(0)
        outputs = model(images)
        loss = criterion(outputs, targets)
        multi_auroc = MulticlassAUROC(num_classes=4)
        multi_auroc.update(input=outputs, target=targets)
        auroc = multi_auroc.compute()
        
        running_loss += (loss.item() * batch_size)
        running_auroc  += (auroc * batch_size)
        dataset_size += batch_size    
        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Auroc=epoch_auroc,)   
    
  
    #mlflow.log_metric("eval_loss", f"{epoch_loss:2f}", step=epoch)
    #mlflow.log_metric("eval_MulticlassAUROC", f"{epoch_auroc:2f}", step=epoch)
    gc.collect()
    
    return epoch_loss, epoch_auroc
