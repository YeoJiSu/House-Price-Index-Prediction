import torch
from tqdm import tqdm
import time
import numpy as np
import torch.nn.functional as F

def train_deep_learning_model(model, train_dl, test_dl, criterion, optimizer, num_epochs, patience, save_path):
    best_loss = float('inf')
    patience_counter = 0
    train_loss_list = []
    test_loss_list = []
    train_times = [] # For computational cost analysis
    for epoch in tqdm(range(num_epochs)):
        train_time = 0 # For computational cost analysis
        model.train()
        loss_list = []
        for data, target in train_dl:
            optimizer.zero_grad()
            temp = time.time() # For computational cost analysis
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_time += time.time() - temp # For computational cost analysis
            loss_list.append(loss.item())
        train_loss = sum(loss_list) / len(loss_list)
        train_loss_list.append(train_loss)
        
        train_times.append(train_time/len(train_dl)) # For computational cost analysis
        
        model.eval()
        test_losses = []
        with torch.no_grad():
            for data, target in test_dl:
                output = model(data)
                test_loss = criterion(output, target)
                test_losses.append(test_loss.item())
        test_loss = sum(test_losses) / len(test_losses)
        test_loss_list.append(test_loss)
        
        if train_loss < best_loss:
            torch.save(model.state_dict(), save_path)
            best_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    print("Training time: {}".format(np.sum(train_times)/len(train_times))) # For computational cost analysis
    return train_loss_list, test_loss_list

def train_deep_learning_model_valid(model, train_dl, valid_dl, criterion, optimizer, num_epochs, patience, save_path):
    patience_counter = 0
    
    train_loss_list = []
    valid_loss_list = []
    
    train_times = [] # For computational cost analysis
    best_mse = float('inf')  
    for epoch in tqdm(range(num_epochs)):
        train_time = 0 # For computational cost analysis
        loss_list = []
        model.train()
            
        for data, target in train_dl:
            optimizer.zero_grad()
            temp = time.time() # For computational cost analysis
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_time += time.time() - temp # For computational cost analysis
            loss_list.append(loss.item())
        train_loss = sum(loss_list) / len(loss_list)
        train_loss_list.append(train_loss)
        
        train_times.append(train_time/len(train_dl)) # For computational cost analysis
        
        model.eval()
        valid_losses = []
        total_squared_error = 0.0
        n_samples = 0
        with torch.no_grad():
            for data, target in valid_dl:
                output = model(data)
                valid_loss = criterion(output, target)
                valid_losses.append(valid_loss.item())
                mse = F.mse_loss(output, target, reduction='sum')  
                total_squared_error += mse.item()
                n_samples += target.numel()  # total number of elements
        valid_loss = sum(valid_losses) / len(valid_losses)
        valid_loss_list.append(valid_loss)
        val_mse = total_squared_error / n_samples
    
        if val_mse < best_mse:
            torch.save(model.state_dict(), save_path)
            best_mse = val_mse
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
    print("Training time: {}".format(np.sum(train_times)/len(train_times))) # For computational cost analysis
    return train_loss_list, valid_loss_list

def evaluate_model(model, data_loader):
    test_time = 0 # For computational cost analysis
    model.eval()
    outputs, targets = None, None
    with torch.no_grad():
        for data, target in data_loader:
            temp = time.time() # For computational cost analysis
            output = model(data)
            test_time += time.time() - temp # For computational cost analysis
            outputs = output
            targets = target
    print("Inference time: {}".format(test_time/len(data_loader))) # For computational cost analysis
    return outputs, targets
