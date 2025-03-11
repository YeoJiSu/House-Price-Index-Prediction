import torch
from tqdm import tqdm

def train_deep_learning_model(model, train_dl, test_dl, criterion, optimizer, num_epochs, patience, save_path):
    best_loss = float('inf')
    patience_counter = 0
    train_loss_list = []
    test_loss_list = []
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        loss_list = []
        for data, target in train_dl:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        train_loss = sum(loss_list) / len(loss_list)
        train_loss_list.append(train_loss)
        
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
    return train_loss_list, test_loss_list

def evaluate_model(model, data_loader):
    model.eval()
    outputs, targets = None, None
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            outputs = output
            targets = target
    return outputs, targets
