import os
import numpy as np
import torch
import torch.utils
import torch.utils.data
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

from datareader.dataset import GameScreenShotDataset


def create_inceptionv3_model(num_classes):
    # Create the InceptionV3 model
    model = inception_v3()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def inceptionv3_predict_fix(output_res):
    outputs, _ = output_res
    return outputs


def load_datasets(dataset_folder, transform):
    train_dataset = GameScreenShotDataset(
        root=os.path.join(dataset_folder, "train"), transform=transform
    )
    val_dataset = GameScreenShotDataset(
        root=os.path.join(dataset_folder, "val"), transform=transform
    )
    test_dataset = GameScreenShotDataset(
        root=os.path.join(dataset_folder, "test"), transform=transform
    )
    return train_dataset, val_dataset, test_dataset


def train_model(
    model,
    dataset,
    batch_size=32,
    cur_epoch=0,
    num_epochs=1,
    optimizer=None,
    scheduler=None,
    limit_dataset_size=None,
    val_dataset=None,
    val_every=100,
    val_sample_size=100,
    val_sample_size_epoch=1000,
    model_name="model",
    predict_fix=None,
    seed=None,
    dataloader_num_workers=1,
    model_folder="checkpoints",
):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    model_folder = os.path.join("checkpoints", model_name)
    #if limit_dataset_size:
    #    dataset = torch.utils.data.Subset(dataset, range(limit_dataset_size))

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=dataloader_num_workers)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    val_loss_per_epoch = []

    # Training loop
    for epoch in range(cur_epoch, cur_epoch+num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch}/{cur_epoch+num_epochs-1}", leave=False
        )
        for i, (images, labels) in enumerate(progress_bar):
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            # outputs, _ = model(images)
            # I don't know why inception model return tuple. the predict_fix is to fix this problem
            outputs = model(images)
            if predict_fix:
                outputs = predict_fix(outputs)
            
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({"Loss": running_loss / i if i > 0 else 1})
            
            if val_dataset and i % val_every == 0:
                val_loss, val_acc = test_model(model, val_dataset, batch_size=batch_size, limit_dataset_size=val_sample_size,seed=seed, disable_tqdm=True, dataloader_num_workers=dataloader_num_workers)
                print(f"Batch [{i}/{len(progress_bar)}] Validation accuracy: {val_acc:.2f}%, Loss: {val_loss:.4f}")
                
                if i != 0:
                    save_model(model, optimizer, epoch, f"{model_folder}/{model_name}_{epoch}-{i}_({val_loss}).pth")
                    # val_loss_per_epoch.append((epoch, val_loss))

        # Print the average loss for each epoch
        print(
            f"Epoch [{epoch}/{cur_epoch+num_epochs-1}], Loss: {running_loss / len(dataloader):.4f} out of {len(dataloader)} batches"
        )
        
        # test the model
        ep_loss, ep_acc = test_model(model, val_dataset, batch_size=batch_size, limit_dataset_size=val_sample_size_epoch,seed=seed, disable_tqdm=True, dataloader_num_workers=dataloader_num_workers)
        print(f"Epoch [{epoch}/{cur_epoch+num_epochs-1}] Validation accuracy: {ep_acc:.2f}%, Loss: {ep_loss:.4f}")
        val_loss_per_epoch.append((epoch,ep_loss))
        
        # Save the model
        # save_model(model, optimizer, epoch, f"{model_name}_{epoch}_({ep_acc}).pth")
        save_model(model, optimizer, epoch, f"{model_folder}/{model_name}_{epoch}_({ep_acc}).pth")
        
        # Update the learning rate
        if scheduler:
            scheduler.step()

    print("Finished Training")
    print(val_loss_per_epoch)
    return cur_epoch+num_epochs, val_loss_per_epoch

def test_model(model, dataset, batch_size=32, limit_dataset_size=None, seed=None, disable_tqdm=False, dataloader_num_workers=1):
    if limit_dataset_size:
        ignore_size = len(dataset) - limit_dataset_size
        if seed:
            torch.manual_seed(seed)
        dataset, _ = torch.utils.data.random_split(dataset, [limit_dataset_size, ignore_size])
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluation loop
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for images, labels in (pbar:=tqdm(dataloader, desc="Testing", leave=False)):
            pbar.disable = disable_tqdm
            
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images) 
            loss += nn.CrossEntropyLoss()(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print(predicted,labels)
            correct += (predicted == labels).sum().item()

    # print(f"Accuracy: {100 * correct / total:.2f}%")
    # print("Accuracy: {:.2f}%".format(100 * correct / total))
    # print(f"Loss: {loss / len(dataloader)}")
    return loss / len(dataloader), 100 * correct / total

def test_model_topk(model, dataset, batch_size=32, limit_dataset_size=None, topk=1, seed=None, disable_tqdm=False):
    if limit_dataset_size:
        ignore_size = len(dataset) - limit_dataset_size
        if seed:
            torch.manual_seed(seed)
        dataset, _ = torch.utils.data.random_split(dataset, [limit_dataset_size, ignore_size])
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluation loop
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for images, labels in (pbar:=tqdm(dataloader, desc="Testing", leave=False)):
            pbar.disable = disable_tqdm
            
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            # print(nn.CrossEntropyLoss()(outputs, labels).item())
            loss += nn.CrossEntropyLoss()(outputs, labels).item()
            _, predicted = torch.topk(outputs.data, topk, 1)
            total += labels.size(0)
            # print(predicted,labels)
            correct += sum([l in p for l,p in zip(labels,predicted)])
            
    return loss / len(dataloader), 100 * correct / total

def test_model_with_error_record(model, dataset, batch_size=32, limit_dataset_size=None, seed=None, disable_tqdm=False, dataloader_num_workers=1):
    if limit_dataset_size:
        ignore_size = len(dataset) - limit_dataset_size
        if seed:
            torch.manual_seed(seed)
        dataset, _ = torch.utils.data.random_split(dataset, [limit_dataset_size, ignore_size])
    
    if seed:
        torch.manual_seed(seed)
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluation loop
    model.eval()
    correct = 0
    total = 0
    loss = 0
    err = np.array([])
    with torch.no_grad():
        for images, labels in (pbar:=tqdm(dataloader, desc="Testing", leave=False)):
            pbar.disable = disable_tqdm
            
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images) 
            loss += nn.CrossEntropyLoss()(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print(predicted,labels)
            res = (predicted == labels)
            correct += res.sum().item()
            wrong = (predicted != labels).nonzero()
            for i in wrong:
                # print(f"Wrong: {predicted[i].item()}")
                err = np.append(err, predicted[i].item())
                
            # err = np.append(err, wrong)
            # wrong = torch.where(~res)[0]
            # err = np.append(err, wrong)
            # np.append(err, [i for i in range(len(res)) if not res[i]])
            # correct += (predicted == labels).sum().item()  
            
    unique, counts = np.unique(err, return_counts=True)
    braid = dict(zip(unique.astype(int) , counts))
    # print(f"Accuracy: {100 * correct / total:.2f}%")
    # print("Accuracy: {:.2f}%".format(100 * correct / total))
    # print(f"Loss: {loss / len(dataloader)}")
    return loss / len(dataloader), 100 * correct / total, braid

def save_model(model, optimizer, epoch, path):
    model_data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    # create dir
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save(model_data, path)
    
def load_model(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
      optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return model, optimizer, epoch
