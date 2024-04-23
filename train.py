import os
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
    limit_dataset_size=None,
    val_dataset=None,
    val_every=100,
    val_sample_size=100,
    val_sample_size_epoch=1000,
    model_name="model",
    predict_fix=None,
):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    if limit_dataset_size:
        dataset = torch.utils.data.Subset(dataset, range(limit_dataset_size))

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(cur_epoch, cur_epoch+num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
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
                acc = test_model(model, val_dataset, batch_size=batch_size, limit_dataset_size=val_sample_size)
                print(f"Batch [{i}/{len(progress_bar)}] Validation accuracy: {acc:.2f}%")

        # Print the average loss for each epoch
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f} out of {len(dataloader)} batches"
        )
        
        # test the model
        acc = test_model(model, val_dataset, batch_size=batch_size, limit_dataset_size=val_sample_size_epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation accuracy: {acc:.2f}%")
        
        # Save the model
        save_model(model, optimizer, epoch, f"{model_name}_{epoch}.pth")


def test_model(model, dataset, batch_size=32, limit_dataset_size=None):
    if limit_dataset_size:
        dataset = torch.utils.data.Subset(dataset, range(limit_dataset_size))

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # print(f"Accuracy: {100 * correct / total:.2f}%")
    return 100 * correct / total

def save_model(model, optimizer, epoch, path):
    model_data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(model_data, path)
    
def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return model, optimizer, epoch
