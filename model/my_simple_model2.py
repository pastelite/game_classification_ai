import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torch
import torchvision
import torch.nn.functional as F
import math

# Define the model architecture
class MySimpleModel2(nn.Module):
    def __init__(self, inp_shape: tuple[int, int], num_classes: int = 30):
        super(MySimpleModel2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2), use conv2d with stride 2 instead
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )
        # to reduce the number of parameters
        self.finalconv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
        )
        self.fc1 = nn.Linear(32 * math.ceil(inp_shape[0] / 16) * math.ceil(inp_shape[1] / 16), 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.finalconv(x)
        print(x.shape)
        x = x.view(x.size(0), -1) # flatten
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
def load_mysimplemodel2(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MySimpleModel2((320, 180), 30)
    checkpoint = torch.load(model_path,map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

model = MySimpleModel2((320, 180), 30)
print(model(torch.randn(1, 3, 320, 180)).shape)