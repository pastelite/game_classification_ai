import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torch
import torchvision
import torch.nn.functional as F

# Define the model architecture
class Net(nn.Module):
    def __init__(self, inp_shape: tuple[int, int], num_classes: int = 30):
        super(Net, self).__init__()
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
        self.fc1 = nn.Linear(64 * (inp_shape[0] // 4) * (inp_shape[1] // 4), 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
model = Net((320, 180), 30)
print(model(torch.randn(1, 3, 320, 180)).shape)