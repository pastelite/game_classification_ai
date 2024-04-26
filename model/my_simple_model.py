import torch.nn as nn
import torch.utils.data
import torch
import torch.nn.functional as F
from torchvision import transforms as T

# Define the model architecture
class MySimpleModel(nn.Module):
    transform = T.Compose(
        [
            T.Resize((180, 320)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    
    def __init__(self, inp_shape: tuple[int, int], num_classes: int = 30):
        super(MySimpleModel, self).__init__()
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
        
def load_mysimplemodel(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MySimpleModel((320, 180), 30)
    checkpoint = torch.load(model_path,map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model
# model = MyCustomModel((320, 180), 30)
# print(model(torch.randn(1, 3, 320, 180)).shape)