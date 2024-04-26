import torch
from torch import nn
from torchvision.models import resnet18
from torchvision import transforms as T


class ResNet18Modified(nn.Module):
    # for ease of use, we define a transform here
    transform = T.Compose(
        [
            T.Resize((244, 244)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    def __init__(self, num_classes=30):
        super(ResNet18Modified, self).__init__()
        self.backbone = resnet18()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    
def load_resnet18modified(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Modified()
    checkpoint = torch.load(model_path,map_location=torch.device(device))
    model.backbone.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model
    
    