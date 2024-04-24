import torch
from torch import nn
from torchvision.models import efficientnet_b4
from torchvision import transforms as T


class EfficientNetModified(nn.Module):
    # for ease of use, we define a transform here
    transform = T.Compose(
        [
            T.Resize((180, 320)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def __init__(self, num_classes=30):
        super(EfficientNetModified, self).__init__()
        self.backbone = efficientnet_b4()
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x


def load_efficientnetmodified(model_path):
    model = EfficientNetModified()
    checkpoint = torch.load(model_path)
    model.backbone.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model
