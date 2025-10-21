"""Module containing the CNN architecture definitions of the candidate piece classifiers."""

from torch import nn
from torchvision import models

from chesscog.core.models import MODELS_REGISTRY
from chesscog.core.registry import Registry

NUM_CLASSES = 2

#: Registry of occupancy classifiers (registered in the global :attr:`chesscog.core.models.MODELS_REGISTRY` under the key ``OCCUPANCY_CLASSIFIER``)
MODEL_REGISTRY = Registry()
MODELS_REGISTRY.register_as("OCCUPANCY_CLASSIFIER")(MODEL_REGISTRY)


# @MODEL_REGISTRY.register
# class AlexNet(nn.Module):
#     """AlexNet model."""
#
#     input_size = 100, 100
#     pretrained = True
#
#     def __init__(self):
#         super().__init__()
#         self.model = models.alexnet(weights="DEFAULT")
#         n = self.model.classifier[6].in_features
#         self.model.classifier[6] = nn.Linear(n, NUM_CLASSES)
#         self.params = {"head": list(self.model.classifier[6].parameters())}
#
#     def forward(self, x):
#         return self.model(x)


@MODEL_REGISTRY.register
class ResNet(nn.Module):
    """ResNet model."""

    input_size = 100, 100
    pretrained = True

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights="DEFAULT")
        n = self.model.fc.in_features
        self.model.fc = nn.Linear(n, NUM_CLASSES)
        self.params = {"head": list(self.model.fc.parameters())}

    def forward(self, x):
        return self.model(x)


# @MODEL_REGISTRY.register
# class VGG(nn.Module):
#     """VGG model."""
#
#     input_size = 100, 100
#     pretrained = True
#
#     def __init__(self):
#         super().__init__()
#         self.model = models.vgg11_bn(weights="DEFAULT")
#         n = self.model.classifier[6].in_features
#         self.model.classifier[6] = nn.Linear(n, NUM_CLASSES)
#         self.params = {"head": list(self.model.classifier[6].parameters())}
#
#     def forward(self, x):
#         return self.model(x)

@MODEL_REGISTRY.register
class SwinTV2(nn.Module):
    "Vision Transformer Large model."

    input_size = 224, 224
    pretrained = True

    def __init__(self):
        super().__init__()
        self.model = models.swin_v2_t(weights="DEFAULT")
        n = self.model.head.in_features
        self.model.head = nn.Linear(
            in_features=n, out_features=NUM_CLASSES, bias=True
        )
        self.params = {"head": list(getattr(self.model.heads, "head").parameters())}

    def forward(self, x):
        return self.model(x)
