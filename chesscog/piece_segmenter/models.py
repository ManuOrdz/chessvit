from torch import nn
from torchvision import models

from chesscog.core.models import MODELS_REGISTRY
from chesscog.core.registry import Registry

NUM_CLASSES = 1  # len({"pawn", "knight", "bishop", "rook", "queen", "king"}) * 2

#: Registry of piece classifiers (registered in the global :attr:`chesscog.core.models.MODELS_REGISTRY` under the key ``PIECE_CLASSIFIER``)
MODEL_REGISTRY = Registry()
MODELS_REGISTRY.register_as("PIECE_SEGMENTER")(MODEL_REGISTRY)


@MODEL_REGISTRY.register
class DLResNet(nn.Module):
    """DeepLabV3 ResNet model."""

    input_size = 100, 200
    pretrained = True

    def __init__(self):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
        n = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(n, NUM_CLASSES, 1)
        self.params = {"head": list(self.model.classifier[4].parameters())}

    def forward(self, x):
        return self.model(x)
