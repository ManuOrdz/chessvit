"""Methods specific to handling chess datasets."""

import logging

import chess
import torch
import torchvision
from PIL import Image
import numpy as np
from recap import URI
from recap import CfgNode as CN

from .datasets import Datasets
from .transforms import build_transforms, build_mask_transforms

logger = logging.getLogger(__name__)


class SegmentationDataset(torch.utils.data.Dataset):

    def __init__(self, root, mode, transforms, mask_transforms):
        self.img_dir = root / "pieces" / mode.value
        self.mask_dir = root / "masks" / mode.value
        self.transforms = transforms
        self.mask_transforms = mask_transforms

        self.classes = sorted([d.name for d in self.img_dir.iterdir() if d.is_dir()])

        # Construir lista (imagen, máscara, clase_id)
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            img_dir = self.img_dir / class_name
            mask_dir = self.mask_dir / class_name
            for img_path in img_dir.glob("*.*"):
                mask_path = mask_dir / img_path.name
                if mask_path.exists():
                    self.samples.append((img_path, mask_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, mask_path, class_idx = self.samples[index]

        image = Image.open(img_path)
        mask = Image.open(mask_path).convert("L")

        image = self.transforms(image)
        mask = self.mask_transforms(mask)

        return image, mask


def color_name(color: chess.Color) -> str:
    """Convert a chess color to a string.

    Args:
        color (chess.Color): the color

    Returns:
        str: the string representation
    """
    return {chess.WHITE: "white", chess.BLACK: "black"}[color]


def piece_name(piece: chess.Piece) -> str:
    """Convert a chess piece to a string.

    Args:
        piece (chess.Piece): the piece

    Returns:
        str: the corresponding string
    """
    return f"{color_name(piece.color)}_{chess.piece_name(piece.piece_type)}"


def name_to_piece(name: str) -> chess.Piece:
    """Convert the name of a piece to an instance of :class:`chess.Piece`.

    Args:
        name (str): the name of the piece

    Returns:
        chess.Piece: the instance of :class:`chess.Piece`
    """
    color, piece_type = name.split("_")
    color = color == "white"
    piece_type = chess.PIECE_NAMES.index(piece_type)
    return chess.Piece(piece_type, color)


def build_dataset(
    cfg: CN, mode: Datasets, is_segmentation: bool = False
) -> torch.utils.data.Dataset:
    """Build a dataset from its configuration.

    Args:
        cfg (CN): the config object
        mode (Datasets): the split (important to figure out which transforms to apply)

    Returns:
        torch.utils.data.Dataset: the dataset
    """
    transform = build_transforms(cfg, mode)

    if getattr(cfg.TASK, "TYPE", "classification") == "segmentation":

        mask_transforms = build_mask_transforms(cfg, mode)
        dataset = SegmentationDataset(
            root=URI(cfg.DATASET.PATH),
            mode=mode,
            transforms=transform,
            mask_transforms=mask_transforms,
        )
        return dataset
    else:
        dataset = torchvision.datasets.ImageFolder(
            root=URI(cfg.DATASET.PATH) / mode.value, transform=transform
        )
        return dataset


def build_data_loader(
    cfg: CN, dataset: torch.utils.data.Dataset, mode: Datasets
) -> torch.utils.data.DataLoader:
    """Build a data loader for a dataset.

    Args:
        cfg (CN): the config object
        dataset (torch.utils.data.Dataset): the dataset
        mode (Datasets): the split

    Returns:
        torch.utils.data.DataLoader: the data loader
    """
    shuffle = mode in {Datasets.TRAIN, Datasets.VAL}

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.DATASET.BATCH_SIZE,
        pin_memory=True,
        prefetch_factor=2,
        shuffle=shuffle,
        num_workers=cfg.DATASET.WORKERS,
    )
