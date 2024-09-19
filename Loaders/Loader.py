# --------------------------------------------------------------------------------------------------
# --------------------------------- MELD :: Base Data-Loader Class ---------------------------------
# --------------------------------------------------------------------------------------------------
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

import pathlib
import random
import torch
import os
import typing


# --------------------------------------------------------------------------------------------------
# ------------------------------------ CLASS :: Base Data-Loader -----------------------------------
# --------------------------------------------------------------------------------------------------
class Loader(Dataset):

    # ----------------------------------------------------------------------------------------------
    # --------------------------------- CONSTRUCTOR :: Constructor ---------------------------------
    # ----------------------------------------------------------------------------------------------
    def __init__(self, directories: str, transform: typing.Optional[typing.Callable]) -> None:
        super().__init__()

        self.directories = pathlib.Path(directories)
        self.images = []

        for directory in self.directories.iterdir():
            images = [image for image in directory.rglob('*.png') if not image.name.startswith('.')]
            images = random.sample(images, min(len(images), 1000))
            self.images.extend(map(str, images))

        random.shuffle(self.images)

        self.transform = transform or transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(Image.open(self.images[index]).convert('RGB'))