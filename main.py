# --------------------------------------------------------------------------------------------------
# ---------------------------------------- MELD Entry Point ----------------------------------------
# --------------------------------------------------------------------------------------------------
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

import torch.optim as optim
import torch.nn as nn
import torch

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from PIL import Image

import os
import uuid

from Networks.Flattener import Flattener
from Networks.Resampler import Resampler
from Networks.Projector import Projector
from Networks.Encoder import Encoder
from Networks.Decoder import Decoder


# --------------------------------------------------------------------------------------------------
# -------------------------------------- Configure the System --------------------------------------
# --------------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

learning_rate = 1e-4
epochs        = 8
batch_size    = 2

transform = transforms.Compose([
    transforms.Resize(1024),
    transforms.CenterCrop(1024),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

reversals = transforms.Compose([
    transforms.Normalize((-1, -1, -1), (2, 2, 2)),
    transforms.ToPILImage()
])


# --------------------------------------------------------------------------------------------------
# -------------------------------- Initialize the Models and Tools ---------------------------------
# --------------------------------------------------------------------------------------------------
flattener = Flattener()
flattener = flattener.to(device)
flattener.train(True)

resampler = Resampler()
resampler = resampler.to(device)
resampler.train(True)

projector = Projector()
projector = projector.to(device)
projector.train(True)

encoder = Encoder()
encoder = encoder.to(device)
encoder.train(True)

decoder = Decoder()
decoder = decoder.to(device)
decoder.train(True)

criterion = nn.MSELoss()
scaler = GradScaler()

dataset = datasets.ImageFolder(root='/content/drive/MyDrive/datasets/dataset', transform=transform)
loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)


# --------------------------------------------------------------------------------------------------
# --------------------------------------------- Train ----------------------------------------------
# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for epoch in range(epochs):

        running_loss = 0.0

        for index, (batch, label) in enumerate(loader):

            batch = batch.to(device)
            optimizer.zero_grad()

            with autocast():

                encoded = encoder(batch)
                flattened = flattener(encoded)
                resampled, mu, var = resampler(flattened)
                projected = projector(resampled)
                decoded = decoder(projected)

                recon_loss = criterion(decoded, batch)
                kl_loss = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())  # KL Divergence
                loss = recon_loss + kl_loss
                running_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print(f'[{index + 1}] Loss: {loss.item():.4f}')

            for image in range(batch.size(0)):

                source_image = reversals(batch[image].cpu())
                target_image = reversals(decoded[image].cpu())
                merged_image = Image.new('RGB', (2048, 1024))

                combined_image.paste(source_image, (0, 0))
                combined_image.paste(target_image, (1024, 0))

                filename = f'{epoch:06}-{index:06}-{image:06}-{uuid.uuid4().replace("-", "")}.png'
                merged_image.save(f'/content/drive/MyDrive/Output/{filename}')

        running_loss = 0