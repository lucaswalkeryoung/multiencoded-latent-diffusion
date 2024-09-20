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
print(f"Device set to {'cuda' if torch.cuda.is_available() else 'mps'}")

learning_rate = 1e-4
epochs        = 8
batch_size    = 2
print(f"Learning Rate: {learning_rate}")
print(f"Epochs: {epochs}")
print(f"Batch Size: {batch_size}")

transform = transforms.Compose([
    transforms.Resize(1024),
    transforms.CenterCrop(1024),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print(f"Transforms compiled: Resize(1024), CenterCrop(1024), ToTensor(), Normalize()")

reversals = transforms.Compose([
    transforms.Normalize((-1, -1, -1), (2, 2, 2)),
    transforms.ToPILImage()
])

print(f"Reversals compiled: Normalize(), ToPILImage()")


# --------------------------------------------------------------------------------------------------
# -------------------------------- Initialize the Models and Tools ---------------------------------
# --------------------------------------------------------------------------------------------------
flattener = Flattener()
flattener = flattener.to(device)
flattener.train(True)
print(f"Flattener compiled")

resampler = Resampler()
resampler = resampler.to(device)
resampler.train(True)
print(f"Resampler compiled")

projector = Projector()
projector = projector.to(device)
projector.train(True)
print(f"Projector compiled")

encoder = Encoder()
encoder = encoder.to(device)
encoder.train(True)
print(f"Encoder compiled")

decoder = Decoder()
decoder = decoder.to(device)
decoder.train(True)
print(f"Decoder compiled")

criterion = nn.MSELoss()
scaler = GradScaler()
print(f"Criterion compiled: MSELoss")
print(f"Scaler compiled")

print("Collecting dataset...")
dataset = datasets.ImageFolder(root='/content/drive/MyDrive/datasets/dataset', transform=transform)
print("Dataset collected")
print("Loading dataset...")
loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
print("Dataset loaded")

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
print(f"Adam compiled")


# --------------------------------------------------------------------------------------------------
# --------------------------------------------- Train ----------------------------------------------
# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for epoch in range(epochs):

        print(f"Beginning epoch {epoch}")

        running_loss = 0.0

        for index, (batch, label) in enumerate(loader):

            print(f"Beginning batch {index}")

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