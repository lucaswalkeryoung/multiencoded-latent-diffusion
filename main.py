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

if device.type == 'mps':
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

learning_rate = 1e-4
epochs        = 8
batch_size    = 1
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

bid = uuid.uuid4().hex
if device.type == 'cuda':
    os.mkdir(f'/content/drive/MyDrive/Output/{bid}')
else:
    os.mkdir(f'/Users/lucasyoung/Desktop/Art/Output/{bid}')
print(f"BID: {bid}")


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

if device.type == 'cuda':
    scaler = GradScaler('cuda')
    print(f"Scaler compiled")

criterion = nn.MSELoss()
print(f"Criterion compiled")

print("Collecting dataset...")
if device.type == 'cuda':
    dataset = datasets.ImageFolder(root='/content/drive/MyDrive/datasets/dataset', transform=transform)
else:
    dataset = datasets.ImageFolder(root='/Users/lucasyoung/Desktop/Art/Stock/txt2img-images/2024-06-22', transform=transform)
print("Dataset collected")

print("Loading dataset...")
loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
print("Dataset loaded")

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
print(f"Adam compiled")


# --------------------------------------------------------------------------------------------------
# ---------------------------------------- Memory Profiling ----------------------------------------
# --------------------------------------------------------------------------------------------------
def print_memory_usage(tag: str):

    if device.type == 'cuda':
        print(f"[{tag}] Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"[{tag}] Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    else:
        print(f"[{tag}] Allocated: {torch.mps.current_allocated_memory() / 1024 ** 3:.2f} GB")
        print(f"[{tag}] Reserved: {torch.mps.current_allocated_memory() / 1024 ** 3:.2f} GB")


# --------------------------------------------------------------------------------------------------
# --------------------------------------------- Train ----------------------------------------------
# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for epoch in range(epochs):

        print(f"Beginning epoch {epoch + 1}")

        running_loss = 0.0

        for index, (batch, label) in enumerate(loader):

            print(f"Beginning batch {index + 1}")

            batch = batch.to(device)
            optimizer.zero_grad()

            if device.type == 'cuda':

                with autocast():

                    encoded = encoder(batch)
                    print_memory_usage('Encoded')

                    flattened = flattener(encoded)
                    print_memory_usage('Flattened')

                    resampled, mu, var = resampler(flattened)
                    print_memory_usage('Resampled')

                    projected = projector(resampled)
                    print_memory_usage('Projected')

                    decoded = decoder(projected)
                    print_memory_usage('Decoded')

                    recon_loss = criterion(decoded, batch)
                    kl_loss = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())  # KL Divergence
                    loss = recon_loss + kl_loss
                    running_loss += loss.item()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                encoded = encoder(batch)
                print_memory_usage('Encoded')

                flattened = flattener(encoded)
                print_memory_usage('Flattened')

                resampled, mu, var = resampler(flattened)
                print_memory_usage('Resampled')

                projected = projector(resampled)
                print_memory_usage('Projected')

                decoded = decoder(projected)
                print_memory_usage('Decoded')

                recon_loss = criterion(decoded, batch)
                kl_loss = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())  # KL Divergence
                loss = recon_loss + kl_loss
                running_loss += loss.item()
                print_memory_usage('Propagated')

                loss.backward()
                optimizer.step()

            print(f'[{index + 1}] Loss: {loss.item():.4f}')

            for image in range(batch.size(0)):

                source_image = reversals(batch[image].cpu())
                target_image = reversals(decoded[image].cpu())
                merged_image = Image.new('RGB', (2048, 1024))

                merged_image.paste(source_image, (0, 0))
                merged_image.paste(target_image, (1024, 0))

                filename = f'{epoch:06}-{index:06}-{image:06}-{uuid.uuid4().hex}.png'

                if device.type == 'cuda':
                    merged_image.save(f'/content/drive/MyDrive/Output/{bid}/{filename}')

                else:
                    merged_image.save(f'/Users/lucasyoung/Desktop/Art/Output/{bid}/{filename}')

        running_loss = 0