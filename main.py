# --------------------------------------------------------------------------------------------------
# ---------------------------------------- MELD Entry Point ----------------------------------------
# --------------------------------------------------------------------------------------------------
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

import torch.optim as optim
import torch.nn as nn
import torch

from Networks.Encoder import Encoder
from Networks.Decoder import Decoder
from Transforms.RCrop import RCrop
from Loaders.Loader   import Loader

import os


# --------------------------------------------------------------------------------------------------
# -------------------------------------- Configure the System --------------------------------------
# --------------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 1e-4
epochs        = 8
batch_size    = 2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --------------------------------------------------------------------------------------------------
# -------------------------------- Initialize the Models and Tools ---------------------------------
# --------------------------------------------------------------------------------------------------
encoder = Encoder()
encoder = encoder.to(device)
encoder.train(True)

decoder = Decoder()
decoder = decoder.to(device)
decoder.train(True)

criterion = nn.MSELoss()

dataset = datasets.ImageFolder(root='/content/drive/MyDrive/datasets/dataset', transform=transform)
loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)


# # --------------------------------------------------------------------------------------------------
# # --------------------------------------------- Train ----------------------------------------------
# # --------------------------------------------------------------------------------------------------
# if __name__ == '__main__':
#
#     for epoch in range(epochs):
#
#         for index, batch in enumerate(dataset):
#
#             batch = batch.to(device)
#             optimizer.zero_grad()
#
#             encoded = encoder(batch)
#             decoded = decoder(encoded)
#
#             loss = criterion(decoded, batch)
#             loss.backward()
#
#             optimizer.step()
#
#             print(f'[{index + 1}] Loss: {loss.item():.4f}')