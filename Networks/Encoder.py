# --------------------------------------------------------------------------------------------------
# ---------------------------------- MELD :: Base Encoder Network ----------------------------------
# --------------------------------------------------------------------------------------------------
import torch.nn as nn
import torch


# --------------------------------------------------------------------------------------------------
# ---------------------------------------- CLASS :: Encoder ----------------------------------------
# --------------------------------------------------------------------------------------------------
class Encoder(nn.Module):

    # ----------------------------------------------------------------------------------------------
    # --------------------------------- CONSTRUCTOR :: Constructor ---------------------------------
    # ----------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        super(Encoder, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2)

        self.conv01 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.norm01 = nn.BatchNorm2d(num_features=8)

        self.conv02 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.norm02 = nn.BatchNorm2d(num_features=16)

        self.conv03 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.norm03 = nn.BatchNorm2d(num_features=32)

        self.conv04 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.norm04 = nn.BatchNorm2d(num_features=64)

        self.conv05 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.norm05 = nn.BatchNorm2d(num_features=128)

        self.conv06 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.norm06 = nn.BatchNorm2d(num_features=256)

        self.conv07 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.norm07 = nn.BatchNorm2d(num_features=512)

        self.conv08 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.norm08 = nn.BatchNorm2d(num_features=1024)


    # ----------------------------------------------------------------------------------------------
    # ------------------------------- METHOD :: Forward Propagation --------------------------------
    # ----------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pool(self.relu(self.norm01(self.conv01(x))))
        x = self.pool(self.relu(self.norm02(self.conv02(x))))

        x = self.pool(self.relu(self.norm03(self.conv03(x))))
        x = self.pool(self.relu(self.norm04(self.conv04(x))))

        x = self.pool(self.relu(self.norm05(self.conv05(x))))
        x = self.pool(self.relu(self.norm06(self.conv06(x))))

        x = self.pool(self.relu(self.norm07(self.conv07(x))))
        x = self.pool(self.relu(self.norm08(self.conv08(x))))

        return x