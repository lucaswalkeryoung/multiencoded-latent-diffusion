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

        self.relu00 = nn.ReLU(inplace=True)

        self.conv01 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1)
        self.norm01 = nn.BatchNorm2d(num_features=128)

        self.conv02 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.norm02 = nn.BatchNorm2d(num_features=256)

        self.pool01 = nn.AvgPool2d(kernel_size=2)

        self.conv03 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.norm03 = nn.BatchNorm2d(num_features=384)

        self.conv04 = nn.Conv2d(in_channels=384, out_channels=512, kernel_size=3, padding=1)
        self.norm04 = nn.BatchNorm2d(num_features=512)

        self.pool02 = nn.AvgPool2d(kernel_size=2)

        self.conv05 = nn.Conv2d(in_channels=512, out_channels=640, kernel_size=3, padding=1)
        self.norm05 = nn.BatchNorm2d(num_features=640)

        self.conv06 = nn.Conv2d(in_channels=640, out_channels=768, kernel_size=3, padding=1)
        self.norm06 = nn.BatchNorm2d(num_features=768)

        self.pool03 = nn.AvgPool2d(kernel_size=2)

        self.conv07 = nn.Conv2d(in_channels=768, out_channels=896, kernel_size=3, padding=1)
        self.norm07 = nn.BatchNorm2d(num_features=896)

        self.conv08 = nn.Conv2d(in_channels=896, out_channels=1024, kernel_size=3, padding=1)
        self.norm08 = nn.BatchNorm2d(num_features=1024)

        self.pool04 = nn.AvgPool2d(kernel_size=2)


    # ----------------------------------------------------------------------------------------------
    # ------------------------------- METHOD :: Forward Propagation --------------------------------
    # ----------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.relu00(self.norm01(self.conv01(x)))
        x = self.relu00(self.norm02(self.conv02(x)))
        x = self.pool01(x)

        x = self.relu00(self.norm03(self.conv03(x)))
        x = self.relu00(self.norm04(self.conv04(x)))
        x = self.pool02(x)

        x = self.relu00(self.norm05(self.conv05(x)))
        x = self.relu00(self.norm06(self.conv06(x)))
        x = self.pool03(x)

        x = self.relu00(self.norm07(self.conv07(x)))
        x = self.relu00(self.norm08(self.conv08(x)))
        x = self.pool04(x)

        return x