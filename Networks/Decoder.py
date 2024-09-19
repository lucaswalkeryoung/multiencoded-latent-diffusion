# --------------------------------------------------------------------------------------------------
# ---------------------------------- MELD :: Base Decoder Network ----------------------------------
# --------------------------------------------------------------------------------------------------
import torch.nn as nn
import torch


# --------------------------------------------------------------------------------------------------
# ---------------------------------------- CLASS :: Decoder ----------------------------------------
# --------------------------------------------------------------------------------------------------
class Decoder(nn.Module):

    # ----------------------------------------------------------------------------------------------
    # --------------------------------- CONSTRUCTOR :: Constructor ---------------------------------
    # ----------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        super(Decoder, self).__init__()

        self.relu00 = nn.ReLU(inplace=True)

        self.pool01 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv01 = nn.ConvTranspose2d(in_channels=1024, out_channels=896, kernel_size=3, padding=1)
        self.norm01 = nn.BatchNorm2d(num_features=896)

        self.conv02 = nn.ConvTranspose2d(in_channels=896, out_channels=768, kernel_size=3, padding=1)
        self.norm02 = nn.BatchNorm2d(num_features=768)

        self.pool02 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv03 = nn.ConvTranspose2d(in_channels=768, out_channels=640, kernel_size=3, padding=1)
        self.norm03 = nn.BatchNorm2d(num_features=640)

        self.conv04 = nn.ConvTranspose2d(in_channels=640, out_channels=512, kernel_size=3, padding=1)
        self.norm04 = nn.BatchNorm2d(num_features=512)

        self.pool03 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv05 = nn.ConvTranspose2d(in_channels=512, out_channels=384, kernel_size=3, padding=1)
        self.norm05 = nn.BatchNorm2d(num_features=384)

        self.conv06 = nn.ConvTranspose2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.norm06 = nn.BatchNorm2d(num_features=256)

        self.pool04 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv07 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.norm07 = nn.BatchNorm2d(num_features=128)

        self.conv08 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, padding=1)


    # ----------------------------------------------------------------------------------------------
    # ------------------------------- METHOD :: Forward Propagation --------------------------------
    # ----------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pool01(x)
        x = self.relu00(self.norm01(self.conv01(x)))
        x = self.relu00(self.norm02(self.conv02(x)))

        x = self.pool02(x)
        x = self.relu00(self.norm03(self.conv03(x)))
        x = self.relu00(self.norm04(self.conv04(x)))

        x = self.pool03(x)
        x = self.relu00(self.norm05(self.conv05(x)))
        x = self.relu00(self.norm06(self.conv06(x)))

        x = self.pool04(x)
        x = self.relu00(self.norm07(self.conv07(x)))

        return self.conv08(x)