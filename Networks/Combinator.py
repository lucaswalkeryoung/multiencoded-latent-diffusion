# --------------------------------------------------------------------------------------------------
# ----------------------------------- MELD :: Encoder Combinator -----------------------------------
# --------------------------------------------------------------------------------------------------
import torch.nn as nn
import torch


# --------------------------------------------------------------------------------------------------
# --------------------------------------- CLASS :: Combinator --------------------------------------
# --------------------------------------------------------------------------------------------------
class Combinator(nn.Module):

    # ----------------------------------------------------------------------------------------------
    # -------------------------------------- CLASS :: Encoder --------------------------------------
    # ----------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        super(Combinator, self).__init__()

        self.relu00 = nn.ReLU(inplace=True)

        self.conv01 = nn.Conv2d(in_channels=4096, out_channels=3968, kernel_size=3, padding=1)
        self.norm01 = nn.BatchNorm2d(num_features=3968)

        self.conv02 = nn.Conv2d(in_channels=3968, out_channels=3840, kernel_size=3, padding=1)
        self.norm02 = nn.BatchNorm2d(num_features=3840)

        self.conv03 = nn.Conv2d(in_channels=3840, out_channels=3712, kernel_size=3, padding=1)
        self.norm03 = nn.BatchNorm2d(num_features=3712)

        self.conv04 = nn.Conv2d(in_channels=3712, out_channels=3584, kernel_size=3, padding=1)
        self.norm04 = nn.BatchNorm2d(num_features=3584)

        self.conv05 = nn.Conv2d(in_channels=3584, out_channels=3456, kernel_size=3, padding=1)
        self.norm05 = nn.BatchNorm2d(num_features=3456)

        self.conv06 = nn.Conv2d(in_channels=3456, out_channels=3328, kernel_size=3, padding=1)
        self.norm06 = nn.BatchNorm2d(num_features=3328)

        self.conv07 = nn.Conv2d(in_channels=3328, out_channels=3200, kernel_size=3, padding=1)
        self.norm07 = nn.BatchNorm2d(num_features=3200)

        self.conv08 = nn.Conv2d(in_channels=3200, out_channels=3072, kernel_size=3, padding=1)
        self.norm08 = nn.BatchNorm2d(num_features=3072)

        self.conv09 = nn.Conv2d(in_channels=3072, out_channels=2944, kernel_size=3, padding=1)
        self.norm09 = nn.BatchNorm2d(num_features=2944)

        self.conv10 = nn.Conv2d(in_channels=2944, out_channels=2816, kernel_size=3, padding=1)
        self.norm10 = nn.BatchNorm2d(num_features=2816)

        self.conv11 = nn.Conv2d(in_channels=2816, out_channels=2688, kernel_size=3, padding=1)
        self.norm11 = nn.BatchNorm2d(num_features=2688)

        self.conv12 = nn.Conv2d(in_channels=2688, out_channels=2560, kernel_size=3, padding=1)
        self.norm12 = nn.BatchNorm2d(num_features=2560)

        self.conv13 = nn.Conv2d(in_channels=2560, out_channels=2432, kernel_size=3, padding=1)
        self.norm13 = nn.BatchNorm2d(num_features=2432)

        self.conv14 = nn.Conv2d(in_channels=2432, out_channels=2304, kernel_size=3, padding=1)
        self.norm14 = nn.BatchNorm2d(num_features=2304)

        self.conv15 = nn.Conv2d(in_channels=2304, out_channels=2176, kernel_size=3, padding=1)
        self.norm15 = nn.BatchNorm2d(num_features=2176)

        self.conv16 = nn.Conv2d(in_channels=2176, out_channels=2048, kernel_size=3, padding=1)
        self.norm16 = nn.BatchNorm2d(num_features=2048)


    # --------------------------------------------------------------------------------------------------
    # ------------------------------- METHOD :: Forward Propagation -----------------------------------
    # --------------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.relu00(self.norm01(self.conv01(x)))
        x = self.relu00(self.norm02(self.conv02(x)))
        x = self.relu00(self.norm03(self.conv03(x)))
        x = self.relu00(self.norm04(self.conv04(x)))

        x = self.relu00(self.norm05(self.conv05(x)))
        x = self.relu00(self.norm06(self.conv06(x)))
        x = self.relu00(self.norm07(self.conv07(x)))
        x = self.relu00(self.norm08(self.conv08(x)))

        x = self.relu00(self.norm09(self.conv09(x)))
        x = self.relu00(self.norm10(self.conv10(x)))
        x = self.relu00(self.norm11(self.conv11(x)))
        x = self.relu00(self.norm12(self.conv12(x)))

        x = self.relu00(self.norm13(self.conv13(x)))
        x = self.relu00(self.norm14(self.conv14(x)))
        x = self.relu00(self.norm15(self.conv15(x)))
        x = self.relu00(self.norm16(self.conv16(x)))

        return x