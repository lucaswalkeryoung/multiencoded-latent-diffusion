# --------------------------------------------------------------------------------------------------
# --------------------------------- MELD :: Dense-Layer Projector ----------------------------------
# --------------------------------------------------------------------------------------------------
import torch.nn as nn
import torch


# --------------------------------------------------------------------------------------------------
# --------------------------------- CLASS :: Dense-Layer Projector ---------------------------------
# --------------------------------------------------------------------------------------------------
class Projector(nn.Module):

    # ----------------------------------------------------------------------------------------------
    # -------------------------------------- CLASS :: Encoder --------------------------------------
    # ----------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        super(Projector, self).__init__()

        projected = 8 * 8 * 1024 # 65536

        num00 = int(projected / (2 ** 4)) # 4096
        num01 = int(projected / (2 ** 3)) # 8096
        num02 = int(projected / (2 ** 2)) # 16384
        num03 = int(projected / (2 ** 1)) # 32768
        num04 = int(projected)

        self.flat01  = nn.Linear(num00, num01)
        self.flat02  = nn.Linear(num01, num02)
        self.flat03  = nn.Linear(num02, num03)
        self.flat04  = nn.Linear(num03, num04)

        self.relu00 = nn.ReLU(inplace=True)


    # ----------------------------------------------------------------------------------------------
    # ------------------------------- METHOD :: Forward Propagation --------------------------------
    # ----------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.relu00(self.flat01(x))
        x = self.relu00(self.flat02(x))
        x = self.relu00(self.flat03(x))
        x = self.flat04(x)

        x = x.view(-1, 1024, 128, 128)

        return x