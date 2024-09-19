# --------------------------------------------------------------------------------------------------
# --------------------------------- MELD :: Dense-Layer Flattener ----------------------------------
# --------------------------------------------------------------------------------------------------
import torch.nn as nn
import torch


# --------------------------------------------------------------------------------------------------
# --------------------------------- CLASS :: Dense-Layer Flattener ---------------------------------
# --------------------------------------------------------------------------------------------------
class Flattener(nn.Module):

    # ----------------------------------------------------------------------------------------------
    # -------------------------------------- CLASS :: Encoder --------------------------------------
    # ----------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        super(Flattener, self).__init__()

        flattened = 8 * 8 * 1024

        num00 = int(flattened)
        num01 = int(flattened / (4 ** 1))
        num02 = int(flattened / (4 ** 2))
        num03 = int(flattened / (4 ** 3))
        num04 = int(flattened / (4 ** 4))

        self.flatten = nn.Flatten()
        self.flat01  = nn.Linear(num00, num01)
        self.flat02  = nn.Linear(num01, num02)
        self.flat03  = nn.Linear(num02, num03)
        self.flat04  = nn.Linear(num03, num04)

        self.relu00 = nn.ReLU(inplace=True)


    # ----------------------------------------------------------------------------------------------
    # ------------------------------- METHOD :: Forward Propagation --------------------------------
    # ----------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.flatten(x)

        x = self.relu00(self.flat01(x))
        x = self.relu00(self.flat02(x))
        x = self.relu00(self.flat03(x))
        x = self.flat04(x)

        return x