# --------------------------------------------------------------------------------------------------
# ---------------- MELD :: Transform Class to Resize Min-Dimension to 1024 and Crop ----------------
# --------------------------------------------------------------------------------------------------
from torchvision import transforms
from PIL import Image


# --------------------------------------------------------------------------------------------------
# ---------------------------- CLASS :: Random Resize & Crop Transform -----------------------------
# --------------------------------------------------------------------------------------------------
class RCrop(object):

    # ------------------------------------------------------------------------------------------
    # ----------------------------- OPERATOR :: Call the Transform -----------------------------
    # ------------------------------------------------------------------------------------------
    def __call__(self, image: Image.Image) -> Image.Image:

        width, height = image.size

        if min(width, height) < 1024:

            if width < height:
                new_width  = 1024
                new_height = int(height * (1024 / width))
            else:
                new_height = 1024
                new_width  = int(width * (1024 / height))

            image = image.resize((new_width, new_height), Image.LANCZOS)

        return transforms.RandomCrop(1024)(image)