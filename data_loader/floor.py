from PIL import Image
import os
import torch
import numpy as np
import torch.utils.data as data


class FloorDataset(data.Dataset):

    NUM_CLASS = 2

    def __init__(
        self,
        root="./datasets/floor",
        split="train",
        mode=None,
        transform=None,
        **kwargs
    ):
        super(FloorDataset, self).__init__()

        self.root = root
        self.split = split
        self.mode = mode if mode is not None else split
        self.transform = transform

        self.image_dir = os.path.join(root, split, "images")
        self.mask_dir = os.path.join(root, split, "masks")

        if not os.path.exists(self.image_dir):
            raise RuntimeError(f"Image folder not found: {self.image_dir}")

        if not os.path.exists(self.mask_dir):
            raise RuntimeError(f"Mask folder not found: {self.mask_dir}")

        self.images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_name = self.images[index]

        image_path = os.path.join(self.image_dir, image_name)

        base_name = os.path.splitext(image_name)[0]

        mask_path = os.path.join(
            self.mask_dir,
            base_name + ".png"
        )

        img = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform is not None:
            img = self.transform(img)

        mask = np.array(mask, dtype=np.uint8)

        # Convert to binary labels
        mask = (mask > 0).astype(np.int64)

        mask = torch.from_numpy(mask).long()

        return img, mask

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0