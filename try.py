

import torch

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision import io

# # Create dummy target image
# nb_classes = 19 - 1 # 18 classes + background
# idx = np.linspace(0., 1., nb_classes)
# cmap = matplotlib.cm.get_cmap('viridis')
# rgb = cmap(idx, bytes=True)[:, :3]  # Remove alpha value

# h, w = 190, 100
# rgb = rgb.repeat(1000, 0)
# target = np.zeros((h*w, 3), dtype=np.uint8)
# target[:rgb.shape[0]] = rgb
# target = target.reshape(h, w, 3)
target = io.read_image('/Users/marco/zaneta-tiles-pos0_02/train/masks/PAS_I_4_S_6_ROI_5_6.png')
target = target.permute(1, 2, 0).numpy()
plt.imshow(target) # Each class in 10 rows
H, W, C = target.shape
print(target.shape)
# Create mapping
# Get color codes for dataset (maybe you would have to use more than a single
# image, if it doesn't contain all classes)
target = torch.from_numpy(target)
colors = torch.unique(target.view(-1, target.size(2)), dim=0).numpy()
print(colors)
target = target.permute(2, 0, 1).contiguous()
print(target.shape)
mapping = {(0, 0, 0): 0, (0, 255, 0): 1, (255, 0, 0): 2}
# mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}
print(f"mapping:{mapping}")

mask = torch.empty(H, W, dtype=torch.long)
for k in mapping:
    # Get all indices for current class
    idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
    validx = (idx.sum(0) == 3)  # Check that all channels match
    mask[validx] = torch.tensor(mapping[k], dtype=torch.long)

mask = mask.unsqueeze(0)
print(mask.shape)