import os
import cv2
import numpy as np

def load_dataset(lr_dir, hr_dir, limit=None, patch_size=64, stride=64):
    lr_images, hr_images = [], []
    files = sorted(os.listdir(lr_dir))

    if limit:
        files = files[:limit]   # only first N files

    for file in files:
        lr = cv2.imread(os.path.join(lr_dir, file))
        hr = cv2.imread(os.path.join(hr_dir, file))

        if lr is None or hr is None:
            continue

        h, w = hr.shape[:2]
        lr_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)

        lr_up = lr_up.astype(np.float32) / 255.0
        hr = hr.astype(np.float32) / 255.0

        # crop into small patches instead of keeping whole 2K image
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                lr_patch = lr_up[i:i+patch_size, j:j+patch_size, :]
                hr_patch = hr[i:i+patch_size, j:j+patch_size, :]
                lr_images.append(lr_patch)
                hr_images.append(hr_patch)

    return np.array(lr_images), np.array(hr_images)
