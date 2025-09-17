

# test.py
from model import build_SRCNN
from dataset import load_dataset
from utils import psnr, ssim
import tensorflow as tf
import numpy as np
from utils import psnr, ssim 
# Paths
lr_test_dir = "/content/drive/MyDrive/SUPER RESOLUTION/DIV2K_valid_LR_bicubic_X2"
hr_test_dir = "/content/drive/MyDrive/SUPER RESOLUTION/DIV2K_val_HR/DIV2K_valid_HR"

# Load test dataset
x_test, y_test = load_dataset(lr_test_dir, hr_test_dir, limit=20)  # limit optional

print("Test set:", x_test.shape, y_test.shape)

# Load trained model
model = tf.keras.models.load_model(
    "/content/drive/MyDrive/SUPER RESOLUTION/SRCNN/srcnn.h5",
    custom_objects={
        "psnr_metric": psnr,
        "ssim_metric": ssim,
        "mse": tf.keras.losses.MeanSquaredError()
    }
)

# Run predictions
y_pred = model.predict(x_test)
y_pred = np.clip(y_pred, 0, 1)

# Compute average PSNR & SSIM
psnr_scores = [psnr(y_test[i], y_pred[i]) for i in range(len(y_test))]
ssim_scores = [ssim(y_test[i], y_pred[i]) for i in range(len(y_test))]

print("Average PSNR on test set:", np.mean(psnr_scores))
print("Average SSIM on test set:", np.mean(ssim_scores))

