from model import build_SRCNN
from dataset import load_dataset
from utils import psnr , ssim
import tensorflow as tf

lr_train='/content/drive/MyDrive/SUPER RESOLUTION/DIV2K_train_LR_bicubic_X2'
hr_train='/content/drive/MyDrive/SUPER RESOLUTION/DIV2K_train_HR/DIV2K_train_HR'

lr_val= '/content/drive/MyDrive/SUPER RESOLUTION/DIV2K_valid_LR_bicubic_X2'
hr_val= '/content/drive/MyDrive/SUPER RESOLUTION/DIV2K_val_HR/DIV2K_valid_HR'

#load datasets
print('hi')
x_train, y_train = load_dataset(lr_train, hr_train, limit=100, patch_size=64, stride=128)
print(x_train.shape, y_train.shape)

print('hello')
x_val,y_val=load_dataset(lr_val,hr_val,limit=20, patch_size=64, stride=128)

print("Training set:", x_train.shape, y_train.shape)
print("Validation set:", x_val.shape, y_val.shape)
#evaluation metrics during training

def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


#build model

with tf.device('/GPU:0'):

 model=build_SRCNN()
 model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse',metrics=[psnr_metric, ssim_metric])

#training loop
print('traning started')
history= model.fit(x_train,y_train,
                   validation_data=(x_val,y_val),
                   batch_size=16,
                   epochs=50)
print('traning completed')


# Save model
model.save("/content/drive/MyDrive/SUPER RESOLUTION/SRCNN/srcnn.h5")

