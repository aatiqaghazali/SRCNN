import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def psnr(img1,img2):
  mse=np.mean((img1-img2) ** 2)
  if (mse==0):
    return 100

  PIXEL_MAX=1.00    # since values normalized to [0,1] if not then 255 here will be used

  return 20* np.log10(PIXEL_MAX/np.sqrt(mse))




def ssim(img1 , img2):

  # You can use skimage for SSIM (simpler)
    from skimage.metrics import structural_similarity as ssim_metric
    img1_gray = np.mean(img1, axis=2) if img1.ndim == 3 else img1
    img2_gray = np.mean(img2, axis=2) if img2.ndim == 3 else img2
    return ssim_metric(img1_gray, img2_gray, data_range=1.0)



def resize_bicubic(image,scale=2):

  h,w=image.shape[:2]

  return cv2.resize(image,(w*scale , h*scale), interpolation=cv2.INTER_CUBIC)






