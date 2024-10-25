import rawpy
import numpy as np
import cv2
from skimage import img_as_float
import os

def raw_to_rgb(raw):
    """
    Taking raw data (Bayer pattern) and producing RGB image with half of the resolution of the Bayer pattern.
    We slide 2x2 window and always take the red, the blue and the average of the two green elements to
    form one RGB pixel. This is not suitable for visualization, but is fine for computation purposes.
    """
    img = raw.raw_image_visible.copy()

    img = img_as_float(img)

    img_r = img[1:img.shape[0]:2, 0:img.shape[1]:2]
    img_g1 = img[1:img.shape[0]:2, 1:img.shape[1]:2]
    img_g2 = img[0:img.shape[0]:2, 0:img.shape[1]:2]
    img_b = img[0:img.shape[0]:2, 1:img.shape[1]:2]
    img_g = (img_g1 + img_g2) / 2
    rgb_img = cv2.merge([img_r, img_g, img_b])

    return rgb_img


def load_n_process_image(full_img_name, subfolder, downscale, method, invs_comb):
    """
    Loads and process image with chosen settings
    """
    if "CR2" in full_img_name:
        raw = rawpy.imread(os.path.join("images", subfolder, full_img_name))
        image = raw_to_rgb(raw)
        if (invs_comb == (['gray']) and 'crosscorr' not in method) or method == 'gray_croscorr':
            image = np.mean(image, axis=2)
    else:
        if (invs_comb == (['gray']) and 'crosscorr' not in method) or method == 'gray_crosscorr':
            image = cv2.imread(os.path.join("images", subfolder, full_img_name), flags=cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(os.path.join("images", subfolder, full_img_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (0, 0), fx=1/downscale, fy=1/downscale)
        image = img_as_float(image)

    return image
