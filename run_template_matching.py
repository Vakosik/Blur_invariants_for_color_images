import numpy as np
import os
from template_matching import template_matching
from load_image import load_n_process_image
from blurs import blur


# A list of settings to run the template matching experiment
# _______________________________________________________________

# Name of a sharp image (located in the image folder) in which we want to match templates
full_img_name = "sharp1.JPG"
# sharp image name without extension
img_name = os.path.splitext(full_img_name)[0]
# Multiple by which the sharp and blurred images will be downsampled
downscale = 2
# Either "square", "T", "square_padded, "T_padded" or name of an image that is a blurred version of full_img_name
blur_type = "IMG_5124.CR2"
# the size of blurring kernel. It has impact only if blur_type is "square" or "T", not separate image
size_of_blur = 0
# Path to npy file with upper left corners (in pixels) of selected templates (templates are in rows,
# the height is in the first column, the width is in the second column).
selected_templates = img_name + "100templates.npy"
# We will load the left upper corners of templates. The user chooses template size in pixels,
# but if it extends outside the image size, the template is not selected
temp_sz = 100
# Right now, the options are "centrosym_invs", "unconstrained_invs", "RGB_crosscorr", "gray_croscorr"
method = 'centrosym_invs'
# invs_comb has an effect only if croscorr is not selected for invs_type. It indicates the invariants I_jk used for
# the template matching experiment. One digit refers to single-channel invs, two digits denote cross-channel invariants,
# (['gray']) denotes invariants computed from intensities
invs_comb = ('0', '1', '2', '10', '21')
# the maximum order of moments (does not have an effect for cross-correlation)
order = 5
# In case of real blur, we normalize the mean of templates. (It does not have an effect for cross-correlation)
temp_normalization = True
# Normalizing invariants to similar scale when 1. In the paper, it was used only for template matching with
# computer-generated blur as it slightly improved results. (It does not have an effect for cross-correlation)
typennum = 0
# Computing image invariants in all patches is time-demanding. If you have already computed them for the selected
# sharp image, place it to the computed_invs folder and assign full name of the npy file to image_invariants.
# If you want to save computed invariants after this run, assign 'save' to image_invariants. If neither of these two,
# assign empty string ''
img_invariants = ''

# _______________________________________________________________


print(f"Setting: image downscale: {downscale}x, blur type: {blur_type}, size of blur kernel: {size_of_blur}, "
      f"max order of moments: {order}, template size: {temp_sz}," f" invariant combinations: {invs_comb},\n",
      f"Templates are normalized to mean: {temp_normalization}",
      f"typen is {typennum} \n")

img_sharp = load_n_process_image(full_img_name, downscale, method, invs_comb)

if '.' in blur_type:
    img_blurred = load_n_process_image(blur_type, downscale, method, invs_comb)
elif 'padded' in blur_type:
    img_blurred = np.copy(img_sharp)
else:
    img_blurred = blur(img_sharp, size=size_of_blur, btype=blur_type)

print(f"Shape of the sharp image {full_img_name}:", img_sharp.shape)

temp_pos = np.load(os.path.join("templates", selected_templates))
print(f"Templates were loaded from {selected_templates}")

detect_pos = template_matching(img_sharp, img_blurred, temp_pos, temp_sz, order, img_name, method, invs_comb,
                               img_invariants, size_of_blur, blur_type, temp_normalization, typennum)

if "." in blur_type:
    image_n_blur = img_name + f"_{os.path.splitext(blur_type)[0]}blur"
else:
    image_n_blur = img_name + f"_{blur_type}blur_size{size_of_blur:02d}"

if 'crosscorr' in method:
    det_positions_file_name = (f'detected_pos_{image_n_blur}_{method}_temp_sz_{temp_sz}'
                               f'_BtempSimg.npy').replace(' ', '')
else:
    det_positions_file_name = (f'detected_pos_{image_n_blur}_{method}_r_{order}_inv_comb_{invs_comb}_temp_sz_{temp_sz}_'
                               f'typen{typennum}_tempnorm{temp_normalization}_BtempSimg.npy').replace(' ', '')

np.save(f'detected_positions/{det_positions_file_name}'.replace(" ", ""), detect_pos)
print(detect_pos)