import numpy as np
import os
from template_matching import template_matching
from load_image import load_n_process_image
from transformations import blur, add_noise, mixing_channels, rgb_to_pca, rgb_to_ycbcr


# A list of settings to run the template matching experiment
# _______________________________________________________________

# One of subfolders in images/templates folder. Right now either synthethic_blur_experiment or real_blur_experiment.
# Or you can create your own with your images and template positions.
subfolder = 'synthetic_blur_experiment'
# Name of a sharp image (located in the images folder) in which we want to match templates
full_img_name = "sharp01.JPG"
# sharp image name without extension
img_name = os.path.splitext(full_img_name)[0]
# Multiple by which the sharp and blurred images will be downsampled
downscale = 2
# Either "square", "square_padded", "T", "T_padded", "triangle", "triangle_padded", "disk", "disk_padded",
# or name of an image that is a blurred version of full_img_name
blur_type = "square"
# the size of blurring kernel. It has impact only if blur_type is "square" or "T", not separate image
size_of_blur = 33
# Path to npy file with upper left corners (in pixels) of selected templates (templates are in rows,
# the height is in the first column, the width is in the second column).
selected_templates = img_name + "_100temp_positions.npy"
# We will load the left upper corners of templates. The user chooses template size in pixels,
# but if it extends outside the image size, the template is not selected
temp_sz = 98
# Right now, the options are "N?fold_invs", "RGB_crosscorr", "gray_croscorr". The ? in N?fold_invs corresponds to N-fold
# symmetry assumption that we put on the blurring kernel. If ? = 1, then unconstrained blur is assumed.
method = 'N2fold_invs'
# invs_comb has an effect only if croscorr is not selected for invs_type. It indicates the invariants I_jk used for
# the template matching experiment. One digit refers to single-channel invs, two digits denote cross-channel invariants,
# (['gray']) denotes invariants computed from intensities
invs_comb = ('0', '1', '2', '10', '21')
# the maximum order of moments (does not have an effect for cross-correlation)
order = 5
# If True, complex moments are used. Otherwise, central geometric moments are used. We must select complex moments if
# N-fold symmetry of blur kernel is assumed and N > 2. (It does not have an effect for cross-correlation)
complex = False
# If true, then gravity center is computed as mean of gravity centers of image channels
# (It does not have an effect for cross-correlation)
one_center = False
# If blur invariants are used as method, then the detected position is chosen as the minimum norm of relative errors
# of the invariants. If norm=n, then Ln norm is used. By experiments, L1 usually comes out as the best.
norm = 1
# In case of real blur, we normalize the mean of templates. (It does not have an effect for cross-correlation)
temp_normalization = False
# Normalizing invariants to similar scale when 1. In the paper, it was used only for template matching with
# computer-generated blur as it slightly improved results. (It does not have an effect for cross-correlation)
typennum = 0
# Computing image invariants in all patches is time-demanding. If you have already computed them for the selected
# sharp image, place it to the computed_invs folder and assign full name of the npy file to image_invariants.
# If you want to save computed invariants after this run, assign 'save' to image_invariants. If neither of these two,
# assign empty string ''
img_invariants = 'save'
# std of Gaussian noise to add to the blurred image. If sigma = 0, then no noise is added.
sigma = 0
# If True, then invariants for channel-mixing and blur are used. Otherwise, blur invariants are used
channel_mixing = True
# Matrix for channel mixing. If channel_mixing==False, the A has no effect.
a = 0.1
A = np.array([[1-2*a, a, a],[a, 1-2*a, a],[a, a, 1-2*a]])
# Orders of moments to construct Cs in blur channel mixing invariants. It has no effect if channel_mixing==False.
C_indices = [[(0,0)], [(1,0)], [(1,0)], [(0,1)]]

# _______________________________________________________________

if complex:
    moment_type = 'complex'
else:
    moment_type = 'geometric'

print(f"Setting: image downscale: {downscale}x, blur type: {blur_type}, size of blur kernel: {size_of_blur}, "
    f"method: {method},", f"max order of moments: {order}, template size: {temp_sz},"
                          f" invariant combinations: {invs_comb},\n",
    f" Templates are normalized to mean: {temp_normalization},",
    f"moment type: {moment_type},", f"typen is {typennum},",
    f"one center of gravity: {one_center},", f"Gaussian noise sigma: {sigma},", f"Norm used to detection: L{norm}"
    f"channel mixing: {channel_mixing}, A: {A}")

img_sharp = load_n_process_image(full_img_name, subfolder, downscale, method, invs_comb)

if '.' in blur_type:
    img_transformed = load_n_process_image(blur_type, subfolder, downscale, method, invs_comb)
elif 'padded' in blur_type:
    img_transformed = np.copy(img_sharp)
else:
    img_transformed = blur(img_sharp, size=size_of_blur, btype=blur_type)
    if sigma > 0:
        img_transformed = add_noise(img_transformed, sigma)

img_transformed = mixing_channels(img_transformed, A)
# img_transformed = rgb_to_pca(img_transformed)
# img_transformed = rgb_to_ycbcr(img_transformed)

print(f"Shape of the sharp image {full_img_name}:", img_sharp.shape)

temp_pos = np.load(os.path.join("templates", subfolder, selected_templates))
print(f"Templates were loaded from {selected_templates}")

detect_pos = template_matching(img_sharp, img_transformed, temp_pos, temp_sz, order, complex, img_name, moment_type,
                                method, invs_comb, C_indices, img_invariants, size_of_blur, blur_type,
                                temp_normalization, typennum, subfolder, one_center, norm, channel_mixing, A, sigma)


# preparing name of a file (that we are going to save) containing detected positions, distance from true positions
# and relative errors of invariants
if "." in blur_type:
    image_n_blur = img_name + f"_{os.path.splitext(blur_type)[0]}blur_temp_sz_{temp_sz:03d}_PCA"
else:
    image_n_blur = img_name + f"_{blur_type}blur_size{size_of_blur:02d}"

if 'crosscorr' in method:
    det_positions_file_name = (f'DP_{image_n_blur}_{method}_temp_sz_{temp_sz}'
                               f'_BtempSimg.npy').replace(' ', '')
elif channel_mixing:
    det_positions_file_name = (
        f'DP_{image_n_blur}_a_{a:.1f}_noise{sigma:.2f}_{method}_{moment_type}_r_{order}_j1_{C_indices[0][0]}{C_indices[1][0]}_j2_{C_indices[2][0]}{C_indices[3][0]}_'
        f'temp_sz_{temp_sz}_tempnorm{temp_normalization}_BtempSimg.npy')
else:
    det_positions_file_name = (
        f'DP_{image_n_blur}_a_{a:.1f}_noise{sigma:.2f}_{method}_{moment_type}_{invs_comb}_r_{order}_'
        f'typen{typennum}_tempnorm{temp_normalization}_BtempSimg.npy').replace(' ', '')
if channel_mixing:
    folder = 'blur_n_channelmixing_invariants'
else:
    folder = 'blur_invariants'
save_path = f'{folder}/detected_positions/{subfolder}/square_blur_new_vs_old_invariants/{img_name}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.save(f'{save_path}/{det_positions_file_name}'.replace(" ", ""),
        detect_pos)
print(detect_pos)

