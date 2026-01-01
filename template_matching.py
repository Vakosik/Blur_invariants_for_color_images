import numpy as np
from blur_invariants.assign_blur_invariants import compute_img_n_temp_blur_invariants
from blur_n_channelmixing_invariants.assign_blur_channelmix_invariants import compute_img_n_temp_blurchannelmix_invariants
from detect_position import correlation, compare_invariants
from transformations import template_padding_blur
from joblib import Parallel, delayed
from tqdm import tqdm
from transformations import nearest_multiple_of_14


def template_matching(img, img_temp, temp_pos, temp_sz, order, complex, img_name, moment_type, method='centrosym_invs',
                      invs_comb=('0', '1', '2', '10', '21'), C_indices=[[()], [()]], img_invariants='', size_of_blur=21, blur_type="square",
                      temp_normalization=False, typennum=0, subfolder='', one_center=True, norm=1, channel_mixing=False,
                      A=np.identity(3), sigma=0):
    """
    Search for the templates in image img and return the best match
    Input arguments
    ---
      img      ... image, where we are looking for the templates
      temp_pos ... Nx2 matrix of positions (left top corner)
      temp_sz  ... size of the template (side of the square) in pixels
      order    ... max order of the invariants
      method   ... type of method for the template matching
                   "centrosym_invs" | "unconstrained_invs" | "RGB_crosscorr" | "gray_croscorr"
      fname    ... image name
      img_temp ... from this image we take the templates

    Output arguments
    ---
      detected_pos ... array of vectors in the form
                      [detected_row, detected_col, normRE, L2_distance]

    """

    n_temp = temp_pos.shape[0]

    temps = []
    remove_excessive_temps = []
    for k in range(n_temp):
        temp_pos[k] = nearest_multiple_of_14(temp_pos[k][0], temp_pos[k][1])
        i, j = temp_pos[k]
        template = img_temp[i:i + temp_sz, j:j + temp_sz].copy()
        if temp_normalization and 'crosscorr' not in method:
            template /= np.mean(template)

        if template.shape[0] == temp_sz and template.shape[1] == temp_sz:
            if 'padded' in blur_type:
                template = template_padding_blur(template, size_of_blur, blur_type)
            temps.append(template)
        else:
            remove_excessive_temps.append(k)

    temp_pos = np.delete(temp_pos, remove_excessive_temps, axis=0)
    temps = np.array(temps)
    n_temp = temps.shape[0]

    if 'crosscorr' not in method:
        if channel_mixing:
            image_invs, temp_invs = compute_img_n_temp_blurchannelmix_invariants(img, temps, temp_sz, order, complex,
                                                                                 img_name, method, C_indices, img_invariants,
                                                                                 temp_normalization, subfolder,
                                                                                 moment_type, one_center)
        else:
            image_invs, temp_invs = compute_img_n_temp_blur_invariants(img, temps, temp_sz, order, complex, img_name,
                                                                method, invs_comb, img_invariants, temp_normalization,
                                                                typennum, subfolder, moment_type, one_center)

        del temps, img

    print("Detecting positions of templates by comparing the invariants or taking the maximum of cross-correlation...")

    def compute_positions(z):
        if 'crosscorr' in method:
            ypos, xpos, NRE = correlation(temps[z], img)
        else:
            ypos, xpos, NRE = compare_invariants(image_invs.copy(), temp_invs[z], norm)

        distance = np.linalg.norm(temp_pos[z] - [ypos, xpos])

        return [ypos, xpos, NRE, distance]

    # uses all cores in CPU
    detected_pos = Parallel(n_jobs=10)(delayed(compute_positions)(z) for z in tqdm(range(n_temp)))
    detected_pos = np.array(detected_pos)

    return detected_pos