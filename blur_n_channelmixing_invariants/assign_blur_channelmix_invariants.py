import numpy as np
from blur_n_channelmixing_invariants.blur_channelmixing_invariants import (blur_channel_mixing_invariants_2channels,
                                           blur_channel_mixing_invariants_3channels)
from blur_invariants.blur_invariants import central_moments, average_center
from blur_invariants.assign_blur_invariants import choose_moments
from joblib import Parallel, delayed
from tqdm import tqdm
import os


def compute_img_n_temp_blurchannelmix_invariants(img, temps, temp_sz, order, complex, img_name, method, C_indices,
                                                 img_invariants, temp_normalization, subfolder, moment_type, one_center):
    """
    Computes invariants in all patches in img and for all templates in temps. If img_invariants is a name of npy
    file with already computed invariants in all patches, then it just loads it. If img_invariants=='save' then it
    computes the invariants in all patches and saves them.
    """
    N = int(method[1])
    num_channels = img.shape[2]
    invs_count = invs_counter(num_channels, C_indices, order, N)

    print("number of moment invariants:", invs_count)

    path_img_invs = os.path.join('blur_n_channelmixing_invariants', 'computed_invs', subfolder, img_name)

    if img_invariants == '' or img_invariants == 'save':
        print("Computing invariants of the image in all patches...")
        img_invs = get_image_invariants(img, invs_count, order, complex, N, num_channels, C_indices, temp_sz,
                                        temp_normalization, one_center)
        if img_invariants == 'save':
            if not os.path.exists(path_img_invs):
                os.makedirs(path_img_invs)
            invs_name = os.path.join(path_img_invs, f'{img_name}_r_{order}_j1_{C_indices[0][0]}{C_indices[1][0]}_j2_'
                                                    f'{C_indices[2][0]}{C_indices[3][0]}_{method}_{moment_type}_'
                                                    f'inv_temp_sz_{temp_sz}_tempnorm{temp_normalization}'
                                                    f'_BtempSimg'
                                     .replace(" ", ""))
            np.save(f'{invs_name}.npy', img_invs)
    else:
        img_invs = np.load(os.path.join(path_img_invs, img_invariants))
        print(f"Image invariants loaded from {img_invariants}.")

    print("Computing invariants of the chosen templates...")
    temp_invs = get_template_invariants(temps, invs_count, order, complex, N, num_channels, C_indices, one_center)

    return img_invs, temp_invs


def get_image_invariants(img, invs_count, order, complex, N, num_channels, C_indices, temp_sz, temp_normalization, one_center):
    s = (np.array(img.shape) - temp_sz + 1).astype(int)
    img_invs = np.zeros((s[0], s[1], invs_count))

    def invariants_by_row(r):
        row_invariants = np.zeros_like(img_invs[r, :, :])

        for c in range(s[1]):
            segment = img[r:r + temp_sz, c:c + temp_sz].copy()
            if temp_normalization:
                segment /= np.mean(segment)

            if len(img.shape) > 2:
                if one_center:
                    tx, ty = average_center(segment)
                else:
                    tx, ty = segment.shape[0]/2, segment.shape[1]/2
                gm = np.zeros((order+1, order+1, 3))
                gm[:, :, 0] = central_moments(segment[:, :, 0], r=order, tx=tx, ty=ty)
                gm[:, :, 1] = central_moments(segment[:, :, 1], r=order, tx=tx, ty=ty)
                gm[:, :, 2] = central_moments(segment[:, :, 2], r=order, tx=tx, ty=ty)
            else:
                gm = central_moments(segment, r=order)

            moments = choose_moments(gm, order, complex)
            if complex:
                typex = 1
            else:
                typex = 0

            if num_channels == 2:
                row_invariants[c], _ = blur_channel_mixing_invariants_2channels(moments, C_indices, order, N, typec=1,
                                                                                typex=typex)
            else:
                row_invariants[c], _ = blur_channel_mixing_invariants_3channels(moments, C_indices, order, N, typec=1,
                                                                                typex=typex)

        return row_invariants

    # uses all cores in CPU
    invariants_by_rows = Parallel(n_jobs=48)(delayed(invariants_by_row)(r) for r in tqdm(range(s[0])))

    for r, row_result in enumerate(invariants_by_rows):
        img_invs[r, :, :] = row_result

    return img_invs


def get_template_invariants(temps, invs_count, order, complex, N, num_channels, C_indices, one_center):
    n_temp = len(temps)
    temp_invs = np.zeros((n_temp, invs_count))

    for temp in range(n_temp):
        if len(temps[temp].shape) > 2:
            if one_center:
                tx, ty = average_center(temps[temp])
            else:
                tx, ty = temps[temp].shape[0]/2, temps[temp].shape[1]/2
            gm = np.zeros((order + 1, order + 1, 3))
            gm[:, :, 0] = central_moments(temps[temp, :, :, 0], r=order, tx=tx, ty=ty)
            gm[:, :, 1] = central_moments(temps[temp, :, :, 1], r=order, tx=tx, ty=ty)
            gm[:, :, 2] = central_moments(temps[temp, :, :, 2], r=order, tx=tx, ty=ty)
        else:
            gm = central_moments(temps[temp], r=order)

        moments = choose_moments(gm, order, complex)
        if complex:
            typex = 1
        else:
            typex = 0

        if num_channels == 2:
            temp_invs[temp], _ = blur_channel_mixing_invariants_2channels(moments, C_indices, order, N, typec=1,
                                                                          typex=typex)
        else:
            temp_invs[temp], _ = blur_channel_mixing_invariants_3channels(moments, C_indices, order, N, typec=1,
                                                                          typex=typex)

    return temp_invs


def invs_counter(num_channels, C_indices, order, N):
    """
    Just returns the number of moment invariants for single or cross channel invariants of selected order
    and N-fold symmetry (N=1 if PSF is unconstrained)
    """
    gm_sample = np.zeros((order + 1, order + 1, 3))
    gm_sample[0, 0] = 1

    if num_channels == 2:
        inv_sample, _ = blur_channel_mixing_invariants_2channels(gm_sample, C_indices, order, N, typec=1, typex=0)
    else:
        inv_sample, _ = blur_channel_mixing_invariants_3channels(gm_sample, C_indices, order, N, typec=1, typex=0)

    number_of_invs = len(inv_sample)

    return number_of_invs
