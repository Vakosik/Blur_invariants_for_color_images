import numpy as np
from blur_invariants import central_moments, complex_moments, blur_invariants
from joblib import Parallel, delayed
from tqdm import tqdm


def compute_img_n_temp_invariants(img, temps, temp_sz, order, complex, img_name, method, combs, img_invariants,
                                  temp_normalization, typennum):
    """
    Computes invariants in all patches in img and for all templates in temps. If img_invariants is a name of npy
    file with already computed invariants in all patches, then it just loads it. If img_invariants=='save' then it
    computes the invariants in all patches and saves them.
    """

    invs_counts = []
    N = int(method[1])

    for comb in combs:
        if len(comb) == 1 or comb == 'gray':
            inv_kind = 'single_channel'
            if method == "unconstrained_invs":
                raise Exception("Only cross-channel invariants for unconstrained blur exist. Use only two-digits "
                                "elements in invs_comb")
        else:
            inv_kind = 'cross_channel'
        invs_counts.append(invs_counter(order, inv_kind, N))
    print("number of invariants in each combination:", invs_counts)

    if img_invariants == '' or img_invariants == 'save':
        print("Computing invariants of the image in all patches...")
        img_invs = get_image_invariants(img, invs_counts, order, complex, N, combs, temp_sz, temp_normalization,
                                        typennum)
        if img_invariants == 'save':
            invs_name = (f'computed_invs/{img_name}_r_{order}_{method}_inv_comb_{combs}_temp_sz_{temp_sz}'
                          f'_typen{typennum}_tempnorm{temp_normalization}_BtempSimg').replace(" ", "")
            np.save(f'{invs_name}.npy', img_invs)
    else:
        img_invs = np.load(f'computed invs/{img_invariants}')
        print(f"Image invariants loaded from {img_invariants}.")

    print("Computing invariants of the chosen templates...")
    temp_invs = get_template_invariants(temps, invs_counts, order, complex, N, combs, typennum)

    return img_invs, temp_invs


def get_image_invariants(img, invs_counts, order, complex, N, combs, temp_sz, temp_normalization, typennum):
    s = (np.array(img.shape) - temp_sz + 1).astype(int)
    img_invs = np.zeros((s[0], s[1], sum(invs_counts)))

    def invariants_by_row(r):
        row_invariants = np.zeros_like(img_invs[r, :, :])

        for c in range(s[1]):
            segment = img[r:r + temp_sz, c:c + temp_sz].copy()
            if temp_normalization:
                segment /= np.mean(segment)

            if len(img.shape) > 2:
                gm = np.zeros((order+1, order+1, 3))
                gm[:, :, 0] = central_moments(segment[:, :, 0], r=order)
                gm[:, :, 1] = central_moments(segment[:, :, 1], r=order)
                gm[:, :, 2] = central_moments(segment[:, :, 2], r=order)
            else:
                gm = central_moments(segment, r=order)

            moments = choose_moments(gm, order, complex)
            if complex:
                typex = 1
            else:
                typex = 0

            for idx, comb in enumerate(combs):
                start = sum(invs_counts[:idx])
                stop = sum(invs_counts[:idx+1])
                if len(comb) == 1:
                    row_invariants[c, start:stop], _ = blur_invariants(moments[:, :, int(comb)], order, N=N,
                                                                       typex=typex, typen=typennum)
                elif comb == 'gray':
                    row_invariants[c, :], _ = blur_invariants(moments, order, N=N, typex=typex, typen=typennum)
                else:
                    row_invariants[c, start:stop], _ = blur_invariants(moments[:, :, int(comb[0])], order, N=N,
                                                                       cmm2=moments[:,:, int(comb[1])], typex=typex,
                                                                       typen=typennum)

        return row_invariants

    # uses all cores in CPU
    invariants_by_rows = Parallel(n_jobs=-1)(delayed(invariants_by_row)(r) for r in tqdm(range(s[0])))

    for r, row_result in enumerate(invariants_by_rows):
        img_invs[r, :, :] = row_result

    return img_invs


def get_template_invariants(temps, invs_counts, order, complex, N, combs, typennum):
    n_temp = len(temps)
    temp_invs = np.zeros((n_temp, sum(invs_counts)))

    for temp in range(n_temp):
        if len(temps[temp].shape) > 2:
            gm = np.zeros((order + 1, order + 1, 3))
            gm[:, :, 0] = central_moments(temps[temp, :, :, 0], r=order)
            gm[:, :, 1] = central_moments(temps[temp, :, :, 1], r=order)
            gm[:, :, 2] = central_moments(temps[temp, :, :, 2], r=order)
        else:
            gm = central_moments(temps[temp], r=order)

        moments = choose_moments(gm, order, complex)
        if complex:
            typex = 1
        else:
            typex = 0

        for idx, comb in enumerate(combs):
            start = sum(invs_counts[:idx])
            stop = sum(invs_counts[:idx+1])
            if len(comb) == 1:
                temp_invs[temp, start:stop], _ = blur_invariants(moments[:, :, int(comb)], order, N=N, typex=typex,
                                                                 typen=typennum)
            elif comb == 'gray':
                temp_invs[temp, :], _ = blur_invariants(moments, order, N=N, typex=typex, typen=typennum)
            else:
                temp_invs[temp, start:stop], _ = blur_invariants(moments[:, :, int(comb[0])], order, N=N,
                                                                 cmm2=moments[:, :, int(comb[1])], typex=typex,
                                                                 typen=typennum)

    return temp_invs


def choose_moments(gm, order, complex):
    if complex:
        if len(gm.shape) > 2:
            moments = np.zeros_like(gm, dtype=np.cdouble)
            moments[:, :, 0] = complex_moments(order, gm[:, :, 0])
            moments[:, :, 1] = complex_moments(order, gm[:, :, 1])
            moments[:, :, 2] = complex_moments(order, gm[:, :, 2])
        else:
            moments = complex_moments(order, gm)
    else:
        moments = gm

    return moments


def invs_counter(order, inv_kind, N):
    """
    Just returns the number of moment invariants for single or cross channel invariants of selected order
    and N-fold symmetry (N=1 if PSF is unconstrained)
    """
    gm_sample = np.zeros((order + 1, order + 1))
    gm_sample[0, 0] = 1

    if inv_kind == 'single_channel':
        inv_sample, _ = blur_invariants(gm_sample, order, N=N, typex=0, typen=0)
    else:
        inv_sample, _ = blur_invariants(gm_sample, order, N=N, cmm2=gm_sample, typex=0, typen=0)

    number_of_invs = len(inv_sample)

    return number_of_invs
