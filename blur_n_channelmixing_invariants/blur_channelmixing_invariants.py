import numpy as np
from math import comb
from blur_invariants.blur_invariants import build_invvec


def blur_channel_mixing_invariants_2channels(cmm, C_indices, r, N, typec=1, typex=0):
    """computes moment invariants to channel mixing and convolution where the matrix mixing the channels is arbitrary
     and point spread function (PSF) has N-fold rotation symmetry or is unconstrained. The 3x3 mixing matrix A
     transforms the image as A*f = A*(f_1, f_2) where * is matrix multiplication.

    cmm is a matrix of moments containing the channels in the first dimension
    j is a list in the form [[(i1,k1),...,(i?, k?)], [(l1,o1),...,(l?, o?)]]. In j[0], there are indices of moments
        invariant to blur that will sum up to create the first C. In j[1], there are indices of moments whose sum
        defines the second C.
    r is the maximum order of the invariants, so the size of cmm is = 2 x (r+1) x (r+1).
    N is the fold number of the rotation symmetry of the PSF.
    If N==np.inf, circular symmetry of PSF is supposed
    If N==1, unconstrained PSF is supposed
    If N>1 finite, then rotation N-fold symmetry of PSF is supposed.
    If the moments are geometric, N should be 1 or 2 (geometric moments don't separate the moments otherwise)

    typec and typex are types of moments, typen is normalization:
    typec=0 means general moments, typec=1 means central moments,
    typex=0 means geometric moments, typex=1 means complex moments.

    inv is vector of the invariants,
    invmat are the invariants in matrix form, C(p,q)=invmat(p,q).
    ind is matrix of indices, ind(1,k) is p-index of the k-th invariant,
    ind(2,k) is its q-index and ind(3,k) is indicator of imaginarity,
    if ind(3,k)==0, then inv(k) is the value of real part
    of C(ind(1,k),ind(2,k)), if ind(3,k)==1, it is the value of
    the imaginary part."""

    if typex == 0:
        dtype = np.double
    else:
        dtype = np.cdouble

    invmat = np.zeros((r + 1, r + 1), dtype)
    index = 0
    inv = np.array([])
    ind = np.zeros((1, 3))

    C_j1, C_j2 = [0, 0], [0, 0]
    for indices in C_indices[0]:
        C_j1 += cmm[indices[0], indices[1], :]
    for indices in C_indices[1]:
        C_j2 += cmm[indices[0], indices[1], :]

    for current_order in range(r + 1):
        for p in range(current_order + 1):
            q = current_order - p
            s = 0
            for n in range(p + 1):
                for m in range(q + 1):
                    if (m + n > 0) and ((n - m) % N) == 0:
                        s += comb(p, n) * comb(q, m) * invmat[p - n, q - m] * (
                                    C_j2[1] * cmm[n, m, 0] - C_j2[0] * cmm[n, m, 1])

            invmat[p, q] = (C_j1[1] * cmm[p, q, 0] - C_j1[0] * cmm[p, q, 1]) - s

            if np.all(cmm[:, 0, 0] != 0):
                invmat[p, q] /= (C_j2[1] * cmm[0, 0, 0] - C_j2[0] * cmm[0, 0, 1])

            invvec, ind, index = build_invvec(p, q, invvec, invmat, ind, index, typec, typex)

    ind = ind[1:]
    index -= 1

    return invvec, invmat


def blur_channel_mixing_invariants_3channels(cmm, C_indices, r, N, typec=1, typex=0):
    """computes moment invariants to channel mixing and convolution where the matrix mixing the channels is arbitrary
     and point spread function (PSF) has N-fold rotation symmetry or is unconstrained. The 3x3 mixing matrix A
     transforms the image as A*f = A*(f_1, f_2, f_3) where * is matrix multiplication.

    cmm is a matrix of moments containing the channels in the first dimension
    j is a list containing indices of moments that are invariant to blur and that define the four Cs in the same way as
        in 2channel case.
    r is the maximum order of the invariants, so the size of cmm is = (r+1) x (r+1).
    N is the fold number of the rotation symmetry of the PSF.
    If N==np.inf, circular symmetry of PSF is supposed
    If N==1, unconstrained PSF is supposed
    If N>1 finite, then rotation N-fold symmetry of PSF is supposed.
    If the moments are geometric, N should be 1 or 2 (geometric moments don't separate the moments otherwise)

    typec and typex are types of moments, typen is normalization:
    typec=0 means general moments, typec=1 means central moments,
    typex=0 means geometric moments, typex=1 means complex moments.

    inv is vector of the invariants,
    invmat are the invariants in matrix form, C(p,q)=invmat(p,q).
    ind is matrix of indices, ind(1,k) is p-index of the k-th invariant,
    ind(2,k) is its q-index and ind(3,k) is indicator of imaginarity,
    if ind(3,k)==0, then inv(k) is the value of real part
    of C(ind(1,k),ind(2,k)), if ind(3,k)==1, it is the value of
    the imaginary part."""

    if typex == 0:
        dtype = np.double
    else:
        dtype = np.cdouble

    invmat = np.zeros((r + 1, r + 1), dtype)
    index = 0
    invvec = np.array([])
    ind = np.zeros((1, 3))

    C_j1, C_j2 = [0, 0, 0], [0, 0, 0]
    C_j3, C_j4 = [0, 0, 0], [0, 0, 0]
    for indices in C_indices[0]:
        C_j1 += cmm[indices[0], indices[1], :]
    for indices in C_indices[1]:
        C_j2 += cmm[indices[0], indices[1], :]
    for indices in C_indices[2]:
        C_j3 += cmm[indices[0], indices[1], :]
    for indices in C_indices[3]:
        C_j4 += cmm[indices[0], indices[1], :]

    for current_order in range(r + 1):
        for p in range(current_order + 1):
            q = current_order - p
            s = 0
            for n in range(p + 1):
                for m in range(q + 1):
                    if (m + n > 0) and ((n - m) % N) == 0:
                        cross_product = 0
                        for i in range(3):
                            cross_product += C_j3[i] * C_j4[(i + 1) % 3] * cmm[n, m, (i + 2) % 3] - C_j3[i] * C_j4[
                                (i + 2) % 3] * cmm[n, m, (i + 1) % 3]
                        s += comb(p, n) * comb(q, m) * invmat[p - n, q - m] * cross_product

            cross_product = 0
            for i in range(3):
                cross_product += (C_j1[i] * C_j2[(i + 1) % 3] * cmm[p, q, (i + 2) % 3] - C_j1[i] * C_j2[(i + 2) % 3] *
                                  cmm[p, q, (i + 1) % 3])
            invmat[p, q] = cross_product - s

            cross_product = 0
            for i in range(3):
                cross_product += C_j3[i] * C_j4[(i + 1) % 3] * cmm[0, 0, (i + 2) % 3] - C_j3[i] * C_j4[
                    (i + 2) % 3] * cmm[0, 0, (i + 1) % 3]
            if cross_product != 0:
                invmat[p, q] /= cross_product
            else:
                invmat[p, q] = 1
                if p == 0 and q == 0:
                    print(f"division by 0 in cmm[0, 0] normalization, the corresponding invmat was set to 1 for all p, q")

            if (p,q) not in C_indices[0] and (p,q) not in C_indices[1] and (p,q) not in C_indices[2] and (p,q) not in C_indices[3]:
                invvec, ind, index = build_invvec(p, q, invvec, invmat, ind, index, typec, typex)

    ind = ind[1:]
    index -= 1

    return invvec, invmat
