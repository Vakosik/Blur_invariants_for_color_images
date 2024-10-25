import numpy as np
from math import comb


def central_moments(l, r):
    """ M is a matrix of central moments up to the r-th order of the image l
        l - image matrix
        The moment μ_{pq} = M(p+1,q+1) """

    n1, n2 = l.shape

    m00 = np.sum(l)

    w = np.arange(1, n2 + 1)
    v = np.arange(1, n1 + 1)

    if m00 != 0:
        tx = np.sum(l * w) / m00
        ty = np.sum(v * np.transpose(l)) / m00
    else:
        tx = 0
        ty = 0

    a = w - tx
    c = v - ty

    M = np.zeros((r + 1, r + 1))

    for i in range(r + 1):
        for j in range(r + 1 - i):
            A = np.power(a, i)
            C = np.power(c, j)
            M[i, j] = np.matmul(C, np.matmul(l, A))

    if r > 0:
        M[0, 1] = 0
        M[1, 0] = 0

    return M


def blur_invariants(cmm, r, N, cmm2=None, typec=1, typex=0, typen=0):
    """computes moment invariants to convolution where point spread function (PSF) has N-fold rotation symmetry
    or is unconstrained.

    cmm is a matrix of moments of a certain channel
    r is the maximum order of the invariants, so the size of cmm is = (r+1) x (r+1).
    N is the fold number of the rotation symmetry of the PSF.
    If N==np.inf, circular symmetry of PSF is supposed
    If N==1, unconstrained PSF is supposed
    If N>1 finite, then rotation N-fold symmetry of PSF is supposed.
    If cmm2 is passed, then cross-channel blur invariants are computed
    typec and typex are types of moments, typen is normalization:
    typec=0 means general moments, typec=1 means central moments,
    typex=0 means geometric moments, typex=1 means complex moments.
    typen=0 no normalization, typen=1 scaling normalization

    inv is vector of the invariants,
    invmat are the invariants in matrix form, C(p,q)=invmat(p,q).
    ind is matrix of indices, ind(1,k) is p-index of the k-th invariant,
    ind(2,k) is its q-index and ind(3,k) is indicator of imaginarity,
    if ind(3,k)==0, then inv(k) is the value of real part
    of C(ind(1,k),ind(2,k)), if ind(3,k)==1, it is the value of
    the imaginary part."""

    invmat = np.zeros((r + 1, r + 1))
    index = 0
    inv = np.array([])
    ind = np.zeros((1, 3))

    for current_order in range(r + 1):
        for p in range(current_order + 1):
            q = current_order - p
            if cmm2 is None and (p - q) % N == 0:
                continue
            if cmm2 is not None and (p - q) % N != 0:
                continue

            s = 0
            for n in range(p + 1):
                for m in range(q + 1):
                    if (m + n > 0) and ((n - m) % N) == 0:
                        s += comb(p, n) * comb(q, m) * invmat[p - n, q - m] * cmm[n, m]

            if cmm[0, 0] != 0:
                s /= cmm[0, 0]
            if cmm2 is None:
                invmat[p, q] = cmm[p, q] - s
            else:
                invmat[p, q] = cmm[0, 0] * cmm2[p, q] - s

            if ((cmm2 is None and (p - q) % N != 0)
                or (cmm2 is not None and (p - q) % N == 0 and (p+q) > 0)) and (typec == 0 or p + q > 1):
                if typex == 0:
                    inv = np.append(inv, invmat[p, q])
                    ind = np.vstack((ind, np.array([p, q, 0])))
                    index += 1
                else:
                    if p > q:
                        inv = np.append(inv, np.real(invmat[p, q]))
                        ind = np.vstack((ind, [p, q, 0]))
                        index += 1
                        inv = np.append(inv, np.imag(invmat[p, q]))
                        ind = np.vstack((ind, [p, q, 1]))
                        index += 1

    ind = ind[1:]
    index -= 1

    if typen > 0:
        # scaling normalization
        pm, qm = np.meshgrid(np.arange(r + 1), np.arange(r + 1))
        if cmm2 is None:
            scaling_factor = abs(cmm[0][0]) ** ((qm + pm + 2) / 2)
            scaling_factor2 = abs(cmm[0][0]) ** ((ind[:, 0] + ind[:, 1] + 2) / 2)
        else:
            scaling_factor = (0.5 * (abs(cmm[0][0]) + abs(cmm2[0][0]))) ** ((qm + pm + 4) / 2)
            scaling_factor2 = (0.5 * (abs(cmm[0][0]) + abs(cmm2[0][0]))) ** ((ind[:, 0] + ind[:, 1] + 4) / 2)

        invmat /= scaling_factor
        inv /= scaling_factor2

    return inv
