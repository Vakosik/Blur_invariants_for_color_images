import numpy as np
from math import comb


def average_center(l):
    n1, n2, n3 = l.shape
    tx_stack = []
    ty_stack = []

    for channel in range(n3):
        m00 = np.sum(l[:,:,channel])

        w = np.arange(1, n2 + 1)
        v = np.arange(1, n1 + 1)

        if m00 != 0:
            tx_stack.append(np.sum(l[:,:,channel] * w) / m00)
            ty_stack.append(np.sum(v * np.transpose(l[:,:,channel])) / m00)
        else:
            tx_stack.append(0)
            ty_stack.append(0)

    tx = np.mean(np.array(tx_stack))
    ty = np.mean(np.array(ty_stack))

    return tx, ty


def central_moments(l, r, tx=None, ty=None):
    """ M is a matrix of central moments up to the r-th order of the image l
        l - image matrix
        The moment μ_{pq} = M(p+1,q+1) """

    n1, n2 = l.shape

    w = np.arange(1, n2 + 1)
    v = np.arange(1, n1 + 1)

    if tx is None or ty is None:
        m00 = np.sum(l)
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

    # if r > 0 and (tx is None or tx is None):
    #     M[0, 1] = 0
    #     M[1, 0] = 0

    return M


def complex_moments(order, gm):
    # Compute complex moments c_{pq} from geometric moments gm
    # up to the specified order.
    #
    # Input arguments
    # ---
    #   order ... maximum order of moments
    #   gm    ... matrix of geometric moments (order +1) x (order +1)
    #
    # Output arguments
    # ---
    #   c ... matrix of complex moments

    c = np.zeros((order + 1, order + 1), dtype=np.cdouble)
    for p in range(order + 1):
        for q in range(order+1-p):
            for k in range(p+1):
                pk = comb(p, k)
                for j in range(q+1):
                    qj = comb(q, j)
                    c[p, q] = c[p, q] + pk * qj * (-1)**(q - j) * 1j**(p + q - k - j
                                                                            ) * gm[k + j, p + q - k - j]
    return c


def blur_invariants(cmm, r, N, cmm2=None, typec=1, typex=0, typen=0):
    """computes moment invariants to convolution where point spread function (PSF) has N-fold rotation symmetry
    or is unconstrained.

    cmm is a matrix of moments of a certain channel
    r is the maximum order of the invariants, so the size of cmm is = (r+1) x (r+1).
    N is the fold number of the rotation symmetry of the PSF.
    If N==np.inf, circular symmetry of PSF is supposed
    If N==1, unconstrained PSF is supposed
    If N>1 finite, then rotation N-fold symmetry of PSF is supposed.
    If the moments are geometric, N should be 1 or 2 (geometric moments don't separate the moments otherwise)
    If cmm2 is passed, then cross-channel blur invariants are computed
    typec and typex are types of moments, typen is normalization:
    typec=0 means general moments, typec=1 means central moments,
    typex=0 means geometric moments, typex=1 means complex moments.
    typen=0 no normalization, typen=1 scaling normalization

    invvec is vector of the invariants,
    invmat are the invariants in matrix form, C(p,q)=invmat(p,q).
    ind is matrix of indices, ind(1,k) is p-index of the k-th invariant,
    ind(2,k) is its q-index and ind(3,k) is indicator of imaginarity,
    if ind(3,k)==0, then inv(k) is the value of real part
    of C(ind(1,k),ind(2,k)), if ind(3,k)==1, it is the value of
    the imaginary part."""

    if typex == 0:
        dtype = float
    else:
        dtype = np.cdouble

    invmat = np.zeros((r + 1, r + 1), dtype)
    index = 0
    invvec = np.array([])
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

            invvec, ind, index = build_invvec(p, q, invvec, invmat, ind, index, typec, typex)

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
        invvec /= scaling_factor2

    return invvec, invmat


def build_invvec(p, q, invvec, invmat, ind, index, typec, typex):
    if ((p + q) > 0) and (typec == 0 or p + q > 1):
        if typex == 0:
            invvec = np.append(invvec, invmat[p, q])
            ind = np.vstack((ind, np.array([p, q, 0])))
            index += 1
        else:
            if p >= q:
                invvec = np.append(invvec, np.real(invmat[p, q]))
                ind = np.vstack((ind, [p, q, 0]))
                index += 1
                if p > q:
                    invvec = np.append(invvec, np.imag(invmat[p, q]))
                    ind = np.vstack((ind, [p, q, 1]))
                    index += 1

    return invvec, ind, index
