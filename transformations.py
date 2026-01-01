import numpy as np
from scipy.signal import convolve
import math
from sklearn.decomposition import PCA


def create_T_blur(size):
    """
    Function designing a non-symmetric blur kernel resembling T letter
    """
    kernel = np.zeros((size, size))

    if size < 5:
        width = 1
    else:
        width = 3
    center = int((size - 1) / 2)

    kernel[:, int((size - 3) / 6 - 1):width + int((size - 3) / 6 - 1)] = 1

    half_width = int((width - 1) / 2)
    for i in range(1, center + 1):
        kernel[center - half_width:center + half_width + 1, center + i] = (i) / ((center) * (size - 1) * 3)

    ratio = np.sum(kernel[:, center + 1:]) / np.sum(kernel[:, :center])
    kernel[:, :center] *= ratio
    kernel /= np.sum(kernel)

    return kernel


def create_triangle_vertices_kernel(size):
    """
    Function designing a 3-fold blur kernel of equilateral triangle. Intensities are only in the vertices.
    The kernel is centered to the center of mass. In the discrete case, we cannot design the perfect equilateral
    triangle, but the approximation should be decent.
    """

    height = int(np.ceil((size / 2) / math.tan(math.pi / 6)))
    kernel = np.zeros((height, size), dtype=np.float64)

    kernel[0, int(size / 2)] = 1
    kernel[-1, 0] = 1
    kernel[-1, -1] = 1

    row_padding = np.zeros((int(height / 3), size))
    kernel = np.vstack((kernel, row_padding))

    kernel /= np.sum(kernel)

    return kernel


def create_disk_kernel(size):
    y, x = np.mgrid[:size, :size]
    kernel = np.hypot(y - (size - 1) / 2, x - (size - 1) / 2) < (size + 1) / 2
    kernel = kernel.astype(np.float64)
    kernel /= np.sum(kernel)

    return kernel


def blur(i, size=3, btype="square"):
    """
    Blur by convolution. Right now either "square" kernel or "T" kernel
    """
    if size == 0:
        return i

    i_blured = np.zeros_like(i)

    if btype == "square":
        kernel = np.ones((size, size)) / (size ** 2)
    elif btype == "T":
        kernel = create_T_blur(size)
    elif btype == "triangle":
        kernel = create_triangle_vertices_kernel(size)
    elif btype == "disk":
        kernel = create_disk_kernel(size)

    if len(i.shape) > 2:
        for channel in range(i.shape[2]):
            i_blured[:, :, channel] = convolve(i[:, :, channel], kernel, mode="same")
    else:
        i_blured = convolve(i, kernel, mode="same")

    return i_blured


def template_padding_blur(template, size_of_blur, blur_type):
    """
    Padding sharp templates with zero and then blurring them with chosen kernel
    """
    btype = blur_type[:-7]

    if btype == 'triangle':
        kernel = create_triangle_vertices_kernel(size_of_blur)
        padsize = int(max(kernel.shape[0], kernel.shape[1])/2)
    else:
        padsize = int(size_of_blur / 2)

    if len(template.shape) > 2:
        template_padded = np.pad(template, ((padsize, padsize), (padsize, padsize), (0, 0)),
                                 'constant', constant_values=0)
    else:
        template_padded = np.pad(template, ((padsize, padsize), (padsize, padsize)),
                                 'constant', constant_values=0)

    template_blured = blur(template_padded, size=size_of_blur, btype=btype)

    return template_blured


def add_noise(image, sigma):
    if sigma == 0:
        return image

    row, col, ch = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = image + gauss

    return noisy_image

def rgb_to_pca(img, n_components=3):
    H, W, C = img.shape
    assert C == 3, "Expected RGB image with 3 channels"

    # Flatten to (N, 3) where N = H * W
    X = img.reshape(-1, 3).astype(np.float64)
    N = X.shape[0]

    # Scatter matrix about the origin: S = (1/N) X^T X
    S = (X.T @ X) / N   # shape (3, 3)

    # Eigen-decomposition: S v = λ v
    eigvals, eigvecs = np.linalg.eigh(S)  # columns of eigvecs are eigenvectors

    # Sort eigenvalues/vectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]  # still (3, 3)
    A = eigvecs[:, :n_components].T  # (n_components, 3)

    X_pca = X @ A.T  # shape (N, n_components)
    pca_img = X_pca.reshape(H, W, n_components)

    return pca_img


def rgb_to_ycbcr(img):
    M = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])

    ycbcr = img @ M.T

    return ycbcr


def mixing_channels(image, w):
    i_combined = np.zeros_like(image)
    i_combined[:, :, 0] = w[0, 0] * image[:, :, 0] + w[0, 1] * image[:, :, 1] + w[0, 2] * image[:, :, 2]
    i_combined[:, :, 1] = w[1, 0] * image[:, :, 0] + w[1, 1] * image[:, :, 1] + w[1, 2] * image[:, :, 2]
    i_combined[:, :, 2] = w[2, 0] * image[:, :, 0] + w[2, 1] * image[:, :, 1] + w[2, 2] * image[:, :, 2]

    return i_combined


def nearest_multiple_of_14(a, b):
    def nearest(n):
        remainder = n % 14
        if remainder < 7:
            return n - remainder
        else:
            return n + (14 - remainder)

    return nearest(a), nearest(b)
