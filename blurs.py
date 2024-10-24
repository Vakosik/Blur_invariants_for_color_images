import numpy as np
from scipy.signal import convolve


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


def blur(i, size=3, btype="square"):
    """
    Blur by convolution. Right now either "square" kernel or "T" kernel
    """
    i_blured = np.zeros_like(i)

    if btype == "square":
        kernel = np.ones((size, size)) / (size ** 2)
    elif btype == "T":
        kernel = create_T_blur(size)

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
    padsize = int((size_of_blur - 1) / 2)

    if len(template.shape) > 2:
        template_padded = np.pad(template, ((padsize, padsize), (padsize, padsize), (0, 0)),
                                 'constant', constant_values=0)
    else:
        template_padded = np.pad(template, ((padsize, padsize), (padsize, padsize)),
                                 'constant', constant_values=0)

    template_blured = blur(template_padded, size=size_of_blur, btype=blur_type)

    return template_blured
