import numpy as np
from skimage.feature import match_template


def correlation(templ, img):
    """
    Returns row, column and value of the maximum of normalized cross-correlation
    """
    c = match_template(img, templ)
    detected_pos = np.unravel_index(np.argmax(np.abs(c)), c.shape)
    row, column = detected_pos[0], detected_pos[1]
    max_c = np.max(np.abs(c))
    return np.array([row, column, max_c])


def compare_invariants(IF, TF, norm=1):
    """
    Compares blur invariants of a template with blur invariants of all possible patches in an image. The minimum of
    L2 norm of relative errors in the feature space is chosen as the best match.
    :param IF: Blur invariants of all patches in the image
    :param TF: Blur invariants of a template
    :return:
    """
    RE = np.abs(IF-TF) / np.abs(IF)

    norm_RE = np.linalg.norm(RE, ord=norm, axis=2)

    NRE = np.min(norm_RE)
    if len(np.where(norm_RE == NRE)[0]) > 1 or len(np.where(norm_RE == NRE)[1]) > 1:
        print("More than one match with the same L2 distance in the feature space. Choosing one of them.")
    row = np.where(norm_RE == NRE)[0][0]
    col = np.where(norm_RE == NRE)[1][0]

    return row, col, NRE
