from transformations import blur
from blur_invariants.blur_invariants import central_moments, blur_invariants
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import img_as_float
import os

# Settings to visualize centrosymmetric objects (both sharp and blurred) in two feature spaces -- the first is a
# feature space of two cross-channel moment invariants (where the objects are supposed to be separated), the other
# of two single-channel invariants (which are theoretically 0 for all centrosymmetric objects)
# _________________________________________________________________________________

channel_0 = 0  # channel used for single-channel invariants and one of the two channels used for cross-channel invs
channel_1 = 2  # the other channel used for cross-channel invariants
sizes_of_blur = np.arange(11, 42, 10)  # kernel sizes used for blurring the centrosymmetric objects
files = ['B29_2bb_sym.png', "B01_inftybb_sym.png", "B26_2abb_sym.png",
         "B02_2bb_sym.png", "hospital_ch_2bb_sym.png", "P03_2bb_sym.png"] # file names of centrosymmetric objects

# indices of blur moment invariants for feature space visualisation. The first two are cross-channel invs, the last
# two are single-schannel invs. For cross-channel invs, choose only even order, for single-channel only odd order!
first_cross_invs = [2, 2]
second_cross_invs = [0, 2]
first_single_invs = [3, 0]
second_single_invs = [0, 3]

figsize = (7, 7)  # size of graphs. Choose square
imgs_title = "Road_signs"  # The beginning of the graph names at the output

# _________________________________________________________________________________

# maximum order of chosen moments
r = np.max(np.sum(np.array([first_cross_invs, second_cross_invs, first_single_invs, second_single_invs]), axis=1))

inv_single_sharp = np.zeros((len(files), r + 1, r + 1))
inv_single_blured = np.zeros((len(files), len(sizes_of_blur), r + 1, r + 1))
inv_cross_sharp = np.zeros((len(files), r + 1, r + 1))
inv_cross_blured = np.zeros((len(files), len(sizes_of_blur), r + 1, r + 1))

for idx_file, file_name in enumerate(files):
    img_sharp = cv2.imread(os.path.join(file_name))
    img_sharp = cv2.resize(img_sharp, (200, 200))
    img_sharp = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2RGB)
    img_sharp = img_as_float(img_sharp)

    gm_sharp_0 = central_moments(img_sharp[:, :, channel_0], r=r)
    gm_sharp_1 = central_moments(img_sharp[:, :, channel_1], r=r)

    _, inv_single_sharp[idx_file, :, :] = blur_invariants(gm_sharp_0, r, N=2, typex=0, typen=0)
    _, inv_cross_sharp[idx_file, :, :] = blur_invariants(gm_sharp_0, r, N=2, cmm2=gm_sharp_1, typex=0, typen=0)

    for idx_blur, size_of_blur in enumerate(sizes_of_blur):
        img_blured = blur(img_sharp, size_of_blur, "square")

        gm_blured_0 = central_moments(img_blured[:, :, channel_0], r=r)
        gm_blured_1 = central_moments(img_blured[:, :, channel_1], r=r)

        _, inv_single_blured[idx_file, idx_blur, :, :] = blur_invariants(gm_blured_0, r, N=2, typex=0, typen=0)
        _, inv_cross_blured[idx_file, idx_blur, :, :] = blur_invariants(gm_blured_0, r, N=2, cmm2=gm_blured_1, typex=0,
                                                                        typen=0)

colors = ['red', 'blue', 'lime', 'gold', 'black', 'cyan']
plt.rc('font', **{'size': '14'})
size_of_dots = 90

# Plotting graph of cross-channel moment invariants

plt.figure(figsize=figsize)
plt.title('Cross-channel BIs of centrosymmetric objects', fontsize=20, y=1.05)
xlabel_name = f"$C^{{({channel_0 + 1}{channel_1 + 1})}}_{{{first_cross_invs[0]}{first_cross_invs[1]}}}$"
ylabel_name = f"$C^{{({channel_0 + 1}{channel_1 + 1})}}_{{{second_cross_invs[0]}{second_cross_invs[1]}}}$"
plt.xlabel(xlabel_name, fontsize=24)
plt.ylabel(ylabel_name, fontsize=24, rotation=0, labelpad=20)

for i in range(len(files)):
    x = inv_cross_sharp[i, first_cross_invs[0], first_cross_invs[1]]
    y = inv_cross_sharp[i, second_cross_invs[0], second_cross_invs[1]]
    plt.scatter(x, y, c=colors[i], s=size_of_dots, marker='o')
    for j in range(len(sizes_of_blur)):
        x = inv_cross_blured[i, j, first_cross_invs[0], first_cross_invs[1]]
        y = inv_cross_blured[i, j, second_cross_invs[0], second_cross_invs[1]]
        plt.scatter(x, y, c=colors[i], s=size_of_dots, marker='^')

lgnd = plt.legend(['sharp', 'blurred'], fontsize=20)
for handle in lgnd.legend_handles:
    handle._sizes = [size_of_dots]
    handle.set_edgecolor("black")  # Set black border
    handle.set_facecolor("white")  # Set white background

plt.savefig( f'plots/{imgs_title}_cross-channel_invs_space_M{first_cross_invs[0]}{first_cross_invs[1]}'
             f'_M{second_cross_invs[0]}{second_cross_invs[1]}.pdf', format="pdf", bbox_inches="tight")


# Plotting graph of single-channel moment invariants
plt.figure(figsize=figsize)
plt.title('Single-channel BIs of centrosymmetric objects', fontsize=20, y=1.05)
xlabel_name = f"$C^{{({channel_0 + 1}{channel_0 + 1})}}_{{{first_single_invs[0]}{first_single_invs[1]}}}$"
ylabel_name = f"$C^{{({channel_0 + 1}{channel_0 + 1})}}_{{{second_single_invs[0]}{second_single_invs[1]}}}$"
plt.xlabel("$C^{(11)}_{30}$", fontsize=24)
plt.ylabel("$C^{(11)}_{03}$", fontsize=24, rotation=0, labelpad=20)

for i in range(len(files)):
    x = inv_single_sharp[i, first_single_invs[0], first_single_invs[1]]
    y = inv_single_sharp[i, second_single_invs[0], second_single_invs[1]]
    plt.scatter(x, y, c=colors[i], s=size_of_dots, marker='o')
    for j in range(len(sizes_of_blur)):
        x = inv_single_blured[i, j, first_single_invs[0], first_single_invs[1]]
        y = inv_single_blured[i, j, second_single_invs[0], second_single_invs[1]]
        plt.scatter(x, y, c=colors[i], s=size_of_dots, marker='^')

lgnd = plt.legend(['sharp', 'blurred'], fontsize=20)
for handle in lgnd.legend_handles:
    handle._sizes = [size_of_dots]
    handle.set_edgecolor("black")  # Set black border
    handle.set_facecolor("white")  # Set white background

plt.savefig( f'plots/{imgs_title}_single-channel_invs_space_M{first_single_invs[0]}{first_single_invs[1]}'
             f'_M{second_single_invs[0]}{second_single_invs[1]}.pdf', format="pdf", bbox_inches="tight")
