# Repository content
The repository now comprises code for two journal articles. The first has already been published in Pattern Recognition https://doi.org/10.1016/j.patcog.2025.112358. The second paper has been submitted to IEEE Transactions on Image Processing.

The PR article introduces blur invariants to color (in general multispectral) images. Its novelty are the so-called cross-channel blur invariants that employ linkage between channels.

The IEEE TIP article comes up with features that are invariant to both blur and linear channel-mixing at the same time. This image degradation model is more general.

The invariants are implemented in the image domain using moments of images. The blur invariants can be found in "blur_invariants/blur_invariants.py", the invariants to blur and channel-mixing in "blur_n_channelmixing_invariants/blur_channelmixing_invariants.py".

The repository contains primarily template matching experiments. We include a set of images used for experiments with synthetic blur (images/synthetic_blur_experiment) and also a set of image pairs where each pair captures the same scene, but one is blurred by real out-of-focus blur (images/real_blur_experiment). The templates from the experiments can be found in the "templates" folder. A possible channel-mixing is implemented by a) a matrix multiplication with chosen matrix A, 2) transformation to YCbCr color space or 3) by the PCA transformation where matrix A is data dependent.
Template matching experiments can be run by "run_template_matching.py". There is a list of paramaters determining the template matching settings (we list the parameters that were used in the two articles below), there are much more options than just those that were performed for the papers. All results from both papers can be replicated by running corresponding loops (images, method, size_of_blur/template_size). 

The user can switch between the invariants (blur vs blur-and-channel-mixing) by a boolean parameter "channel_mixing" in run_template_matching.py.

The IEEE TIP article contains template matching experiment using DINOv2 features. This is implemented in DINOv2_template_matching.py. The settings used in the paper is:  mid_block_idx = 9, P = 14, separate_crops = True, tokens_mode = 'except_cls', patch_norm = True.
The DINOv2 script is fully functional, but I would like to get to some code cleaning for better readibility. In the real-blur experiment, DINOv2 was run on JPEG versions of the images, as it was mostly trained on JPEG data, CR2 was used for the invariants to better preserve the convolutional model.

# Blur Invariants (PR article)
Blur moment invariants (both single-channel and cross-channel) are in "blur_invariants/blur_invariants.py". For now, there are invariants to blur with N-fold symmetry PSF and unconstrained blur. Remember to use complex moments if N > 2.

![sharp08_templates_sz_100_handmade](https://github.com/user-attachments/assets/eae60f6c-8c35-4327-93ce-2cdc88e98f99)

### Parameter setting for the Pattern Recognition paper:
synthetic blur experiments:
subfolder = 'synthetic_blur_experiment';
full_img_name in [f"sharp{i:02}.JPG" for i in range(0, 15)];
downscale = 2;
blur_type = "square" or "T" or "square_padded" or "T_padded" depending on the experiment;
size_of_blur in np.arange(9, 100, 6) ;
selected_templates = img_name + "_100temp_positions.npy";
temp_sz = 100;
method = 'N1fold_invs' or 'N2fold_invs' or "RGB_crosscorr" or "gray_croscorr";
invs_comb in [('0', '1', '2', '10', '21'), ('0', '1', '2'), (['gray'])];
order = 5;
complex = False;
one_center = True;
norm = 2;
temp_normalization = False;
typennum = 1;
sigma = 0 (but sigma = np.arange(0.0, 0.4001, 0.02) for testing the relative errors under noise);
channel_mixing = False;
a = 0;
(C_indices of no relevance when channel_mixing=False)

real blur experiment:
subfolder = 'real_blur_experiment';
full_img_name in [f"sharp{i:02}.CR2" for i in range(0, 3)];
blur_type = "blurred0i.CR2" where i=0,1,2
size_of_blur = 0;
selected_templates = img_name + "_100temp_positions.npy";
temp_sz = np.arange(50, 151, 10);
method = 'N2fold_invs' or "RGB_crosscorr" or "gray_croscorr";
invs_comb in [('0', '1', '2', '10', '21'), ('0', '1', '2'), (['gray'])];
order = 5;
complex = False;
one_center = True;
norm = 2;
temp_normalization = True
typennum = 1;
sigma = 0;
channel_mixing = False;
a = 0;
(C_indices of no relevance when channel_mixing=False)


The code was expanded and made more general for the IPTA conference paper (https://doi.org/10.1109/IPTA66025.2025.11222047). We performed there an experiment with synethetic blur, the differences in settings:
blur_type = "triangle" or "triangle_padded";
size_of_blur in np.arange(9, 64, 6) (the triangle blur is much stronger);
method = 'N2fold_invs' or "N3fold_invs";
complex = True;
norm = 1;
typennum = 0

The experiment with recognizing of centrosymmetric objects is present in "recognition_of_centrosymmetric_objects" folder including the road signs images. In the PR article, we briefly explain that single-channel blur invariants constructed for N2-fold symmetric PSFs are trivial for centrosymmetric objects, but the new cross-channel invariants are able to dinstinguish between them. Eventually, there was not enough space for this experiments due to revisions, but I keep it in the repository.
![road_signs_together](https://github.com/user-attachments/assets/62dc8cfd-189d-45ee-b824-ef4f97561e6e)

# Invariants to blur and channel-mixing (IEEE TIP article)
The implementation of the invariants can be found in "blur_n_channelmixing_invariants/blur_channelmixing_invariants.py". There is a 2-channel and a 3-channel version (but formula for a general c-channel image is written in the article).

### Parameter setting for the IEEE TIP paper:
synthetic blur experiment:
subfolder = 'synthetic_blur_experiment';
full_img_name in [f"sharp{i:02}.JPG" for i in range(0, 15)];
downscale = 2;
blur_type = "square";
size_of_blur in np.concatenate(([0], np.arange(9, 71, 6)));
selected_templates = img_name + "_100temp_positions.npy";
temp_sz = 98;
method = 'N2fold_invs';
order = 5;
complex = False;
one_center = False;
norm = 1;
temp_normalization = False
typennum = 0;
sigma = 0;
channel_mixing = True;
a in [0, 0.2, 0.3];
C_indices = [[(0,0)], [(1,0)], [(1,0)], [(0,1)]];
(inv_comb of no relevance when channel_mixing=True)

real blur experiment:
subfolder = 'real_blur_experiment';
full_img_name in ["sharp00.CR2", "sharp01_alt.CR2", "sharp02_alt.CR2"];
blur_type in ["blurred00.CR2", "blurred01_alt.CR2", "blurred02_alt.CR2"];
size_of_blur = 0;
selected_templates = img_name + "_100temp_positions.npy";
temp_sz = np.arange(56, 155, 14)
method = 'N2fold_invs';
order = 5;
complex = False;
one_center = False;
norm = 1;
temp_normalization = False
typennum = 0;
sigma = 0;
channel_mixing = True;
a = 0 (PCA and YCbCr options are commented later in the code);
C_indices = [[(0,0)], [(1,0)], [(1,0)], [(0,1)]];
(inv_comb of no relevance when channel_mixing=True)
