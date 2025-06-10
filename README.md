Blur moment invariants (both single-channel and cross-channel) are in "blur_invariants.py". For now, there are only invariants to blur with N-fold symmetry PSF and unconstrained blur. Remember to use complex moments if N > 2.

All template matching experiments can run by "run_template_matching.py". Both padding and no_padding options are available as well as invariant and cross-correlation methods. All results from the paper can be replicated by running corresponding loops (images, method, size_of_blur/template_size). 
The used images and templates are saved in "images" and "templates" folders.
![sharp08_templates_sz_100_handmade](https://github.com/user-attachments/assets/eae60f6c-8c35-4327-93ce-2cdc88e98f99)

Parameter setting for the Pattern Recognition paper:
subfolder = 'synthetic_blur_experiment' or 'real_blur_experiment';
full_img_name in [f"sharp{i:02}.JPG" for i in range(0, 15)];
downscale = 2;
blur_type = "square" or "T" or "square_padded" or "T_padded" depending on the experiment;
size_of_blur in np.arange(9, 100, 6);
selected_templates = img_name + "_100temp_positions.npy";
temp_sz = 100;
method = 'N1fold_invs' or 'N2fold_invs' or "RGB_crosscorr" or "gray_croscorr";
invs_comb in [('0', '1', '2', '10', '21'), ('0', '1', '2'), (['gray'])];
order = 5;
complex = False;
one_center = True;
norm = 2;
temp_normalization = False, but True for 'real_blur_experiment';
typennum = 1;
sigma = 0, but sigma = np.arange(0.0, 0.4001, 0.02) for testing the relative errors under noise

The code was expanded and made more general for IPTA paper. The differences in settings:
blur_type = "triangle" or "triangle_padded";
size_of_blur in np.arange(9, 64, 6) (the triangle blur is much stronger);
method = 'N2fold_invs' or "N3fold_invs";
complex = True;
norm = 1;
typennum = 0;

The experiment with recognizing of centrosymmetric objects is present in "recognition_of_centrosymmetric_objects" folder including the road signs images used in the paper. You can modify the two moment invariants used for feature space visualisation.
![road_signs_together](https://github.com/user-attachments/assets/62dc8cfd-189d-45ee-b824-ef4f97561e6e)
