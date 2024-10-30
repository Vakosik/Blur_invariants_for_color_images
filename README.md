Blur moment invariants (both single-channel and cross-channel) are in "blur_invariants.py". For now, there are only invariants to blur with N-fold symmetry PSF and unconstrained blur.

All template matching experiments can run by "run_template_matching.py". Both padding and no_padding options are available as well as invariant and cross-correlation methods. All results from the paper can be replicated by running corresponding loops (images, method, size_of_blur/template_size). 
The used images and templates are saved in "images" and "templates" folders.

The experiment with recognizing of centrosymmetric objects is present in "recognition_of_centrosymmetric_objects" folder including the road signs images used in the paper. You can modify the two moment invariants used for feature space visualisation.
