import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math
import numpy as np
import os
import io
import contextlib

from transformations import blur, add_noise
from load_image import load_n_process_image
from transformations import mixing_channels, rgb_to_pca, rgb_to_ycbcr, nearest_multiple_of_14

import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)


class DINOv2FeatureExtractor(nn.Module):
    """
    DINOv2 ViT-B/14 feature extractor that returns [B, C] (all patches in one feature vector) feature vector if s
    eparate_crops (i.e. DINOv2 processes each image crop separately and is not fed by the whole image)
    or [B, C, Ht, Wt] if not separate_crops (DINOv2 processes whole image)
    It takes features from a block mid_block_idx (default: block 9, the last one is 11).
    This mid-layer is typically better for matching than the very last layer.
    patch_norm=True performs channel-wise L2 normalization
    """
    def __init__(self, mid_block_idx: int = 9, tokens_mode='except_cls', separate_crops=True, patch_norm=True):
        super().__init__()
        # Loads official DINOv2 ViT-B/14 via torch.hub
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            self.vit = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad_(False)

        self.patch_size = 14
        self.mid_block_idx = mid_block_idx
        self.tokens_mode = tokens_mode
        self.separate_crops = separate_crops
        self.patch_norm = patch_norm

        self._hook_out = None

        # Register a forward hook on the chosen block to capture its output tokens.
        # The output tensor at a block is expected to be [B, N+1, C] (cls + patches).
        self._hook_handle = self.vit.blocks[self.mid_block_idx].register_forward_hook(self._save_hook)

    def _save_hook(self, module, inputs, output):
        # Save the tokens coming *out* of this transformer block
        self._hook_out = output

    def forward(self, img_bchw: torch.Tensor) -> torch.Tensor:
        """
        img_bchw: [B,3,H,W], already normalized.
        H and W are multiples of 14.
        Returns:
            fmap: [B, C, Ht, Wt], channel-wise L2-normalized.
        """
        img_bchw = img_bchw.to(dtype=torch.float32)
        B, _, H, W = img_bchw.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"H & W must be multiples of {self.patch_size}, got {H}x{W}"

        _ = self.vit(img_bchw)

        # Tokens after the selected block: [B, N+1, C] (CLS + patches)
        tokens = self._hook_out
        if tokens.dim() != 3:
            raise RuntimeError(f"Unexpected token shape from hook: {tuple(tokens.shape)} (expected [B, N+1, C])")

        if self.tokens_mode == 'except_cls':
            # Drop CLS, keep patch tokens
            patch_tokens = tokens[:, 1:, :]  # [B, N, C]
        elif self.tokens_mode == 'only_cls':
            # Keep only CLS, drop patch tokens
            patch_tokens = tokens[:, 0, :]
        elif self.tokens_mode == 'features_mean':
            patch_tokens = torch.mean(tokens[:, 1:, :], dim=1)
        else:
            # if self.tokens_mode == 'all', then we don't need to do anything
            patch_tokens = tokens

        if self.patch_norm:
            patch_tokens = F.normalize(patch_tokens, dim=-1)  # channel-wise L2 normalization

        if self.separate_crops:
            fmap = patch_tokens.view(B, -1)
        else:  # Reshape to [B, C, Ht, Wt]
            Ht, Wt = H // self.patch_size, W // self.patch_size
            if patch_tokens.shape[1] != Ht * Wt:
                raise RuntimeError(
                    f"Token count mismatch: expected {Ht*Wt}, got {patch_tokens.shape[1]}."
                )

            fmap = patch_tokens.view(B, Ht, Wt, -1).permute(0, 3, 1, 2).contiguous()  # [B, C, Ht, Wt]

        return fmap

    def __del__(self):
        try:
            if hasattr(self, "_hook_handle") and self._hook_handle is not None:
                self._hook_handle.remove()
        except Exception:
            pass


# Use standard ImageNet normalization (DINOv2 works with these stats)
preprocess = transforms.Compose([
    transforms.ToTensor(),  # [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


def extract_feature_map(model, img_pil):
    """ preprocess image and assign features """
    x = preprocess(img_pil).unsqueeze(0).to(DEVICE)  # [1,3,H,W]
    x = x.float()
    fmap = model(x)
    return fmap


def cosine_similarity(u, v):
    """
    returns scalar cosine sim
    """
    u_n = F.normalize(u, dim=-1)
    v_n = F.normalize(v, dim=-1)

    sim = torch.sum(v_n * u_n, dim=-1)
    return sim.detach().cpu().numpy()


def compute_img_features(search_img, model, Sw, Sh, Tw, Th, separate_crops, BATCH_SIZE, P, img_name, subfolder, features_save):
    # Sliding grid dimensions
    Nx = math.floor((Sw - Tw) / P) + 1
    Ny = math.floor((Sh - Th) / P) + 1

    # Prepare a small helper to convert a list of PIL imgs -> batched tensor on DEVICE
    def to_batch_tensor(pil_list):
        # Preprocess each crop, then stack
        tensors = [preprocess(im) for im in pil_list]  # list of [3,H,W]
        batch = torch.stack(tensors, dim=0).to(DEVICE)  # [B,3,H,W]
        return batch

    path_features = os.path.join('blur_n_channelmixing_invariants', 'computed_DINO_features', subfolder, img_name)
    if not os.path.exists(path_features):
        os.makedirs(path_features)

    if os.path.exists(os.path.join(path_features, features_save)):
        img_features = torch.load(os.path.join(path_features, features_save), map_location=DEVICE)
        return img_features

    if separate_crops:
        # a sample crop to get the right number of features for creating img_features torch array
        num_features = extract_feature_map(model, search_img[0:0 + Th, 0:0 + Tw]).shape[1]
        img_features = torch.empty((Ny, Nx, num_features), dtype=torch.float32, device=DEVICE)
        # Iterate rows; for each row, accumulate crops into batches and encode
        for yi in range(Ny):
            top = yi * P

            batch_pils = []
            batch_positions = []  # to map back similarities (xi) in this row

            for xi in range(Nx):
                left = xi * P
                crop = search_img[top:top + Th, left:left + Tw]  # PIL crop
                batch_pils.append(crop)
                batch_positions.append(xi)

                # If batch is full or we're at the end, run the model on the batch
                if len(batch_pils) == BATCH_SIZE or xi == Nx - 1:
                    batch = to_batch_tensor(batch_pils)  # [B,3,Th,Tw]
                    crops_vec = model(batch)  # [B,C,Ht,Wt]
                    # crops_vec = F.normalize(crops_vec.view(fmap_b.size(0), -1), dim=1)  # [B, D]

                    # Fill this segment of the sim_map row
                    for k, xi_k in enumerate(batch_positions):
                        img_features[yi, xi_k] = crops_vec[k]

                    # Reset batch buffers
                    batch_pils.clear()
                    batch_positions.clear()

        if len(features_save) > 0:
            torch.save(img_features, os.path.join(path_features, features_save))

    else:
        search_fmap = extract_feature_map(model, search_img)  # [1, C, Hs, Ws]
        Ht, Wt = int(Th / 14), int(Tw / 14)
        C = search_fmap.shape[1]
        windows = search_fmap.unfold(dimension=2, size=Ht, step=1)  # [1, C, Ny, Ws, Ht]
        windows = windows.unfold(dimension=3, size=Wt, step=1)  # [1, C, Ny, Nx, Ht, Wt]
        # Reorder to [1, Ny, Nx, C, Ht, Wt], then flatten last 3 dims:
        windows = windows.permute(0, 2, 3, 1, 4, 5).contiguous()  # [1, Ny, Nx, C, Ht, Wt]
        img_features = windows.view(1, Ny, Nx, C * Ht * Wt)  # [1, Ny, Nx, D]
        img_features = img_features.squeeze(0)  # [Ny, Nx, D] if B==1

        if len(features_save) > 0:
            torch.save(img_features, os.path.join(path_features, features_save))

    return img_features


def DINO_template_matching(model, img_features, template, P, temp_pos_y, temp_pos_x, separate_crops):
    # Template feature + descriptor
    template_vec = extract_feature_map(model, template)  # [1,C,Ht,Wt]
    if not separate_crops:
        template_vec = template_vec.view(1, -1) # [1, D]

    sim_map = cosine_similarity(template_vec, img_features)

    # Best match
    best_yi, best_xi = np.unravel_index(np.argmax(sim_map), sim_map.shape)
    best_score = float(sim_map[best_yi, best_xi])
    best_top = best_yi * P
    best_left = best_xi * P

    det_position = np.linalg.norm(np.array([temp_pos_y, temp_pos_x]) - np.array([best_top, best_left]))

    temp_result = [best_top, best_left, best_score, det_position]
    print(temp_result)

    return temp_result


if __name__ == "__main__":
    # Best setting for our template matching (experimentally tested): mid_block_idx=9, P=14, separate_crops=True,
    # tokens_mode='except_cls', patch_norm=True

    mid_block_idx = 9  # we take features from the mid_block_idx block of DINOv2 (the last one not very suitable for template matching)
    # subset of image crops for which we compute features, each crop aligned on a P-pixel grid. Relevant only
    # if separate_crops because otherwise P=14 is enforced by DINOv2 architecture
    P = 14
    # if True, DINOv2 computes features on independent crops. This is setting to avoid global context that would take
    # place if not separate crops and DINOv2 computes features on the whole image and then take crops by stride 14
    separate_crops = True
    # which features we want to compute. except_cls takes only patch features, only_cls takes only cls (no patch features),
    # features_mean takes only patch features and their mean (setting that allows us to set low P, e.g. P=1),
    # 'all' takes patch features together with cls token. if not separate_crops, then 'except_cls' automatically.
    tokens_mode = 'except_cls'  # only options: ['except_cls', 'all', 'only_cls', 'features_mean']. If something else, then automatically 'all'.
    # whether we want to do channel-wise L2 normalizing of features. True is recommended
    patch_norm = True

    # One of subfolders in images/templates folder. Right now either synthethic_blur_experiment or real_blur_experiment.
    # Or you can create your own with your images and template positions.
    subfolder = 'synthetic_blur_experiment'
    BATCH_SIZE = 2736  # inference goes through many crops in a batch at once to improve computation time
    temp_sz = 98  # template size
    downscale = 2  # multiple by which the sharp and blurred images will be downsampled
    sigma = 0
    # Either "square", "square_padded", "T", "T_padded", "triangle", "triangle_padded", "disk", "disk_padded",
    # or name of an image that is a blurred version of full_img_name
    blur_type = "square"

    # setting name variables for saving result and img_features
    if separate_crops:
        search_img_approach = 'separatecrops'
    else:
        search_img_approach = 'wholeimg'
    if patch_norm:
        norm_name = 'patchnorm'
    else:
        norm_name = 'nopatchnorm'

    if not separate_crops: # if whole image passed to DINO, P must be 14
        P = 14
        tokens_mode = 'except_cls'
    if tokens_mode == 'only_cls':
        P = 1

    # Init DINOv2 extractor
    model = DINOv2FeatureExtractor(mid_block_idx, tokens_mode, separate_crops, patch_norm).to(DEVICE).eval()

    for full_img_name in [f"sharp{i:02}.JPG" for i in range(0, 15)]:
        for size_of_blur in np.concatenate(([0], np.arange(9, 71, 6))):
            for a in [0, 0.2, 0.3]:

                if "." in blur_type:
                    size_of_blur = 0
                    a = 0

                print(f'img: {full_img_name}\nsize of blur: {size_of_blur}\na: {a}\nsigma: {sigma}\ntemplate size: {temp_sz}\n')
                print(f'{search_img_approach}\ntoken mode: {tokens_mode}\nblock: {mid_block_idx}\nP: {P}\npatch normalization: {patch_norm}\n')

                results = []
                img_name = os.path.splitext(full_img_name)[0]
                features_save = f"DINOv2_vitb14_{search_img_approach}_{tokens_mode}_{norm_name}_{img_name}_P{P:02}_block_{mid_block_idx}_temp_sz_{temp_sz}.pt"

                A = np.array([[1 - 2 * a, a, a], [a, 1 - 2 * a, a], [a, a, 1 - 2 * a]])
                selected_templates = img_name + "_100temp_positions.npy"
                temp_pos = np.load(os.path.join("templates", subfolder, selected_templates))

                search_img = load_n_process_image(full_img_name, subfolder, downscale, "_", "_")    # 2736x1824 or transposed
                Sh, Sw = search_img.shape[:2]
                pad_w = (14 - (Sw % 14)) % 14
                pad_h = (14 - (Sh % 14)) % 14
                search_img = np.pad(search_img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values=0)  # padded to 2744x1834, eventually 189x124 patches
                Sh, Sw = search_img.shape[:2]

                if '.' in blur_type:
                    transformed_img = load_n_process_image(blur_type, subfolder, downscale, "_", "_")
                    # transformed_img = rgb_to_pca(transformed_img)
                    # transformed_img = rgb_to_ycbcr(transformed_img)
                else:
                    transformed_img = blur(search_img, size=size_of_blur, btype="square")
                    transformed_img = add_noise(transformed_img, sigma)
                    transformed_img = mixing_channels(transformed_img, A)

                img_features = compute_img_features(search_img, model, Sw, Sh, temp_sz, temp_sz, separate_crops,
                                                    BATCH_SIZE, P, img_name, subfolder, features_save)

                for k in range(temp_pos.shape[0]):
                    i, j = temp_pos[k]
                    i_corrected, j_corrected = nearest_multiple_of_14(i, j)
                    template = transformed_img[i_corrected:i_corrected + temp_sz, j_corrected:j_corrected + temp_sz].copy()
                    Th, Tw = template.shape[:2]

                    print("template number", k)
                    results.append(DINO_template_matching(model, img_features, template, P, i_corrected, j_corrected, separate_crops))

                results = np.array(results)

                DP_directory = f'blur_n_channelmixing_invariants/detected_positions/{subfolder}/new_experiment/{img_name}/'
                if "." in blur_type:
                    image_n_blur = f'{img_name}_{os.path.splitext(blur_type)[0]}blur_temp_sz_{temp_sz:03d}_ycbcr'
                else:
                    image_n_blur = f'{img_name}_squareblur_size{size_of_blur:02d}_a_{a:.1f}_noise{sigma:.2f}'
                if not os.path.exists(DP_directory):
                    os.makedirs(DP_directory)
                np.save(f"{DP_directory}"
                        f"DP_{image_n_blur}_DINOv2_vitb14_{search_img_approach}_{tokens_mode}_{norm_name}_P{P:02}_block_"
                        f"{mid_block_idx}_BtempSimg.npy", results)
