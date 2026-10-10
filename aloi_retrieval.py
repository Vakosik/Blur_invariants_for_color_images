"""Retrieve degraded ALOI images against one sharp RGB reference per object.

Example (downloads and retains only the r0 PNG of each of the 1,000 objects):
    python aloi_retrieval.py --download --jobs 4

For an existing extracted ALOI collection:
    python aloi_retrieval.py --data-dir /path/to/aloi/png --jobs 4

Defaults follow the joint-invariant experiment in README.md: geometric moments,
order 5, N=2, a fixed image center, and L1 relative-error matching. Blur uses
the repo's same-size, zero-boundary convolution; pixels outside the frame are
discarded. Use --padding to retain the full convolution support instead.
Kernel sizes are in pixels AFTER optional downsampling (default: none).

Outputs: results.csv (24 grid points), retrievals.npz (per-query predictions,
distances, features, and grid axes), and settings.json (experiment metadata).
No degraded images are saved. --limit is only for smaller smoke experiments;
the default requires every object ID from 1 through 1000, with no omissions.

ALOI: https://aloi.science.uva.nl/aloi/download.html
Please cite Geusebroek, Burghouts, and Smeulders, IJCV 61(1), 103–112 (2005).
Dependencies are those used by the repo: numpy, scipy, opencv-python,
scikit-image, scikit-learn, joblib, tqdm.
"""

import argparse
import csv
import json
import tarfile
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm

from blur_invariants.blur_invariants import central_moments
from blur_n_channelmixing_invariants.blur_channelmixing_invariants import (
    blur_channel_mixing_invariants_3channels,
)
from detect_position import compare_invariants
from transformations import blur, mixing_channels


BLUR_SIZES = (0,) + tuple(range(9, 70, 12))
MIXING_VALUES = (0.0, 0.1, 0.2, 0.4)
C_INDICES = [[(0, 0)], [(1, 0)], [(1, 0)], [(0, 1)]]
N_OBJECTS = 1000


def download_sample(object_id, data_dir):
    """Stream an official object tar, stopping after its r0 PNG; save only it.

    ALOI serves lossless PNGs through archives, so preceding archive members
    are transferred but never written to disk. Existing samples are reused.
    """
    destination = data_dir / str(object_id) / f"{object_id}_r0.png"
    if destination.exists():
        return
    url = f"https://aloi.science.uva.nl/tars/png/{object_id}.tar"
    with urlopen(url, timeout=120) as response:
        with tarfile.open(fileobj=response, mode="r|") as archive:
            for member in archive:
                if member.isfile() and Path(member.name).name == destination.name:
                    with archive.extractfile(member) as sample:
                        content = sample.read()
                    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
                    if image is None or image.ndim != 3 or image.shape[2] != 3:
                        raise ValueError(f"Invalid RGB sample in {url}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(".png.part")
                    temporary.write_bytes(content)
                    temporary.replace(destination)
                    return
    raise FileNotFoundError(f"No {destination.name} in {url}")


def select_samples(data_dir, object_ids):
    """Require exactly one r0 PNG for each requested ID, even in a full archive."""
    wanted = {f"{object_id}_r0.png": object_id for object_id in object_ids}
    selected = {}
    for path in sorted(data_dir.rglob("*_r0.png")):
        object_id = wanted.get(path.name)
        if object_id is None:
            continue
        if object_id in selected:
            raise ValueError(f"Duplicate sample for object {object_id}: {selected[object_id]} and {path}")
        selected[object_id] = path
    missing = [object_id for object_id in object_ids if object_id not in selected]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} r0 PNGs under {data_dir}; first IDs: {missing[:10]}. "
            "Use --download or point --data-dir at your extracted color ALOI collection."
        )
    return [selected[object_id] for object_id in object_ids]


def image_descriptor(image, order):
    """Use the repo's joint invariants and fixed-center moment convention."""
    tx, ty = image.shape[0] / 2, image.shape[1] / 2
    moments = np.stack([
        central_moments(image[:, :, channel], r=order, tx=tx, ty=ty)
        for channel in range(3)
    ], axis=-1)
    # The existing invariant routine replaces singular denominators with ones.
    # Mark those descriptors invalid so they cannot produce spurious matches.
    denominator = np.linalg.det(moments[[1, 0, 0], [0, 1, 0], :])
    if denominator == 0 or not np.isfinite(denominator):
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        features, _ = blur_channel_mixing_invariants_3channels(
            moments, C_INDICES, order, N=2, typec=1, typex=0
        )
    return features if np.isfinite(features).all() else None


def object_descriptors(path, order, downscale, padding):
    """Load one object, compute its reference and all 24 query descriptors."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected an 8-bit RGB PNG: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64) / 255
    if downscale != 1:
        if min(image.shape[:2]) < downscale:
            raise ValueError(f"Downscale factor exceeds image dimensions: {path}")
        image = cv2.resize(image, (0, 0), fx=1 / downscale, fy=1 / downscale,
                           interpolation=cv2.INTER_AREA)
    if padding:
        margin = max(BLUR_SIZES) // 2
        image = np.pad(image, ((margin, margin), (margin, margin), (0, 0)))
    reference = image_descriptor(image, order)
    queries = []
    for size in BLUR_SIZES:
        blurred = blur(image, size=size, btype="square")
        for a in MIXING_VALUES:
            matrix = np.array([[1 - 2*a, a, a], [a, 1 - 2*a, a], [a, a, 1 - 2*a]])
            queries.append(image_descriptor(mixing_channels(blurred, matrix), order))
    return reference, queries


def run_experiment(paths, object_ids, order=5, downscale=1, padding=False, jobs=1, norm=1):
    """Return grid statistics and predictions; invalid queries count as failures."""
    with parallel_config(backend="loky", inner_max_num_threads=1):
        descriptors = Parallel(n_jobs=jobs)(
            delayed(object_descriptors)(path, order, downscale, padding)
            for path in tqdm(paths, desc="Object descriptors")
        )
    exemplar = next((ref for ref, _ in descriptors if ref is not None), None)
    if exemplar is None:
        raise ValueError("All sharp references have singular or non-finite descriptors.")
    invalid = np.full_like(exemplar, np.nan)
    references = np.stack([ref if ref is not None else invalid for ref, _ in descriptors])
    queries = np.array([[q if q is not None else invalid for q in qs] for _, qs in descriptors])
    queries = queries.reshape(len(paths), len(BLUR_SIZES), len(MIXING_VALUES), -1)
    valid_gallery = np.isfinite(references).all(axis=1)
    gallery_ids = object_ids[valid_gallery]
    gallery = references[valid_gallery, None, :]
    shape = (len(BLUR_SIZES), len(MIXING_VALUES), len(paths))
    predictions = np.full(shape, -1, dtype=int)
    distances = np.full(shape, np.inf)
    for b in range(len(BLUR_SIZES)):
        for m in range(len(MIXING_VALUES)):
            for q in range(len(paths)):
                descriptor = queries[q, b, m]
                if np.isfinite(descriptor).all():
                    # compare_invariants mutates zeros; give it a fresh copy.
                    # Sharp references are the denominator, as in template matching.
                    index, _, distance = compare_invariants(gallery.copy(), descriptor, norm)
                    if np.isfinite(distance):
                        predictions[b, m, q] = gallery_ids[index]
                        distances[b, m, q] = distance
    correct = np.sum(predictions == object_ids[None, None, :], axis=-1)
    return dict(
        correct=correct, percent=100 * correct / len(paths),
        predictions=predictions, distances=distances, object_ids=object_ids,
        reference_features=references, query_features=queries,
        blur_sizes=np.array(BLUR_SIZES), mixing_values=np.array(MIXING_VALUES),
        invalid_references=np.count_nonzero(~valid_gallery),
        invalid_queries=np.sum(~np.isfinite(queries).all(axis=-1), axis=0),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("images/aloi/png"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/aloi_retrieval"))
    parser.add_argument("--download", action="store_true", help="Download missing r0 PNGs only")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel object workers; -1 uses all CPUs")
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--norm", type=int, choices=(1, 2), default=1)
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument("--padding", action="store_true", help="Zero-pad both references and queries by 34 pixels")
    parser.add_argument("--limit", type=int, default=N_OBJECTS, help="First N objects; default 1000")
    args = parser.parse_args()
    if not 1 <= args.limit <= N_OBJECTS or args.order < 2 or args.downscale < 1 or args.jobs == 0:
        parser.error("Require 1 <= limit <= 1000, order >= 2, downscale >= 1, and jobs != 0")
    object_ids = np.arange(1, args.limit + 1)
    if args.download:
        for object_id in tqdm(object_ids, desc="Download r0 samples"):
            download_sample(object_id, args.data_dir)
    paths = select_samples(args.data_dir, object_ids)
    print(f"Retrieving {len(paths)} degraded queries per grid point against {len(paths)} sharp references.")
    results = run_experiment(paths, object_ids, args.order, args.downscale, args.padding, args.jobs, args.norm)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["blur_size", "a", "correct", "total", "percent", "invalid_queries"])
        for b, size in enumerate(BLUR_SIZES):
            for m, a in enumerate(MIXING_VALUES):
                count, percent = results["correct"][b, m], results["percent"][b, m]
                writer.writerow([size, a, count, len(paths), percent, results["invalid_queries"][b, m]])
                print(f"blur={size:2d}x{size:<2d}  a={a:.1f}  {count:4d}/{len(paths)}  {percent:6.2f}%")
    np.savez_compressed(args.output_dir / "retrievals.npz", **results)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(
        view="r0", method="joint blur/channel-mixing invariants", N=2, complex=False,
        one_center=False, C_indices=C_INDICES, blur_sizes=BLUR_SIZES,
        mixing_values=MIXING_VALUES, boundary="zero", convolution="same",
        matching="relative error with sharp-reference denominator; first object ID wins exact ties",
        invalid_references=int(results["invalid_references"]),
        samples=[str(path.resolve()) for path in paths],
    )
    (args.output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Invalid references: {results['invalid_references']}; invalid queries count as failures.")
    print(f"Results saved in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
