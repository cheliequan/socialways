import argparse
import os

import numpy as np

from utils.parse_utils import BIWIParser, create_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a BIWI-format annotation file (obsmat.txt) to an NPZ dataset "
            "with observation/prediction splits."
        )
    )
    parser.add_argument(
        "--annot",
        "-a",
        required=False,
        help="Path to the BIWI obsmat.txt annotation file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="../data-8-12.npz",
        help="Output NPZ path (default: ../data-8-12.npz).",
    )
    parser.add_argument(
        "--obs-len",
        type=int,
        default=8,
        help="Number of observed timesteps per sample (default: 8).",
    )
    parser.add_argument(
        "--pred-len",
        type=int,
        default=12,
        help="Number of predicted timesteps per sample (default: 12).",
    )
    parser.add_argument(
        "--downsample",
        "-d",
        type=int,
        default=1,
        help="Keep one frame every N (default: 1, i.e., no downsampling).",
    )
    return parser


def main():
    parser = parse_args()
    args = parser.parse_args()

    if not args.annot:
        parser.error("--annot is required (path to obsmat.txt from the BIWI dataset)")

    if not os.path.exists(args.annot):
        parser.error(f"Annotation file not found: {args.annot}")

    data_parser = BIWIParser()
    data_parser.load(args.annot, down_sample=args.downsample)

    obs_len = args.obs_len
    pred_len = args.pred_len
    obsvs, preds, times, batches = create_dataset(
        data_parser.p_data,
        data_parser.t_data,
        range(
            data_parser.t_data[0][0],
            data_parser.t_data[-1][-1],
            data_parser.interval,
        ),
        obs_len,
        pred_len,
    )

    np.savez(args.output, obsvs=obsvs, preds=preds, times=times, batches=batches)
    print("Dataset was created successfully and stored in:", args.output)


if __name__ == "__main__":
    main()
