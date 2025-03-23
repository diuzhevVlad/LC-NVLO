import numpy as np
import argparse
import os
from odom_utils import get_diff_transform, transform_to_pq
import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--poses-dir",
        required=True,
        help="Directory with poses .txt files in format <sequence_name>.txt containing 12 numbers in each row (spatial transform params)",
        type=str,
    )
    parser.add_argument(
        "--sequences-dir", required=True, help="Directory with all sequences", type=str
    )
    parser.add_argument(
        "--filename", required=True, help="Difference poses filename", type=str
    )
    args = parser.parse_args()

    seq_pose_filenames = os.listdir(args.poses_dir)
    seqs = os.listdir(args.sequences_dir)

    it = tqdm.tqdm(sorted(seq_pose_filenames))
    for _seq_pose_filename in it:
        it.set_description(f"Processing: {_seq_pose_filename}")
        _seq_pose = os.path.splitext(_seq_pose_filename)[0]
        if _seq_pose not in seqs:
            raise RuntimeError(f"There is no sequence '{_seq_pose}'!")
        transforms = np.loadtxt(
            os.path.join(args.poses_dir, _seq_pose_filename)
        ).reshape((-1, 3, 4))
        diff_pqs = []
        for i in range(len(transforms) - 1):
            diff_pqs.append(
                transform_to_pq(
                    get_diff_transform(
                        np.vstack([transforms[i], np.array([[0, 0, 0, 1]])]),
                        np.vstack([transforms[i + 1], np.array([[0, 0, 0, 1]])]),
                    )
                )
            )
        np.save(os.path.join(args.sequences_dir, _seq_pose, args.filename), diff_pqs)
