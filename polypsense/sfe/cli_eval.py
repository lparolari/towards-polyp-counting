import argparse

from polypsense.sfe.engine import eval


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="realcolon",
        choices=["realcolon", "cifar10"],
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--im_size", type=int, default=232)
    parser.add_argument(
        "--contrastive_strategy",
        default="temporal",
        choices=["temporal", "augmentation"],
    )
    parser.add_argument(
        "--view_generator",
        type=str,
        default="gaussian",
        choices=["gaussian", "uniform"],
    )
    parser.add_argument("--view_generator_gaussian_std", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_id", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--exp_notes", type=str, default=None)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    eval(args)


if __name__ == "__main__":
    main()
