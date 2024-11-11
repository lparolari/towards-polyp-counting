import argparse

from polypsense.mve.engine import train


def main():
    parser = get_parser()
    args = parser.parse_args()
    train(args)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--s", type=int, required=True, help="Tracklet length")
    parser.add_argument("--im_size", type=int, required=True)
    parser.add_argument("--sfe_ckpt", type=str, required=True)
    parser.add_argument("--sfe_freeze", action="store_true", default=None)

    parser.add_argument("--lr", type=float, default=None)

    parser.add_argument("--aug_vflip", action="store_true")
    parser.add_argument("--aug_hflip", action="store_true")
    parser.add_argument("--aug_affine", action="store_true")
    parser.add_argument("--aug_colorjitter", action="store_true")

    parser.add_argument("--exp_id", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--train_ratio", type=float, default=None)

    parser.add_argument(
        "--alpha_scheduler", type=str, default=None, choices=["constant", "epoch"]
    )
    parser.add_argument("--alpha_max_epochs", type=float, default=None)
    parser.add_argument("--alpha_initial_value", type=float, default=None)
    parser.add_argument("--alpha_min_value", type=float, default=None)

    return parser


if __name__ == "__main__":
    main()
