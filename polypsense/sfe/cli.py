import argparse

from polypsense.sfe.engine import train


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="realcolon",
        choices=["realcolon", "cifar10"],
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--n_views", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--backbone_arch", type=str, default="resnet50")
    parser.add_argument(
        "--backbone_weights", default=None, choices=["IMAGENET1K_V1", "IMAGENET1K_V2"]
    )
    parser.add_argument("--backbone_out_dim", type=int, default=128)
    parser.add_argument("--im_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--aug_hflip", action="store_true")
    parser.add_argument("--aug_vflip", action="store_true")
    parser.add_argument("--aug_affine", action="store_true")
    parser.add_argument("--aug_colorjitter", action="store_true")
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
    parser.add_argument("--view_generator_uniform_window", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_id", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--exp_notes", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
