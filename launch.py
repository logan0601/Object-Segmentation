import argparse
import os
import logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0", help="GPU(s) to be used")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")

    parser.add_argument(
        "--resume", action="store_true", help="if true, resume from previous ckpt"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )
    args, extras = parser.parse_known_args()

    import segment
    from segment.utils.process import setup
    setup()

    logger = logging.getLogger("object_segmentation")
    logger.setLevel(logging.INFO)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs("outputs/", exist_ok=True)

    # 2D detect
    from segment.data.base import BaseDataModule
    from segment.models.base import BaseSystem
    dm: BaseDataModule = segment.find("image-datamodule")()
    module: BaseSystem = segment.find("2d-detector")(dm, resume=args.resume)
    # module.fit()
    # module.restore(name="model_final.pth")
    # module.inference()
    # module.visualize()

    # 3D detect
    fm = BaseDataModule = segment.find("frustum-datamodule")("apply")
    system: BaseSystem = segment.find("3d-segmentor-v2")(fm, resume=False)
    system.restore("model_00089.pth")
    # system.fit()
    # system.evaluate()
    # system.inference()
    system.fit_pose()


if __name__ == "__main__":
    main()
