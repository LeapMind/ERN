import hydra
from omegaconf import DictConfig

from efficiera.models.object_detection.training.pretrain.src.trainer import train


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Main
    Args:
        cfg (DictConfig): A dicts defining parameter for training
    """
    train(cfg)


if __name__ == "__main__":
    main()
