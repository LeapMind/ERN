import hydra
from omegaconf import DictConfig

from efficiera.models.object_detection.training.pretrain.src.evaluator import evaluate


@hydra.main(config_path="configs", config_name="evaluation_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Main
    Args:
        cfg (DictConfig): A dicts defining parameter for training
    """
    evaluate(cfg)


if __name__ == "__main__":
    main()
