import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    # dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)
    dataset = instantiate(config.datasets)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    # print(model)

    # # get metrics
    # metrics = {"inference": []}
    # for metric_config in config.metrics.get("inference", []):
    #     # use text_encoder in metrics
    #     metrics["inference"].append(
    #         instantiate(metric_config)
    #     )

    # save_path for model predictions
    save_path = Path(config.inferencer.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataset=dataset,
        save_path=save_path,
        metrics=None,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()



if __name__ == "__main__":
    main()
