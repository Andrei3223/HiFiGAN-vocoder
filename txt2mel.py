import warnings

import torch
from hydra.utils import instantiate
import hydra
from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH
from pathlib import Path

from speechbrain.pretrained import Tacotron2
import torch
import torchaudio

warnings.filterwarnings("ignore", category=UserWarning)

def split_text_by_words(text, words_per_chunk=15):
    words = text.split()
    return [' '.join(words[i:i + words_per_chunk]) 
            for i in range(0, len(words), words_per_chunk)]

@hydra.main(version_base=None, config_path="src/configs", config_name="inference_text")
def main(config):
    
    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    dataset = instantiate(config.datasets)

    # build model architecture, then print to console
    print("Geting model")
    model = instantiate(config.model).to(device)
    checkpoint = torch.load(config.inferencer.from_pretrained, device)

    # load architecture params from checkpoint.
    if checkpoint["config"]["model"] != config["model"]:
        print(
            "Warning: Architecture configuration given in the config file is different from that "
            "of the checkpoint. This may yield an exception when state_dict is loaded."
        )
    model.load_state_dict(checkpoint["state_dict"])

    tacotron2 = Tacotron2.from_hparams(
        source="speechbrain/tts-tacotron2-ljspeech",
        savedir="tmpdir"
    )
    i = 0
    for text in dataset:
        # print(text)
        text_list = split_text_by_words(text, 15)
        mel_list = []
        for sub_str in text_list:
            mel_output = tacotron2.encode_text(sub_str)
            mel_list.append(mel_output[0])

        mel_spec = torch.cat(mel_list, dim=2)

        wav = model(mel_spec)
        wav = wav.squeeze(0).detach().cpu()

        out_path = ROOT_PATH / config.inferencer.out_path
        # print(out_path)
        if out_path.exists() is False:
            out_path.mkdir(exist_ok=True, parents=True)
        torchaudio.save(f'{config.inferencer.out_path}/from_text{i}.wav', wav, 22050)

        i += 1


if __name__ == "__main__":
    main()
