from pathlib import Path

import pandas as pd
import torch
from numpy import inf
from random import shuffle

from src.logger.utils import plot_spectrogram
from src.datasets.data_utils import inf_loop
from src.utils.io_utils import ROOT_PATH
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.transforms.mel_spec import MelSpectrogram
from src.loss import GeneratorAdvLoss, DiscriminatorAdvLoss, FeatureMapLoss, MelSpecLoss


from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def __init__(
        self,
        model,
        metrics,
        optimizer_d,
        optimizer_g,
        lr_scheduler_d,
        lr_scheduler_g,
        lambda_mel_loss,
        lambda_fm_loss,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model
        self.criterion_discriminator = DiscriminatorAdvLoss()
        self.criterion_generator = GeneratorAdvLoss()
        self.criterion_feat_map = FeatureMapLoss()
        self.criterion_mel = MelSpecLoss()
        self.optimizer_d = optimizer_d
        self.optimizer_g = optimizer_g
        self.lr_scheduler_d = lr_scheduler_d
        self.lr_scheduler_g = lr_scheduler_g

        self.lambda_mel_loss = lambda_mel_loss
        self.lambda_fm_loss = lambda_fm_loss
        # self.text_encoder = text_encoder
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # configuration to monitor model performance and save best

        self.save_period = (
            self.cfg_trainer.save_period
        )  # checkpoint each save_period epochs
        self.monitor = self.cfg_trainer.get(
            "monitor", "off"
        )  # format: "mnt_mode mnt_metric"

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )

        # define checkpoint dir and init everything if required

        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

        self.get_mel_spec = MelSpectrogram().to(self.device)

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        wav = batch["audio"]
        mel = batch["spectrogram"]
        # print(wav.shape)
        wav_gen = self.model.generator(mel).squeeze(1)
        # print("gen:", wav_gen.shape)
        mel_gen = self.get_mel_spec(wav_gen)  # .squeeze(1)
        # print("mel", mel.shape, mel_gen.shape)
        self.optimizer_d.zero_grad()

        # print(wav.is_cuda, mel.is_cuda, wav_gen.detach().is_cuda)
        mpd_outs, mpd_gen_outs_gen, _, _ = self.model.mpd(wav, wav_gen.detach())
        msd_outs, msd_gen_outs_gen, _, _ = self.model.msd(wav, wav_gen.detach())

        batch.update(self.criterion_discriminator(mpd_outs, mpd_gen_outs_gen, "mpd"))
        batch.update(self.criterion_discriminator(msd_outs, msd_gen_outs_gen, "msd"))

        discriminator_loss = batch["discriminator_loss_mpd"] + batch["discriminator_loss_msd"]

        if self.is_train:
            discriminator_loss.backward()
            self._clip_grad_norm_mpd()
            self._clip_grad_norm_msd()
            self.optimizer_d.step()

        # Generator
        self.optimizer_g.zero_grad()

        _, mpd_gen_outs, mpd_feat_maps, mpd_gen_feat_maps = self.model.mpd(
            wav, wav_gen
        )
        _, msd_gen_outs, msd_feat_maps, msd_gen_feat_maps = self.model.msd(
            wav, wav_gen
        )
        # print("mpd outs", len(mpd_outs), mpd_outs[0][0].shape, len(mpd_gen_outs), mpd_gen_outs[0][0].shape)
        # print("mpd feats", len(mpd_feat_maps), mpd_feat_maps[0][0].shape, len(mpd_gen_feat_maps), mpd_gen_feat_maps[0][0].shape)
        batch.update(self.criterion_feat_map(mpd_feat_maps, mpd_gen_feat_maps, "mpd"))
        # print("msd outs", len(msd_outs), msd_outs[0][0].shape, len(msd_gen_outs), msd_gen_outs[0][0].shape)
        # print("msd feats", len(msd_feat_maps), msd_feat_maps[0][0].shape, len(msd_gen_feat_maps), msd_gen_feat_maps[0][0].shape)
        batch.update(self.criterion_feat_map(msd_feat_maps, msd_gen_feat_maps, "msd"))

        batch.update(self.criterion_generator(mpd_gen_outs, "mpd"))
        batch.update(self.criterion_generator(msd_gen_outs, "msd"))

        batch.update(self.criterion_mel(mel, mel_gen))

        generator_loss = (
            batch["mel_loss"] * self.lambda_mel_loss
            + batch["feat_loss_mpd"] * self.lambda_fm_loss
            + batch["feat_loss_msd"] * self.lambda_fm_loss
            + batch["generator_loss_mpd"]
            + batch["generator_loss_msd"]
        )

        if self.is_train:
            generator_loss.backward()
            self._clip_grad_norm_generator()
            self.optimizer_g.step()

            self.lr_scheduler_g.step()

        batch["wav_output"] = wav_gen
        batch["generator_loss"] = generator_loss
        batch["discriminator_loss"] = discriminator_loss
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        return batch

    def _train_epoch(self, epoch):
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise e

            self.train_metrics.update("grad_norm", self._get_grad_norm())

            # log current results
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} generator loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["generator_loss"].item(),
                    )
                )
                self.writer.add_scalar(
                    "learning rate descriminator", self.lr_scheduler_d.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "learning rate generator", self.lr_scheduler_g.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break

        logs = last_train_metrics

        # Run val/test
        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_audio(batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    # def log_predictions(self, wav_output, is_train=True, limit=5, **kwargs):
    #     for i, wav in enumerate(wav_output[:limit]):
    #         self._log_audio(wav.squeeze(0), sr=22050, name=str(i + 1))
    
    def log_audio(self, batch, examples_to_log=8, **kwargs):
        result = {}
        examples_to_log = min(examples_to_log, batch['wav_output'].shape[0])

        tuples = list(batch['wav_output'])
        tuples_input = list(batch['audio'])

        idx = 0
        for input, out in zip(tuples_input, tuples[:examples_to_log]):
            result[idx] = {
                "wav_genereted": self.writer.wandb.Audio(out.squeeze(0).detach().cpu().numpy(), sample_rate=22050),
                "wav_input": self.writer.wandb.Audio(input.squeeze(0).detach().cpu().numpy(), sample_rate=22050)
            }
            idx += 1
        self.writer.add_table("audio", pd.DataFrame.from_dict(result, orient="index"))
    
    def _clip_grad_norm_mpd(self):
        if self.config["trainer"].get("max_grad_norm_mpd", None) is not None:
            clip_grad_norm_(
                self.model.mpd.parameters(), self.config["trainer"]["max_grad_norm_mpd"]
            )
    
    def _clip_grad_norm_msd(self):
        if self.config["trainer"].get("max_grad_norm_msd", None) is not None:
            clip_grad_norm_(
                self.model.msd.parameters(), self.config["trainer"]["max_grad_norm_msd"]
            )
    
    def _clip_grad_norm_generator(self):
        if self.config["trainer"].get("max_grad_norm_generator", None) is not None:
            clip_grad_norm_(
                self.model.generator.parameters(), self.config["trainer"]["max_grad_norm_generator"]
            )
