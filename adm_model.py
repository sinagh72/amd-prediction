from torch.autograd import Variable
# from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUROC
import pytorch_lightning as pl
import torch
import torch.optim as optim
import numpy as np


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super(CosineWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class AMDModel(pl.LightningModule):
    def __init__(self, embed_dim,
                 model_dim, num_classes,
                 num_heads, num_layers,
                 lr, warmup, max_iters,
                 dropout=0.0, input_dropout=0.0,
                 loss_func=None):
        super(AMDModel, self).__init__()
        self.save_hyperparameters()
        self._create_model()
        self.loss_func = loss_func
        # metrics = MetricCollection([ConfusionMatrix(num_classes=3),
        #                             Accuracy(num_classes=3, mdmc_reduce="samplewise"),
        #                             Precision(num_classes=3, mdmc_reduce="samplewise"),
        #                             Recall(num_classes=3, mdmc_reduce="samplewise")])

        metrics = MetricCollection([ConfusionMatrix(num_classes=3),
                                    Accuracy(num_classes=3, average="weighted",
                                             mdmc_reduce="global")])

        # metrics = MetricCollection([Accuracy(num_classes=3, average="weighted",
        #                                      mdmc_reduce="global")])

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def _create_model(self):
        self.model = torch.nn.Transformer(d_model=self.hparams.d_model,
                                          nhead=self.hparams.nhead,
                                          num_encoder_layers=self.hparams.num_encoder_layers,
                                          num_decoder_layers=self.hparams.num_decoder_layers,
                                          dim_feedforward=self.hparams.dim_feedforward,
                                          dropout=self.hparams.dropout,
                                          activation=self.hparams.activation,
                                          custom_encoder=self.hparams.custom_encoder,
                                          custom_decoder=self.hparams.custom_decoder,
                                          layer_norm_eps=self.hparams.layer_norm_eps,
                                          batch_first=self.hparams.batch_first,
                                          norm_first=self.hparams.norm_first)

    def forward(self, x):
        self.model.forward(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def shared_step(self, batch, stage):
        x = batch["visit_seq"]
        y_true = batch["label"]

        y_pred = self.forward(x)

        loss = self.loss_fn(y_true, y_pred)


        tp, fp, fn, tn = smp.metrics.get_stats(o.long(), mask.long(), mode="multiclass")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")
