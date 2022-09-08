from torch.autograd import Variable
# from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUROC
import pytorch_lightning as pl
import torch
import torch.optim as optim
import numpy as np
from torchmetrics import MetricCollection
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, MeanSquaredError
import torch.nn as nn

from transformer import CosineWarmupScheduler, TransformerEncoder, PositionalEncoding


class AMDModel(pl.LightningModule):
    def __init__(self, embed_dim,
                 model_dim, num_classes,
                 num_heads, num_layers,
                 lr, warmup, max_iters,
                 dropout=0.0, input_dropout=0.0,
                 loss_func=None):
        """
        Inputs:
            embed_dim - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
        """
        super(AMDModel, self).__init__()
        self.save_hyperparameters()
        self._create_model()
        self.loss_func = loss_func
        # metrics = MetricCollection([ConfusionMatrix(num_classes=3),
        #                             Accuracy(num_classes=3, mdmc_reduce="samplewise"),
        #                             Precision(num_classes=3, mdmc_reduce="samplewise"),
        #                             Recall(num_classes=3, mdmc_reduce="samplewise")])

        metrics = MetricCollection([Precision(num_classes=3, average='weighted', ignore_index=2),
                                    Recall(num_classes=3, average='weighted', ignore_index=2),
                                    Accuracy(num_classes=3, average="weighted", ignore_index=2)])

        # metrics = MetricCollection([Accuracy(num_classes=3, average="weighted",
        #                                      mdmc_reduce="global")])

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.embed_dim, self.hparams.model_dim)
        )
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim, dropout=self.hparams.dropout)
        # Transformer
        self.transformer = TransformerEncoder(num_layers=self.hparams.num_layers,
                                              embed_dim=self.hparams.model_dim,
                                              dim_feedforward=2 * self.hparams.model_dim,
                                              num_heads=self.hparams.num_heads,
                                              dropout=self.hparams.dropout)
        # Output classifier per sequence element
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes),
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """
        Function for extracting the attention matrices of the whole Transformer
        for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

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
        y_true_hot = batch["label_hot"]

        y_pred = self.forward(x)

        loss = self.loss_fn(y_true, y_pred)

        if stage == "train":
            output = self.train_metrics(y_pred, y_true)
        elif stage == "valid":
            output = self.valid_metrics(y_pred, y_true)
        else:
            output = self.test_metrics(y_pred, y_true)

        log = {'loss': loss, f'{stage}_': output}

        self.log_dict(log, prog_bar=True)

        # return

    # def shared_epoch_end(self, outputs, stage):
    #     # aggregate step metics
    #     tp = torch.cat([x["tp"] for x in outputs])
    #     fp = torch.cat([x["fp"] for x in outputs])
    #     fn = torch.cat([x["fn"] for x in outputs])
    #     tn = torch.cat([x["tn"] for x in outputs])
    #
    #     self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    # def training_epoch_end(self, outputs):
    #     return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    # def validation_epoch_end(self, outputs):
    #     return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    # def test_epoch_end(self, outputs):
    #     return self.shared_epoch_end(outputs, "test")
