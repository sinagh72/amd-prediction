import pytorch_lightning as pl
import torch
import torch.optim as optim
from torchmetrics import MetricCollection, F1Score, Accuracy, Precision, Recall, MeanSquaredError, AUROC
from torch import nn, Tensor
from transformer import CosineWarmupScheduler, TransformerEncoders, PositionalEncoding
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class AMDModel(pl.LightningModule):
    def __init__(self,
                 embed_dim,
                 model_dim,
                 num_classes,
                 num_heads,
                 num_layers,
                 seq_len,
                 lr,
                 warmup,
                 max_iters,
                 dropout=0.0,
                 embedding_dropout=0.0,
                 loss_func=None):
        """
        Inputs:
            embed_dim - dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout - Dropout to apply inside the model
            embedding_dropout - Dropout to apply on the input features
            loss_func - Loss function
        """
        super(AMDModel, self).__init__()
        self.save_hyperparameters(ignore=['loss_func'])
        self._create_model()
        self.loss_func = loss_func
        metrics = MetricCollection([Recall(num_classes=3,
                                           mdmc_average='global',
                                           average='macro'
                                           ),
                                    Precision(num_classes=3,
                                              mdmc_average='global',
                                              average='macro'
                                              ),
                                    Accuracy(num_classes=3,
                                             mdmc_average='global',
                                             average="macro"
                                             )])
        f1 = F1Score(num_classes=3, average='macro', mdmc_average='global')
        # metrics = MetricCollection([Accuracy(num_classes=3, average="weighted",
        #  mdmc_reduce="global")])

        self.train_metrics = metrics.clone()
        self.val_metrics = metrics.clone()
        self.test_metrics = metrics.clone()

        self.train_f1 = f1.clone()
        self.val_f1 = f1.clone()
        self.test_f1 = f1.clone()

    def _create_model(self):
        # convert embed dim (input dim) to model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.embedding_dropout),
            nn.Linear(self.hparams.embed_dim, self.hparams.model_dim)
        )
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim,
                                                      max_len=self.hparams.seq_len)
        # Transformer
        self.transformer = TransformerEncoders(num_layers=self.hparams.num_layers,
                                               embed_dim=self.hparams.model_dim,
                                               dim_feedforward=2048,
                                               num_heads=self.hparams.num_heads,
                                               dropout=self.hparams.dropout)
        # Output classifier per sequence element
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes),
            # nn.Flatten(0,1)
            nn.Softmax(dim=-1)
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

    def shared_step(self, batch):
        x = batch["visit_seq"]
        y_true = batch["label"]
        y_true_hot = batch["label_hot"]

        y_pred = self.forward(x)
        # print(y_true.shape)
        # print(y_true_hot.shape)
        # convert (batch_size, seq_len,  classes) to (batch_size, classes, seq_len)
        # in another words convert (N, d1,..,dk, C) to (N, C, d1,..,dk)
        y = torch.permute(y_pred, (0, 2, 1))

        loss = self.loss_func(y, y_true)

        return loss, y, y_true

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
        loss, pred, target = self.shared_step(batch)
        metrics = self.train_metrics(pred, target)
        f1 = self.train_f1(pred, target)

        # self.train_metrics.update(pred, target)
        # metrics = self.train_metrics.compute()
        log = {'loss': loss, 'train': metrics, 'train_f1': f1}
        self.log_dict(log, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # def training_epoch_end(self, outputs):
    #     metric = self.train_metrics.compute()
    #     loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     log = {"train_loss": loss, 'train_': metric}
    #     self.train_metrics.reset()
    #     self.log_dict(log, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss, pred, target = self.shared_step(batch)
        metrics = self.val_metrics(pred, target)
        f1 = self.val_f1(pred, target)

        log = {"val_loss": loss, 'val': metrics, 'val_f1': f1}
        self.log_dict(log, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # def validation_epoch_end(self, outputs):
    #     metric = self.val_metrics.compute()
    #     loss = torch.stack([x["loss_val"] for x in outputs]).mean()
    #     log = {"val_loss": loss, 'val_': metric}
    #     self.log_dict(log, prog_bar=True)
    #     self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        loss, pred, target = self.shared_step(batch)
        # metrics = self.test_metrics(pred, target)
        metrics = self.test_metrics(pred, target)
        # f1 = self.test_f1(pred, target)
        f1 = self.test_f1(pred, target)

        log = {"test_loss": loss, 'test_': metrics, 'test_f1': f1}
        self.log_dict(log, prog_bar=True, on_step=False, on_epoch=True)
        return loss
        # self.log_dict(log, on_step=True, on_epoch=True, prog_bar=True)
        # return loss

    # def test_epoch_end(self, outputs):
    #     metric = self.test_metrics.compute()
    #     f1 = self.test_f1.compute()

    #     loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     log = {"test_loss": loss, 'test_': metric, 'test_f1': f1}
    #     self.log_dict(log,  on_step=False, on_epoch=True,prog_bar=True)
    #     self.test_metrics.reset()
    #     self.test_f1.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    # def test_epoch_end(self, outputs):
    #     return self.shared_epoch_end(outputs, "test")


class TransformerModel(pl.LightningModule):

    def __init__(self,
                 d_input: int,  # dimension of each input embedding
                 d_model: int,  # dimension of each embedding
                 nhead: int,  # number of heads in nn.MultiheadAttention
                 d_hid: int,  # dimension of the feedforward network model in nn.TransformerEncoder
                 nlayers: int,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                 loss_func,
                 dropout: float = 0.5,
                 num_classes: int = 3,
                 lr: float = 1e-4,
                 warmup: int = 50,
                 max_iters: int = 500):
        super(TransformerModel, self).__init__()
        self.save_hyperparameters(ignore=['loss_func'])

        self.model_type = 'Transformer'
        # convert input dim  to model dim
        self.input_net = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_input, d_model)
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.output_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
            # nn.Flatten(0,1)
            nn.Softmax(dim=-1)
        )

        # self.init_weights()

        self.loss_func = loss_func
        metrics = MetricCollection([Recall(num_classes=3,
                                           mdmc_average='global',
                                           average='macro',
                                           ignore_index=2
                                           ),

                                    Precision(num_classes=3,
                                              mdmc_average='global',
                                              average='macro',
                                              ignore_index=2
                                              ),
                                    Accuracy(num_classes=3,
                                             mdmc_average='global',
                                             average="macro"
                                             )])
        f1 = F1Score(num_classes=3, average='macro', mdmc_average='global',
                     ignore_index=2)

        self.train_metrics = metrics.clone()
        self.val_metrics = metrics.clone()
        self.test_metrics = metrics.clone()

        self.train_f1 = f1.clone()
        self.val_f1 = f1.clone()
        self.test_f1 = f1.clone()

    # def init_weights(self) -> None:
    #     initrange = 0.1
    #     self.output_net.bias.data.zero_()
    #     self.output_net.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, mask=None) -> Tensor:
        x = self.input_net(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, mask)
        output = self.output_net(x)
        return output

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

    def shared_step(self, batch):
        x = batch["visit_seq"]
        y_true = batch["label"]
        y_true_hot = batch["label_hot"]

        y_pred = self.forward(x)
        # print(y_true.shape)
        # print(y_true_hot.shape)
        # convert (batch_size, seq_len,  classes) to (batch_size, classes, seq_len)
        # in another words convert (N, d1,..,dk, C) to (N, C, d1,..,dk)
        y = torch.permute(y_pred, (0, 2, 1))

        loss = self.loss_func(y, y_true)

        return loss, y, y_true

    def training_step(self, batch, batch_idx):
        loss, pred, target = self.shared_step(batch)
        metrics = self.train_metrics(pred, target)
        f1 = self.train_f1(pred, target)

        # self.train_metrics.update(pred, target)
        # metrics = self.train_metrics.compute()
        log = {'loss': loss, 'train': metrics, 'train_f1': f1}
        self.log_dict(log, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, target = self.shared_step(batch)
        metrics = self.val_metrics(pred, target)
        f1 = self.val_f1(pred, target)

        log = {"val_loss": loss, 'val': metrics, 'val_f1': f1}
        self.log_dict(log, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, pred, target = self.shared_step(batch)
        metrics = self.test_metrics(pred, target)
        f1 = self.test_f1(pred, target)

        log = {"test_loss": loss, 'test_': metrics, 'test_f1': f1}
        self.log_dict(log, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
