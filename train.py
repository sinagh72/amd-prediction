import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import torch
import torch.nn.functional as F
import torch.utils.data as data
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
from adm_dataset import AMDDataset
from adm_model import AMDModel
from data_prepration import  load_data, preprocess
import numpy as np
from focal_loss import FocalLoss

# load data
df_harbor, df_miami = load_data()
# set the folds and months
folds = [1, 2, 3, 4, 5]
mon = [3, 6, 9, 12, 15, 18, 21]
# preprocessing and retrieving data
x_train, y_train, seq_train, x_val, y_val, seq_val, x_test, y_test, seq_test = preprocess(df_harbor, df_miami, mon[0],
                                                                                          folds[0])
# max visit among training and validation data
slen = max(max(seq_train), max(seq_val))
print('Slen: ' + str(slen))
# sample of data
idx = 0
visit_num = seq_train[idx]
print('#visits:', visit_num)
ind = int(np.sum(seq_train[:idx]))
print('summ:', ind)
visit_seq = x_train[ind:ind + visit_num]  # get the seq of visits for patient idx
output_seq = y_train[ind:ind + visit_num]  # get the outcome of visit seq for patient idx
print('visit seq:', len(visit_seq))
print('output seq:', len(output_seq))
print()

# creating data set for train, valid, test, prediction
train_dataset = AMDDataset(x_train, y_train, slen)
val_dataset = AMDDataset(x_val, y_val, slen)
test_dataset = AMDDataset(x_test, y_test, slen)
pred_dataset = AMDDataset(x_test, y_test, slen, is_pred=True)
# data loaders
train_loader = data.DataLoader(train_dataset, batch_size=slen, shuffle=True, drop_last=True,
                               num_workers=4,
                               pin_memory=True)
val_loader = data.DataLoader(val_dataset, batch_size=slen, shuffle=False, drop_last=False,
                             num_workers=4)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False,
                              num_workers=4)

pred_loader = data.DataLoader(pred_dataset, batch_size=1, shuffle=False, drop_last=False,
                              num_workers=4)
# set early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=25,
                               verbose=False, mode="min")

# the trainer
trainer = pl.Trainer(accelerator='gpu', devices=[0],
                     max_epochs=200,
                     callbacks=[early_stopping])

alpha = 0.25
gamma = 2
# Check whether pretrained model exists. If yes, load it and skip training
model = AMDModel(embed_dim=53,
                 model_dim=54,
                 num_classes=3,
                 num_heads=6,
                 num_layers=8,
                 lr=1e-4,
                 warmup=50,
                 max_iters=trainer.max_epochs * len(train_loader),
                 dropout=0.1,
                 loss_func=FocalLoss(gamma, alpha)
                 )
trainer.fit(model, train_loader, val_loader)

trainer.test(model, dataloaders=test_loader)
predictions = trainer.predict(model, dataloaders=pred_loader)

y_true = torch.zeros(len(y_test), slen, 3)
for i, t in enumerate(y_test):
    a = torch.tensor(np.pad(t, (0, slen - len(t)), mode='constant', constant_values=2), dtype=torch.long)
    out = F.one_hot(a, num_classes=3)
    y_true[i] = out
print('y test shape:', y_true.shape)

y_pred = torch.zeros(len(y_test), slen, 3)
for i, t in enumerate(predictions):
    y_pred[i] = t

print('y pred shape:', y_pred.shape)

y_true = torch.flatten(y_true, end_dim=-2)
y_pred = torch.flatten(y_pred, end_dim=-2)

target = y_true[y_true[:, 2] == 0, 1].numpy()
pred = y_pred[y_true[:, 2] == 0, 1].numpy()

fpr, tpr, thresholds = roc_curve(target, pred, pos_label=1)
roc_auc = auc(fpr, tpr)

lr_precision, lr_recall, _ = precision_recall_curve(target, pred)
lr_auc = auc(lr_recall, lr_precision)

print('AUC and lr_AUC: ', roc_auc, lr_auc)

## Plot teh ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve for ' + str(mon[0]) + 'months (area = %0.4f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves for short and long term prediction')
plt.legend(loc="lower right")
plt.show()
