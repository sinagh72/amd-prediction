import torch.utils.data as data
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
from data_prepration import load_data, preprocess
from transformer.adm_dataset import AMDDataset
from transformer.adm_model import AMDModel
from transformer.focal_loss import FocalLoss

if __name__ == "__main__":
    # load data
    df_harbor, df_miami = load_data()
    # set the folds and months
    folds = [1, 2, 3, 4, 5]
    mon = [3, 6, 9, 12, 15, 18, 21]
    # preprocessing and retrieving data
    # percentage = np.arange(0.0, 0.5, 0.1)
    percentage = [0]
    x_train, y_train, seq_train, x_val, y_val, seq_val, x_test, y_test, seq_test = preprocess(df_harbor, df_miami,
                                                                                              mon[0],
                                                                                              folds[0], percentage=0)
    # max visit among training and validation data
    slen = max(max(seq_train), max(seq_val))
    print('Slen: ' + str(slen))

    # creating data set for train, valid, test, prediction
    padding = 'pre'
    train_dataset = AMDDataset(x_train, y_train, slen, padding=padding)
    val_dataset = AMDDataset(x_val, y_val, slen, padding=padding)
    test_dataset = AMDDataset(x_test, y_test, slen, padding=padding)
    pred_dataset = AMDDataset(x_test, y_test, slen, is_pred=True, padding=padding)
    # data loaders
    train_loader = data.DataLoader(train_dataset, batch_size=4 * slen, shuffle=True, drop_last=True,
                                   num_workers=4,
                                   pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=4 * slen, shuffle=False, drop_last=False,
                                 num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False,
                                  num_workers=4)

    pred_loader = data.DataLoader(pred_dataset, batch_size=1, shuffle=False, drop_last=False,
                                  num_workers=4)

    early_stopping = EarlyStopping(monitor='val_f1', patience=25,
                                   verbose=False, mode="max")
    loss_function = FocalLoss(gamma=2, alpha=0.25)
    # loss_function = nn.CrossEntropyLoss(weight =torch.tensor([1, 80, 0.01]))

    # the trainer
    trainer = pl.Trainer(accelerator='gpu', devices=1,
                         max_epochs=500,
                         callbacks=[early_stopping],
                         log_every_n_steps=10
                         )
    # Check whether pretrained model exists. If yes, load it and skip training
    model = AMDModel(embed_dim=53,
                     model_dim=54,
                     num_classes=3,
                     num_heads=6,
                     num_layers=8,
                     lr=1e-4,
                     warmup=50,
                     max_iters=trainer.max_epochs * len(train_loader),
                     dropout=0.5,
                     embedding_dropout=0.1,
                     seq_len=slen,
                     loss_func=loss_function
                     )
    # model = TransformerModel(d_input=53,
    #                          d_model=54,
    #                          nhead=6,
    #                          d_hid=54,
    #                          nlayers=8,
    #                          loss_func=loss_function,
    #                          lr=1e-6,
    #                          warmup=50,
    #                          max_iters=trainer.max_epochs * len(train_loader)
    #                          )
    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, dataloaders=test_loader)
    predictions = trainer.predict(model, dataloaders=pred_loader)
    #
    # y_true = torch.zeros(len(y_test), slen, 3)
    # for i, t in enumerate(y_test):
    #     pad = (slen - len(t), 0)
    #     if padding == 'post':
    #         pad = (0, slen - len(t))
    #     a = torch.tensor(np.pad(t, pad, mode='constant', constant_values=2), dtype=torch.long)
    #     out = F.one_hot(a, num_classes=3)
    #     y_true[i] = out
    # print('y test shape:', y_true.shape)
    #
    # y_pred = torch.zeros(len(y_test), slen, 3)
    # for i, t in enumerate(predictions):
    #     y_pred[i] = t
    #
    # print('y pred shape:', y_pred.shape)
    #
    # y_true = torch.flatten(y_true, end_dim=-2)
    # y_pred = torch.flatten(y_pred, end_dim=-2)
    #
    # target = y_true[y_true[:,2] == 0, 1]
    # pred = y_pred[y_true[:,2] == 0, 1]
    #
    #
    # fpr, tpr, thresholds = roc_curve(target, pred, pos_label=1)
    # roc_auc = auc(fpr, tpr)
    #
    # lr_precision, lr_recall, _ = precision_recall_curve(target, pred)
    # lr_auc = auc(lr_recall, lr_precision)
    #
    # print('3 months_auc.txt and lr_AUC: ', roc_auc, lr_auc)
    #
    # ## Plot teh ROC curve
    #
    # plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve for ' + str(mon[0]) + 'months (area = %0.4f)' % roc_auc)
    #
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([-0.01, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC curves for short and long term prediction')
    # plt.legend(loc="lower right")
    # plt.show()
