import torch.utils.data as data
import torch
import torch.nn.functional as F
import numpy as np


class AMDDataset(data.Dataset):
    def __init__(self, visits_seq, output_seq, max_visit, num_categories=3, is_pred=False):
        super(AMDDataset, self).__init__()
        self.num_categories = num_categories
        self.visits_seq = visits_seq
        self.output_seq = output_seq
        self.max_visit = max_visit
        self.is_pred = is_pred

    def __len__(self):
        return len(self.visits_seq)

    def __getitem__(self, idx):
        visit_seq = self.visits_seq[idx]
        output_seq = self.output_seq[idx]
        x, y, y_hot = self.augmentation(visit_seq, output_seq)

        results = dict(visit_seq=x, label=y, label_hot=y_hot)

        return x if self.is_pred else results

    def augmentation(self, visit_seq, output_seq):
        x = torch.tensor(
            np.pad(visit_seq, [(0, self.max_visit - len(visit_seq)), (0, 0)], mode='constant', constant_values=0),
            dtype=torch.float32)
        y = torch.tensor(np.pad(output_seq, (0, self.max_visit - len(output_seq)), mode='constant', constant_values=2),
                         dtype=torch.long)

    def augmentation(self, visit_seq, output_seq, mode='pre'):
        if mode == 'pre':
            padding = (self.max_visit - len(visit_seq), 0)
        elif mode == 'post':
            padding = (0, self.max_visit - len(visit_seq))
        x = torch.tensor(np.pad(visit_seq, [padding, (0, 0)], mode='constant', constant_values=0), dtype=torch.float32)
        y = torch.tensor(np.pad(output_seq, padding, mode='constant', constant_values=2), dtype=torch.long)

        y_hot = F.one_hot(y.long(), num_classes=self.num_categories)
        return x, y, y_hot
