import torch.utils.data as data
import torch
import torch.nn.functional as F
import numpy as np


class AMDDataset(data.Dataset):
    def __init__(self, feature_list, label_list, visits_seq, max_visit, num_categories=3):
        super(AMDDataset, self).__init__()
        self.num_categories = num_categories
        self.visits_seq = visits_seq
        self.patients = feature_list
        self.label_list = label_list
        self.max_visit = max_visit

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        visit_seq = self.patients[idx]
        label = self.label_list[idx]

        x = torch.tensor(np.pad(visit_seq, [(0, self.max_visit - self.visits_seq[idx]), (0, 0)],
                                mode='constant', constant_values=0.0), dtype=torch.float32)
        y = torch.tensor(np.pad(label, (0, self.max_visit - self.visits_seq[idx]),
                                mode='constant', constant_values=2.0))
        y_hot = F.one_hot(y, num_classes=-1)

        results = dict(visit_seq=x, label=y, label_hot=y_hot)

        return results

    def augmentation(self):
        pass
