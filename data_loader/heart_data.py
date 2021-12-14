import os.path as osp
import numpy as np

import scipy.io
import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data
# from torch_geometric.data import InMemoryDataset


class DataWithDomain(Data):
    def __init__(self, x, y, pos, D=None):
        super().__init__()
        self.x = x
        self.y = y
        self.pos = pos
        self.D = D


class HeartGraphDataset(Dataset):
    """
    A dataset of Data objects (in pytorch geometric) with graph attributes
    from a pre-defined graph hierarchy. 
    """

    def __init__(self,
                 root,
                 data_name,
                 signal_type='egm',
                 num_mesh=None,
                 seq_len=None,
                 split='train',
                 subset=1):
        self.root = osp.expanduser(osp.normpath(root))
        self.raw_dir = osp.join(self.root, 'signal/{}/'.format(data_name))

        filename = '{}_{}_{}.mat'.format(split, signal_type, num_mesh)
        self.data_path = osp.join(self.raw_dir, filename)
        matFiles = scipy.io.loadmat(self.data_path, squeeze_me=True, struct_as_record=False)
        dataset = matFiles['params']
        label = matFiles['label']

        dataset = dataset.transpose(2, 0, 1)

        N = dataset.shape[0]
        if subset == 1:
            index = np.arange(N)
        elif subset == 0:
            raise RuntimeError('No data')
        else:
            indices = list(range(N))
            np.random.shuffle(indices)
            split = int(np.floor(subset * N))
            sub_index = indices[:split]
            dataset = dataset[sub_index, :, :]
            index = np.arange(dataset.shape[1])
        
        label = label.astype(int)
        self.label = torch.from_numpy(label[index])
        self.data = torch.from_numpy(dataset[index, :, :]).float()
        # self.corMfree = corMfree
        self.heart_name = data_name
        print('final data size: {}'.format(self.data.shape[0]))

    def __len__(self):
        return (self.data.shape[0])

    def __getitem__(self, idx):
        x = self.data[[idx], :, :]
        y = self.label[[idx]]
        sample = Data(
            x=x,
            y=y,
            pos=self.heart_name
        )
        return sample


class HeartGraphDomainDataset(Dataset):
    def __init__(self,
                 root,
                 data_name,
                 signal_type='egm',
                 num_mesh=None,
                 seq_len=None,
                 split='train',
                 subset=1,
                 k_shot=2):
        self.root = osp.expanduser(osp.normpath(root))
        self.raw_dir = osp.join(self.root, 'signal/{}/'.format(data_name))
        self.k_shot = k_shot

        filename = '{}_{}_{}.mat'.format(split, signal_type, num_mesh)
        self.data_path = osp.join(self.raw_dir, filename)
        matFiles = scipy.io.loadmat(self.data_path, squeeze_me=True, struct_as_record=False)
        dataset = matFiles['params']
        label = matFiles['label']

        dataset = dataset.transpose(2, 0, 1)

        N = dataset.shape[0]
        if subset == 1:
            index = np.arange(N)
        elif subset == 0:
            raise RuntimeError('No data')
        else:
            indices = list(range(N))
            np.random.shuffle(indices)
            split = int(np.floor(subset * N))
            sub_index = indices[:split]
            dataset = dataset[sub_index, :, :]
            index = np.arange(dataset.shape[1])
        
        label = label.astype(int)
        self.label = torch.from_numpy(label[index])
        self.data = torch.from_numpy(dataset[index, :, :]).float()
        # self.corMfree = corMfree
        self.heart_name = data_name

        scar = label[:, 1]
        scar = np.unique(scar)
        self.scar_idx = {}
        for s in scar:
            idx = np.where(label[:, 1] == s)[0]
            self.scar_idx[s] = idx

        print('final data size: {}'.format(self.data.shape[0]))

    def __len__(self):
        return (self.data.shape[0])

    def __getitem__(self, idx):
        x = self.data[[idx], :, :]
        y = self.label[[idx]]

        D = []
        scar = y[:, 1].numpy()[0]
        scar_samples = self.scar_idx[scar]
        scar_samples = np.delete(scar_samples, np.where(scar_samples == idx)[0])
        selected_samples = scar_samples[np.random.choice(len(scar_samples), self.k_shot, replace=False)]
        for item in selected_samples:
            D.append(self.data[[item], :, :])
        D = torch.cat(D, dim=1)

        sample = DataWithDomain(
            x=x,
            y=y,
            pos=self.heart_name,
            D=D
        )
        return sample


class HeartEmptyGraphDataset(Dataset):
    """
    A dataset of Data objects (in pytorch geometric) with graph attributes
    from a pre-defined graph hierarchy. The features and target values are 
    set to zeros in given graph.
    Not suitable for training.
    """

    def __init__(self,
                 mesh_graph,
                 label_type=None):
        self.graph = mesh_graph
        dim = self.graph.pos.shape[0]
        self.datax = np.zeros((dim, 101))
        self.label = np.zeros((101))

    def __len__(self):
        return (self.datax.shape[1])

    def __getitem__(self, idx):
        x = torch.from_numpy(self.datax[:, [idx]]).float()  # torch.tensor(dataset[:,[i]],dtype=torch.float)
        y = torch.from_numpy(self.label[[idx]]).float()  # torch.tensor(label_aha[[i]],dtype=torch.float)

        sample = Data(x=x,
                      y=y,
                      edge_index=self.graph.edge_index,
                      edge_attr=self.graph.edge_attr,
                      pos=self.graph.pos)
        return sample
