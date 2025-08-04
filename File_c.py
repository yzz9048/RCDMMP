import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class CustomDataset(Dataset):
    def __init__(
        self, 
        name,
        data_root, 
        window=64, 
        DeltaT = 10,
        proportion=0.9, 
        save2npy=True, 
        neg_one_to_one=True,
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=3
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.rawdata, self.scaler_rps, self.scaler_data = self.read_data(data_root, self.name)
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.sample_num_total = (self.rawdata.shape[0] - window)//DeltaT +1 # 可用的窗口数量
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.rawdata[:,:-1] = self.__normalize(self.rawdata[:,:-1]) 
        train, inference = self.__getsamples(self.rawdata, proportion, DeltaT, seed) # inference 即 test

        train_data = train[:,:,:-1]
        train_label = train[:,:,-1]
        inference_data = inference[:,:,:-1]
        inference_label = inference[:,:,-1]
        if period == 'train':
            self.samples = train_data  
            self.labels = train_label
        else:
            self.samples = inference_data  
            self.labels = inference_label
        if period == 'test':
            if predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, proportion, DeltaT, seed):
        x = np.zeros((self.sample_num_total, self.window, self.var_num))
        i = 0
        for w in range(self.window, data.shape[0] - self.window + 1, DeltaT): # Split self.data by window
            end= w
            start = w - self.window
            x[i, :, :] = data[start:end, :]
            i += 1

        train_data, test_data = self.divide(x, proportion, seed)

        train_data_raw = train_data[:,:,:-1]
        test_data_raw = test_data[:,:,:-1]
        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data_raw))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data_raw))
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data_raw))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data_raw))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data_raw)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data_raw)

        return train_data, test_data

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num-1))
        return d.reshape(-1, self.window, self.var_num-1)
    
    def __normalize(self, rawdata):
        data = self.scaler_rps.transform(rawdata[:,0].reshape(-1, 1))
        data = np.concatenate([data,self.scaler_data.transform(rawdata[:,1:])], axis=1)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = self.scaler_rps.inverse_transform(data[:,0].reshape(-1, 1))
        x = np.concatenate([x,self.scaler_data.inverse_transform(data[:,1:])], axis=1)
        return x
    
    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        # id_rdm = np.random.permutation(size)
        id_rdm = np.arange(size)
        regular_train_id = id_rdm[:regular_train_num] 
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        df = pd.read_csv(filepath, header=0)
        # if name == 'drift':
            # df.drop(df.columns[0:2], axis=1, inplace=True) 
        df.drop(df.columns[0], axis=1, inplace=True) 
        data = df.values
        scaler_rps = MinMaxScaler()
        scaler_data = MinMaxScaler()
        scaler_rps = scaler_rps.fit(data[:,0].reshape(-1, 1))
        scaler_data = scaler_data.fit(data[:,1:-1])
        return data, scaler_rps, scaler_data
    
    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            labels = self.labels[ind, :]
            return torch.from_numpy(x).float(), torch.from_numpy(m), torch.from_numpy(labels)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        labels = self.labels[ind, :]
        return torch.from_numpy(x).float(), torch.from_numpy(labels)

    def __len__(self):
        return self.sample_num
    
