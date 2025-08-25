import torch
from torch.utils.data import Dataset
import numpy as np
import config
from pyrocko import moment_tensor as pmt


class WaveFormDataset(Dataset):
    def __init__(self, mode):
        '''
        Loading seismic data
        '''
        self.set_mode(mode)
        self.eids = list(self.data.keys())
        self.N = len(self.eids)
        self.polarity = np.zeros((self.N, 32, 11))
        self.sta_param = np.zeros((self.N, 32, 3))
        self.focals = np.zeros((self.N, 2, 3))
        self.sta_num = np.zeros((self.N, 1))
        self.sta_mask = np.zeros((self.N, 32, 1))

        
        #start_time = time.time()
        for j in range(self.N):
            eid = self.eids[j]
            if (j+1) % np.floor(self.N/5) == 0:
                print('loading data %s / %s ...' % (j+1, self.N))
                #print('time used: %s second' % (time.time() - start_time))
                #start_time = time.time()
            focal = self.data[eid]['focalmech']
            strike = focal[0]
            dip = focal[1]
            rake = focal[2]
            moment = pmt.MomentTensor(strike=strike, dip=dip, rake = rake)
            self.focals[j, :, :] = np.concatenate((moment.p_axis().reshape(1,-1), moment.t_axis().reshape(1,-1)), axis=0)
            stas = self.data[eid]['stationxyz'][:, 1:].astype(float)
            n_sta = len(stas[:, 0])
            self.sta_param[j, :n_sta, :] = stas
            self.sta_param[j, :n_sta, 0:2] /= 50
            self.sta_param[j, :n_sta, 2] /= 20
            self.sta_param[j, n_sta:, 2] -= 100
            polarityPS = self.data[eid]['polarities']
            self.polarity[j, :n_sta, :] = polarityPS[:, :11]
            self.sta_num[j, 0] = n_sta
            self.sta_mask[j, :n_sta, 0] = 1

        self.polarity = torch.Tensor(self.polarity)
        self.sta_param = torch.Tensor(self.sta_param)
        self.focals = torch.Tensor(self.focals)
        self.sta_num = torch.Tensor(self.sta_num)
        self.sta_mask = torch.Tensor(self.sta_mask)

    def set_mode(self, mode):
        if mode == 'train':
            print('start loading training set ...')
            self.data = np.load(config.TRAIN_DATA, allow_pickle='True').item()
            self.waveform_dir = config.TRAIN_DIR
        elif mode == 'test':
            print('start loading test set ...')
            self.data = np.load(config.TEST_DATA, allow_pickle='True').item()
            self.waveform_dir = config.TEST_DIR
        elif mode == 'dev':
            print('start loading evaluation set ...')
            self.data = np.load(config.DEV_DATA, allow_pickle='True').item()
            self.waveform_dir = config.DEV_DIR


    def __getitem__(self, index):

        entry = {'wave': self.polarity[index], 'focal': self.focals[index], 'sta_param': self.sta_param[index], 'sta_num': self.sta_num[index], 'sta_mask': self.sta_mask[index]}

        return entry
    
    def __len__(self):
        return self.N
