import torch
import torch.nn as nn
import config
from utils import angle2PTB
from transformer.transformer_utils import *
import random


class FocoNet(nn.Module):
    def __init__(self):
        super(FocoNet, self).__init__()
        
        self.organize = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
        )

        self.endlayer = nn.Sequential(
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 8),
            nn.LeakyReLU(),
            nn.LayerNorm(8),
            nn.Linear(8, 3),
        )
        
        self.beg_layer = nn.Sequential(
            nn.Linear(14, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            )

        self.beg_res1 = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
        )

        self.end_res1 = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 32),
        )

        self.attentionlayer1 = EncoderLayer(d_model=128, num_heads=32, d_ff=1024, dropout=0.1)
        self.attentionlayer2 = EncoderLayer(d_model=128, num_heads=32, d_ff=1024, dropout=0.1)
        self.attentionlayer3 = EncoderLayer(d_model=128, num_heads=32, d_ff=1024, dropout=0.1)
        self.attentionlayer4 = EncoderLayer(d_model=128, num_heads=32, d_ff=1024, dropout=0.1)
        self.attentionlayer5 = EncoderLayer(d_model=128, num_heads=8, d_ff=512, dropout=0.0)
        self.attentionlayer6 = EncoderLayer(d_model=128, num_heads=8, d_ff=512, dropout=0.0)
        self.attentionlayer7 = EncoderLayer(d_model=128, num_heads=8, d_ff=512, dropout=0.0)
    
    def forward(self, x, loc, sta_num, sta_mask, mode='train'):

        N = x.shape[0]
        
        l = len(x[0,:,0])

        if mode == 'train':
            nm = list(range(l))
            random.shuffle(nm)
            x = x[:, nm, :]
            loc = loc[:, nm, :]

        out = torch.concatenate((x, loc), axis=-1)
        
        out = self.beg_layer(out)
        
        out = out + self.beg_res1(out)

        out = self.attentionlayer1(out)
        out = self.attentionlayer2(out)+out
        out = self.attentionlayer3(out)+out
        out = self.attentionlayer4(out)+out
        out = self.attentionlayer5(out)+out
        out = self.attentionlayer6(out)+out
        out = self.attentionlayer7(out)+out

        out = out * sta_mask
        out = torch.sum(out, 1) 
        out = out / sta_num
        
        out = self.organize(out)

        out = out + self.end_res1(out)

        out = self.endlayer(out)
        
        out = torch.clamp(out, -1*torch.pi, 1*torch.pi)

        out = angle2PTB(out)


        return out
    
