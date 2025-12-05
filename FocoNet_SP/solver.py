import os
import sys
import numpy as np
import torch
import torch.nn as nn
import copy 
from torch.utils.data import DataLoader 
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler
import utils
import config
import Kagan
import logging

cuda = torch.cuda.is_available()
kwargs = {'num_workers':0, 'pin_memory':True} if cuda else {}
print ("gpu available :", cuda)
device = torch.device("cuda" if cuda else "cpu")
#device = torch.device("cpu")

num_gpu = torch.cuda.device_count()

torch.cuda.manual_seed(5)

class Solver(object):
    def __init__(self, model, dataset, testset, devset, args, start_epoch = 0):

        torch.cuda.empty_cache()

        self.FocoNet = model
        self.dataset = dataset
        self.testset = testset
        self.devset = devset
        self.args = args

        self.start_epoch = start_epoch
        self.curr_epoch = start_epoch

        self.model_savepath = './model'
        if not os.path.exists(self.model_savepath):
            os.makedirs(self.model_savepath)

        # define loss function 

        self._initialize()
        self.set_mode('train')


    def _initialize(self):

        #self.optimizer = torch.optim.SGD(self.FocoNet.parameters(), lr=config.LR, weight_decay=1e-6, momentum=0.9, nesterov=True)
        self.optimizer = torch.optim.Adam(self.FocoNet.parameters(), lr=config.LR, betas = (0.9, 0.999), weight_decay=1e-6)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.25, patience=2,)

        # initialize cuda
        if len(self.args.gpus) > 1:
            self.multigpu = True
        else:
            self.multigpu = False
        
        utils.handle_multigpu(self.multigpu, self.args.gpus, num_gpu)

        if self.multigpu:
            self.FocoNet = nn.DataParallel(self.FocoNet, device_ids=self.args.gpus)

        self.FocoNet.to(device)

    def set_mode(self, mode):

        if mode == 'train':
            self.dataloader = DataLoader(self.dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True, **kwargs)
        elif mode == 'test':
            self.testloader = DataLoader(self.testset, batch_size=min(self.testset.N, config.BATCH_SIZE), shuffle=False, drop_last=True, **kwargs)
            self.devloader = DataLoader(self.devset, batch_size=min(self.devset.N, config.BATCH_SIZE), shuffle=False, drop_last=False, **kwargs)


    def train(self, finetune=False) :

        if finetune:

            os.system('cp train_loss.txt train_loss_new.txt')
            os.system('cp test_loss.txt test_loss_new.txt')
            os.system('cp dev_loss.txt dev_loss_new.txt')

            loss_file = open('train_loss_new.txt', 'a')
            eval_file = open('test_loss_new.txt', 'a')
            dev_file  = open('dev_loss_new.txt', 'a')

        else:
            loss_file = open('train_loss.txt', 'w')
            eval_file = open('test_loss.txt', 'w')
            dev_file  = open('dev_loss.txt', 'w')

        # Train the network
        for epoch in range(self.start_epoch, config.NUM_EPOCHS):

            self.set_mode('train') 
            print ('Training ...')

            for i, data in enumerate(self.dataloader):

                waveform = data['wave'].to(device)
                sta_param = data['sta_param'].to(device)
                focals = data['focal'].to(device)
                sta_num = data['sta_num'].to(device)
                sta_mask = data['sta_mask'].to(device)

                outputs = self.FocoNet(waveform, sta_param, sta_num, sta_mask)

                self.optimizer.zero_grad()

                loss = Kagan.PTB2Kagan(focals, outputs).sum().to(device)/len(focals[:, 0, 0])



                #kangle = Kagan.PTB2Kagan(focals, outputs).sum().to(device)/len(focals[:, 0, 0])
                #theta = Kagan.PTB2Kagan(focals, outputs)
                #loss = (theta/(120.1*torch.pi/180-theta)).sum().to(device)
                #loss = torch.sum((focals - outputs[:, 0:2, :])**2)
                
                weights = torch.Tensor([0]).to(device)
                for param in self.FocoNet.parameters():
                    weights += 0.005 * param.norm(2) ** 2
                #print(torch.nn.utils.weight_norm(self.FocoNet).weight_g)
                
                loss.backward()
                clip_grad_norm_(self.FocoNet.parameters(), 10)

                self.optimizer.step()

                if (i+1) % 10 == 0:
                    print ("Epoch [%d/%d], Iter [%d/%d] loss : %.4f " % (epoch+1, config.NUM_EPOCHS, i+1, len(self.dataloader), loss.item()))
                loss_file.write('%5s' % str(int(epoch+1)) + '%5s' % str(int(i+1)) + '%8s' % '%.2f' % (loss.item()*180/np.pi) + '\n')


            self.curr_epoch +=1

            print ('Evaluating ...')
            eval_loss, dev_loss = self.validate()
            print ("Test Loss : %.4f" % eval_loss.item())
            print ("Test Kagan Angle : %.2f" % (eval_loss.item()*180/3.1415))
            print ("Dev Loss : %.4f" % dev_loss.item())
            print ("Dev Kagan Angle : %.2f" % (dev_loss.item()*180/3.1415))
            eval_file.write('%5s' % str(int(epoch+1)) + '%8s' % '%.2f' % (eval_loss.item()*180/np.pi) + '\n')
            dev_file.write('%5s' % str(int(epoch+1)) + '%8s' % '%.2f' % (dev_loss.item()*180/np.pi) + '\n')
            # eval_file.write(str(eval_loss.item() *180/np.pi) + '\n')
            # dev_file.write(str(dev_loss.item() *180/np.pi) + '\n')
            
            if self.curr_epoch % 10 == 0:
                self.scheduler.step(loss) # use the learning rate scheduler
            if self.curr_epoch % 50 == 0 or dev_loss.item()*180/3.1415 < 19:
                if dev_loss.item()*180/3.1415 < 19:
                    print("**** Good Model: Saving Model ****")
                torch.save(self.FocoNet.state_dict(), self.model_savepath +'/'+ self.FocoNet.__class__.__name__ + '_' + str(self.curr_epoch) + '.pth')
            curr_lr = self.optimizer.param_groups[0]['lr']
            print ('Learning rate : {}'.format(curr_lr))
            if curr_lr < 1e-8:
                print ("Early stopping")
                break

        loss_file.close()
        eval_file.close()
        dev_file.close()
        
    # During training use this function to validate.
        
    def validate(self):

        self.set_mode('test')
        # TEST SET
        test_loss = 0.0
        n_eq = 0
        for i, data in enumerate(self.testloader):
            polarities = data['wave'].to(device)
            sta_param = data['sta_param'].to(device)
            focals = data['focal'].to(device)
            sta_num = data['sta_num'].to(device)
            sta_mask = data['sta_mask'].to(device)
            outputs = self.FocoNet(polarities, sta_param, sta_num, sta_mask).to(device)
            # if self.curr_epoch%5 == 0:
            #     np.save('test_outputs.npy', outputs.cpu().detach().numpy().reshape(-1, 9))
            #     np.save('test_referece.npy', focals.cpu().detach().numpy().reshape(-1, 6))
            test_loss += Kagan.PTB2Kagan(focals, outputs).sum().to(device)
            n_eq += len(focals[:, 0, 0])
        test_loss = test_loss / n_eq
        # DEV SET
        dev_loss = 0.0
        n_eq = 0
        for i, data in enumerate(self.devloader):
            polarities = data['wave'].to(device)
            sta_param = data['sta_param'].to(device)
            focals = data['focal'].to(device)
            sta_num = data['sta_num'].to(device)
            sta_mask = data['sta_mask'].to(device)
            outputs = self.FocoNet(polarities, sta_param, sta_num, sta_mask).to(device)
            # if self.curr_epoch%5 == 0:
            #     np.save('dev_outputs.npy', outputs.cpu().detach().numpy().reshape(-1, 9))
            #     np.save('dev_referece.npy', focals.cpu().detach().numpy().reshape(-1, 6))
            dev_loss += Kagan.PTB2Kagan(focals, outputs).sum().to(device)
            n_eq += len(focals[:, 0, 0])
        dev_loss = dev_loss / n_eq

        return test_loss, dev_loss

            
class Evaluate(object):
    def __init__(self, model, testset, args):

        torch.cuda.empty_cache()

        self.FocoNet = model.eval()
        self.testset = testset
        self.args = args
        
        self.curr_epoch = 0

        self._initialize()
        self.set_mode('test')


    def _initialize(self):

        # initialize cuda
        if len(self.args.gpus) > 1:
            self.multigpu = True
        else:
            self.multigpu = False
        
        utils.handle_multigpu(self.multigpu, self.args.gpus, num_gpu)

        if self.multigpu:
            self.FocoNet = nn.DataParallel(self.FocoNet, device_ids=self.args.gpus)
        
        self.FocoNet.eval().to(device)

    def set_mode(self, mode):

        if mode == 'test':
            self.testloader = DataLoader(self.testset, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False, **kwargs)
            #self.testloader = DataLoader(self.testset, batch_size=self.testset.N, shuffle=False, drop_last=False, **kwargs)
            print('batchsize: ' + str(config.BATCH_SIZE))

    def memory_stats():
        print(torch.cuda.memory_allocated()/1024**2)
        print(torch.cuda.memory_cached()/1024**2)

    def evalidate(self):

        eval_loss = 0.0
        n_eq = 0

        SUM_out = torch.zeros((1,9)).to(device)
        SUM_foc = torch.zeros((1,6)).to(device)

        for i, data in enumerate(self.testloader):

            #print(len(self.testloader))

            with torch.no_grad():
                #print(torch.cuda.memory_allocated()/1024**2)
                #self.memory_stats()

                polarities = data['wave'].to(device)
                #print('NNNN')
                sta_param = data['sta_param'].to(device)
                #print('KKKK')
                focals = data['focal'].to(device)
                sta_num = data['sta_num'].to(device)
                sta_mask = data['sta_mask'].to(device)

                outputs = self.FocoNet(polarities, sta_param, sta_num, sta_mask, mode='test')#.to(device)

                SUM_out = torch.concatenate([SUM_out, outputs.reshape(-1, 9)], axis=0)
                SUM_foc = torch.concatenate([SUM_foc, focals.reshape(-1, 6)], axis=0)

                eval_loss += Kagan.PTB2Kagan(focals, outputs).sum().to(device)
                n_eq += len(focals[:, 0, 0])

        np.save('dev_outputs.npy', SUM_out.cpu().detach().numpy()[1:,:])
        np.save('dev_referece.npy', SUM_foc.cpu().detach().numpy()[1:,:])
                
        
        avg_loss = eval_loss / n_eq

        return avg_loss


if __name__ == '__main__':
    pass

