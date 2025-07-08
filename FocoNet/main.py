import torch
import model
from data_loader import WaveFormDataset
import argparse
import model
from solver import Solver, Evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', nargs='+', type=int, default=[])
parser.add_argument('--mode', type=str, default='TrainFromStart')
parser.add_argument('--modelPath', type=str, default='model/Focnet_298.pth')
parser.add_argument('--startEpoch', type=int, default=250)
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

print ("gpu devices being used: ", args.gpus)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def main() :

    if args.mode == 'Eval':

        torch.cuda.empty_cache()
        #torch.cuda.memory_summary(device=None, abbreviated=False)

        testset = WaveFormDataset('dev')
        Focnet = model.Focnet()
        Focnet.eval()
        #Focnet.train(False)
        print ('Evaluating... ◉ ▼ ◉')
        Focnet.load_state_dict(torch.load(args.modelPath, torch.device(device)))
        #Focnet.eval()
        evaluator = Evaluate(Focnet, testset, args)
        eval_loss = evaluator.evalidate()
        print ("Eval Loss : %.4f" % eval_loss.item())
        print ("Kagan Angle : %.2f" % (eval_loss.item()*180/3.1416))


    elif args.mode == 'TrainFromStart':

        testset = WaveFormDataset('test')
        dataset = WaveFormDataset('train')
        devset = WaveFormDataset('dev')
        Focnet = model.Focnet()

        # start training
        print ("Start training... (◉▼◉ ﾐ )Э")
        mysolver = Solver(Focnet, dataset, testset, devset, args)
        mysolver.train()
        
        print ("Finished! Hopefully...  (◉▼◉ ﾐ )Э")

    elif args.mode == 'TrainFineTuning':

        testset = WaveFormDataset('test')
        dataset = WaveFormDataset('train')
        devset = WaveFormDataset('dev')
        Focnet = model.Focnet()
        Focnet.load_state_dict(torch.load(args.modelPath))

        # start training
        print ("Start training from existing model ... (◉▼◉ ﾐ )Э")
        mysolver = Solver(Focnet, dataset, testset, devset, args, args.startEpoch)
        mysolver.train(finetune='True')
        
        print ("Finished! Hopefully...  (◉▼◉ ﾐ )Э")
    

if __name__ == '__main__':
    main()

