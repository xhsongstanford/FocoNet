import torch
import model
from data_loader import WaveFormDataset
import argparse
import model
from solver import Solver, Evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', nargs='+', type=int, default=[])
parser.add_argument('--mode', type=str, default='TrainFromStart')
parser.add_argument('--modelPath', type=str, default='model/FocoNet_Full.pth')
parser.add_argument('--startEpoch', type=int, default=250)
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

print ("gpu devices being used: ", args.gpus)

def main() :

    if args.mode == 'Eval':

        # torch.cuda.empty_cache()
        # torch.cuda.memory_summary(device=None, abbreviated=False)

        testset = WaveFormDataset('dev')
        FocoNet = model.FocoNet().to(device=device)
        FocoNet.eval()
        #FocoNet.train(False)
        print ('Evaluating... ◉ ▼ ◉')
        FocoNet.load_state_dict(torch.load(args.modelPath, map_location=device))
        #FocoNet.eval()
        evaluator = Evaluate(FocoNet, testset, args)
        eval_loss = evaluator.evalidate()
        print ("Eval Loss : %.4f" % eval_loss.item())
        print ("Kagan Angle : %.2f" % (eval_loss.item()*180/3.14159265359))


    elif args.mode == 'TrainFromStart':

        testset = WaveFormDataset('test')
        dataset = WaveFormDataset('train')
        devset = WaveFormDataset('dev')
        FocoNet = model.FocoNet()

        # start training
        print ("Start training... (◉▼◉ ﾐ )Э")
        mysolver = Solver(FocoNet, dataset, testset, devset, args)
        mysolver.train()
        
        print ("Finished! Hopefully...  (◉▼◉ ﾐ )Э")

    elif args.mode == 'TrainFineTuning':

        testset = WaveFormDataset('test')
        dataset = WaveFormDataset('train')
        devset = WaveFormDataset('dev')
        FocoNet = model.FocoNet()
        FocoNet.load_state_dict(torch.load(args.modelPath, map_location=device))

        # start training
        print ("Start training from existing model ... (◉▼◉ ﾐ )Э")
        mysolver = Solver(FocoNet, dataset, testset, devset, args, args.startEpoch)
        mysolver.train(finetune='True')
        
        print ("Finished! Hopefully...  (◉▼◉ ﾐ )Э")
    

if __name__ == '__main__':
    main()

