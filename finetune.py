from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from dataloader import KITTIloader2015 as ls
from dataloader import KITTILoader as DA

from models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/training/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='./trained/submission_model.tar',
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--right-to-left', action='store_true',default=False,
                    help='Trains model for right to Left usage. Note: for now supported on KITTI2015')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set gpu id used
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.datatype == '2015':
   from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
   from dataloader import KITTIloader2012 as ls


all_left_img, all_right_img, all_left_disp, all_right_disp, test_left_img, test_right_img, test_left_disp, test_right_disp = ls.dataloader(args.datapath)

if not args.right_to_left :
    #Fine-tunning from Left to Right
    TrainImgLoader = torch.utils.data.DataLoader(
            DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
            batch_size= 12, shuffle= True, num_workers= 8, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
            DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False),
            batch_size= 8, shuffle= False, num_workers= 4, drop_last=False)

else:
    #Fine-tunning from right to Left
    TrainImgLoader = torch.utils.data.DataLoader(
            DA.myImageFloder(all_right_img,all_left_img,all_right_disp, True),
            batch_size= 12, shuffle= True, num_workers= 8, drop_last=False)


    TestImgLoader = torch.utils.data.DataLoader(
            DA.myImageFloder(test_right_img,test_left_img,test_right_disp, False), 
            batch_size= 8, shuffle= False, num_workers= 4, drop_last=False)


if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

def train(imgS,imgD,disp_T):
        '''
        Trains the model
        In Left to Right Case:
            imgS -> Left Image
            imgD -> Right Image
        '''
        model.train()
        imgS   = Variable(torch.FloatTensor(imgS))
        imgD   = Variable(torch.FloatTensor(imgD))   
        disp_T = Variable(torch.FloatTensor(disp_T))

        if args.cuda:
            imgS, imgD, disp_true = imgS.cuda(), imgD.cuda(), disp_T.cuda()

        #---------
        mask = (disp_true > 0)
        mask.detach_()
        #----

        optimizer.zero_grad()
        
        if args.model == 'stackhourglass':
            output1, output2, output3 = model(imgS,imgD)
            output1 = torch.squeeze(output1,1)
            output2 = torch.squeeze(output2,1)
            output3 = torch.squeeze(output3,1)
            loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
        elif args.model == 'basic':
            output = model(imgS,imgD)
            output = torch.squeeze(output3,1)
            loss = F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)

        loss.backward()
        optimizer.step()

        return loss.data[0]

def test(imgS,imgD,disp_true):
        model.eval()
        imgS   = Variable(torch.FloatTensor(imgS))
        imgD   = Variable(torch.FloatTensor(imgD))   
        if args.cuda:
            imgS, imgD = imgS.cuda(), imgD.cuda()

        with torch.no_grad():
            output3 = model(imgS,imgD)

        pred_disp = output3.data.cpu()

        #computing 3-px error#
        true_disp = disp_true
        index = np.argwhere(true_disp>0)
        disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
        correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
        torch.cuda.empty_cache()

        return 1-(float(torch.sum(correct))/float(len(index[0])))

def adjust_learning_rate(optimizer, epoch):
    # modified for fine-tune from right to left on a model trained for left to right
    #if epoch <= 200:
    if epoch <= 100:
       lr = 0.001
    else:
       lr = 0.0001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
	max_acc=0
	max_epo=0
	start_full_time = time.time()
        
        if not args.right_to_left:
            print('Fine-tunning from Left to Right.')
        else:
            print('Fine-tunning from Right to Left.')

	for epoch in range(1, args.epochs+1):
	   total_train_loss = 0
	   total_test_loss = 0
	   adjust_learning_rate(optimizer,epoch)
           
               ## training ##
           for batch_idx, (imgS_crop, imgD_crop, disp_crop_T) in enumerate(TrainImgLoader):
               start_time = time.time() 

               loss = train(imgS_crop,imgD_crop, disp_crop_T)
	       print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
	       total_train_loss += loss
	   print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
	   
               ## Test ##

           for batch_idx, (imgS, imgD, disp_T) in enumerate(TestImgLoader):
               test_loss = test(imgS,imgD, disp_T)
               print('Iter %d 3-px error in val = %.3f' %(batch_idx, test_loss*100))
               total_test_loss += test_loss


	   print('epoch %d total 3-px error in val = %.3f' %(epoch, total_test_loss/len(TestImgLoader)*100))
	   if total_test_loss/len(TestImgLoader)*100 > max_acc:
		max_acc = total_test_loss/len(TestImgLoader)*100
		max_epo = epoch
	   print('MAX epoch %d total test error = %.3f' %(max_epo, max_acc))

	   #SAVE
	   savefilename = args.savemodel+'finetune_'+str(epoch)+'.tar'
	   torch.save({
		    'epoch': epoch,
		    'state_dict': model.state_dict(),
		    'train_loss': total_train_loss/len(TrainImgLoader),
		    'test_loss': total_test_loss/len(TestImgLoader)*100,
		}, savefilename)
	
        print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))
	print(max_epo)
	print(max_acc)


if __name__ == '__main__':
   main()
