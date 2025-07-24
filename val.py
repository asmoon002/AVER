from __future__ import print_function
import argparse
import os
import shutil
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
#import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#from torchsummary import summary
import torchvision.models as models
# from models import *
from collections import OrderedDict
from torch.autograd import Variable
from scipy.ndimage import uniform_filter1d
# import scipy as sp
from scipy import signal
import pickle
from utils.utils import Normalize
from utils.utils import calc_scores
import logging
# import models.resnet as ResNet
import utils
import matplotlib.pyplot as plt
import numpy as np
# import cv2
import sys
from EvaluationMetrics.cccmetric import ccc

import math
from losses.CCC import CCC
#import wandb


def validate(val_loader, model, criterion, epoch, cam):
	# switch to evaluate mode
	global Val_acc
	global best_Val_acc
	global best_Val_acc_epoch
	#model.eval()
	model.eval()
	cam.eval()

	PrivateTest_loss = 0
	epoch_loss = 0
	correct = 0
	total = 0
	running_val_loss = 0
	running_val_accuracy = 0

	vout = []
	vtar = []
	aout = []
	atar = []
	#torch.cuda.synchronize()
	#t7 = time.time()
	pred_a = dict()
	pred_v = dict()
	label_a = dict()
	label_v = dict()
	#files_dict = {}
	count = 0
	global_vid_fts, global_aud_fts= None, None
	
	for batch_idx, (visualdata, audiodata, frame_ids, videos, vid_lengths, labelsV, labelsA) in tqdm(enumerate(val_loader),
														 total=len(val_loader), position=0, leave=True):
		audiodata = audiodata.cuda()#.unsqueeze(2)
		visualdata = visualdata.cuda()

		with torch.no_grad():
			b, seq_t, c, subseq_t, h, w = visualdata.size()
			visual_feats = torch.empty((b, seq_t, 25088), dtype=visualdata.dtype, device = visualdata.device)
			aud_feats = torch.empty((b, seq_t, 512), dtype=visualdata.dtype, device = visualdata.device)
			for i in range(visualdata.shape[0]):
				audio_feat, visualfeat, _ = model(audiodata[i,:,:,:], visualdata[i, :, :, :,:,:])

				visual_feats[i,:,:] = visualfeat
				aud_feats[i,:,:] = audio_feat

			audiovisual_vouts,audiovisual_aouts = cam(aud_feats, visual_feats)
			# audiovisual_vouts,audiovisual_aouts = cam(aud_feats, visual_feats)

			##### 추가 #####
			val_voutputs = audiovisual_vouts.view(-1, audiovisual_vouts.shape[0]*audiovisual_vouts.shape[1])
			val_aoutputs = audiovisual_aouts.view(-1, audiovisual_aouts.shape[0]*audiovisual_aouts.shape[1])   
			vtargets = labelsV.view(-1, labelsV.shape[0]*labelsV.shape[1]).cuda()
			atargets = labelsA.view(-1, labelsA.shape[0]*labelsA.shape[1]).cuda()   
   
			v_loss = criterion(val_voutputs, vtargets)
			a_loss = criterion(val_aoutputs, atargets)
			final_loss = v_loss + a_loss
			epoch_loss += final_loss.cpu().data.numpy()
   			################
   
			audiovisual_vouts = audiovisual_vouts.detach().cpu().numpy()
			audiovisual_aouts = audiovisual_aouts.detach().cpu().numpy()

			labelsV = labelsV.cpu().numpy()
			labelsA = labelsA.cpu().numpy()

			for voutputs, aoutputs, labelV, labelA, frameids, video, vid_length in zip(audiovisual_vouts, audiovisual_aouts, labelsV, labelsA, frame_ids, videos, vid_lengths):
				for voutput, aoutput, labV, labA, frameid, vid, length in zip(voutputs, aoutputs, labelV, labelA, frameids, video, vid_length):
					if vid not in pred_a:
						if frameid>1:
							print(vid)
							print(length)
							print("something is wrong")
							sys.exit()
						count = count + 1

						pred_a[vid] = [0]*length
						pred_v[vid] = [0]*length
						label_a[vid] = [0]*length
						label_v[vid] = [0]*length
						if labV == -5.0:
							continue
						pred_a[vid][frameid-1] = aoutput
						pred_v[vid][frameid-1] = voutput
						label_a[vid][frameid-1] = labA
						label_v[vid][frameid-1] = labV
					else:
						if frameid <= length:
							if labV == -5.0:
								continue
							pred_a[vid][frameid-1] = aoutput
							pred_v[vid][frameid-1] = voutput
							label_a[vid][frameid-1] = labA
							label_v[vid][frameid-1] = labV
       
		# if batch_idx==3:break
       

	for key in pred_a.keys():
		clipped_preds_v = np.clip(pred_v[key], -1.0, 1.0)
		clipped_preds_a = np.clip(pred_a[key], -1.0, 1.0)
  

		smoothened_preds_v = uniform_filter1d(clipped_preds_v, size=20, mode='constant')
		smoothened_preds_a = uniform_filter1d(clipped_preds_a, size=50, mode='constant')
		tars_v = label_v[key]
		tars_a = label_a[key]

		for i in range(len(smoothened_preds_a)):
			vout.append(smoothened_preds_v[i])
			aout.append(smoothened_preds_a[i])
			vtar.append(tars_v[i])
			atar.append(tars_a[i])

	cccV = ccc(np.array(vout), np.array(vtar))
	cccA = ccc(np.array(aout), np.array(atar))
	print("Valence CCC: ", cccV)
	print("Arousal CCC: ", cccA)
	return cccV, cccA, final_loss
