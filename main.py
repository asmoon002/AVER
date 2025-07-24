import argparse
import os
import time
import random
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import gc
from models.pytorch_i3d_new import InceptionI3d
from models.I3DWSDDA import I3D_WSDDA
from train import train
from val import validate
from test import Test
import logging
import utils
import matplotlib.pyplot as plt
from utils.parser import parse_configuration
import numpy as np
# from models.orig_cam import GAT_LSTM_CAM
from models.orig_cam import TLAB_CAM as Custom_CAModel
from models.tsav import TwoStreamAuralVisualModel
import sys
from datasets.dataset_new import ImageList
from datasets.dataset_val import ImageList_val
from datasets.dataset_test import ImageList_test
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses.loss import CCCLoss
from datetime import datetime, timedelta
import pandas as pd
import traceback
from torch import nn
#import wandb
import json
from warnings import filterwarnings
filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#wandb.init(settings=wandb.Settings(start_method="fork"), project='Audio Visual Fusion')

args = argparse.ArgumentParser(description='DomainAdaptation')
args.add_argument('-c', '--config', default="config_file.json", type=str,
					  help='config file path (default: None)')
args.add_argument('-t', '--time_chk', default="False", type=str,
					  help='Time check (default: False)')
args.add_argument('-s', '--seed', default=0, type=int,
					  help='random seed number (default: 0)')
args.add_argument('-fm', '--fusion_model', default="tlab", type=str,
					  help='Fusion Model (default: transformer)')

args.add_argument('-r', '--resume', default=0, type=int,
					  help='resume (default: None)')
args.add_argument('-resume_file', '--resume_file', default=None, type=str,
					  help='resume file (default: None)')
args.add_argument('-ckpt', '--check_point', default=0, type=int,
                  help='check point bool (default : 0)')



args = args.parse_args()
configuration = parse_configuration(args.config)

if (args.time_chk).lower() == "true":
    is_time_chk = True
else:
    is_time_chk = False

best_Val_acc = 0  # best PrivateTest accuracy
#best_Val_acc = 0  # best PrivateTest accuracy
best_Val_acc_epoch = 0
start_epoch = configuration['model_params']['start_epoch'] #0  # start from epoch 0 or last checkpoint epoch
total_epoch = configuration['model_params']['max_epochs'] #0  # start from epoch 0 or last checkpoint epoch

TrainingAccuracy_V = []
TrainingAccuracy_A = []
ValidationAccuracy_V = []
ValidationAccuracy_A = []

Logfile_name = "LogFiles/" + "log_file.log"
logging.basicConfig(filename=Logfile_name, level=logging.INFO)

if args.resume == 1:
    SEED = int(args.resume_file.split("_")[3])
else: 
    SEED = args.seed
    
print("SEED : ", SEED)
    
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)


class TrainPadSequence:
	def __call__(self, sorted_batch):
		sequences = [x[0] for x in sorted_batch]
		aud_sequences = [x[1] for x in sorted_batch]
		spec_dim = []

		for aud in aud_sequences:
			spec_dim.append(aud.shape[3])

		max_spec_dim = max(spec_dim)
		audio_features = torch.zeros(len(spec_dim), 16, 1, 64, max_spec_dim)
		for batch_idx, spectrogram in enumerate(aud_sequences):
			if spectrogram.shape[2] < max_spec_dim:
				audio_features[batch_idx, :, :, :, -spectrogram.shape[3]:] = spectrogram
			else:
				audio_features[batch_idx, :,:, :, :] = spectrogram

		labelV = [x[2] for x in sorted_batch]
		labelA = [x[3] for x in sorted_batch]
		visual_sequences = torch.stack(sequences)
		labelsV = torch.stack(labelV)
		labelsA = torch.stack(labelA)

		return visual_sequences, audio_features, labelsV, labelsA


class ValPadSequence:
	def __call__(self, sorted_batch):

		sequences = [x[0] for x in sorted_batch]
		aud_sequences = [x[1] for x in sorted_batch]
		spec_dim = []
		for aud in aud_sequences:
			spec_dim.append(aud.shape[3])

		max_spec_dim = max(spec_dim)
		audio_features = torch.zeros(len(spec_dim), 16, 1, 64, max_spec_dim)
		for batch_idx, spectrogram in enumerate(aud_sequences):
			if spectrogram.shape[2] < max_spec_dim:
				audio_features[batch_idx, :, :, :, -spectrogram.shape[3]:] = spectrogram
			else:
				audio_features[batch_idx, :,:, :, :] = spectrogram

		frameids = [x[2] for x in sorted_batch]
		v_ids = [x[3] for x in sorted_batch]
		v_lengths = [x[4] for x in sorted_batch]
		labelV = [x[5] for x in sorted_batch]
		labelA = [x[6] for x in sorted_batch]

		visual_sequences = torch.stack(sequences)
		labelsV = torch.stack(labelV)
		labelsA = torch.stack(labelA)
		return visual_sequences, audio_features, frameids, v_ids, v_lengths, labelsV, labelsA


class TestPadSequence:
	def __call__(self, sorted_batch):

		sequences = [x[0] for x in sorted_batch]
		aud_sequences = [x[1] for x in sorted_batch]
		spec_dim = []
		for aud in aud_sequences:
			spec_dim.append(aud.shape[3])

		max_spec_dim = max(spec_dim)
		audio_features = torch.zeros(len(spec_dim), 16, 1, 64, max_spec_dim)
		for batch_idx, spectrogram in enumerate(aud_sequences):
			if spectrogram.shape[2] < max_spec_dim:
				audio_features[batch_idx, :, :, :, -spectrogram.shape[3]:] = spectrogram
			else:
				audio_features[batch_idx, :,:, :, :] = spectrogram

		frameids = [x[2] for x in sorted_batch]
		v_ids = [x[3] for x in sorted_batch]
		v_lengths = [x[4] for x in sorted_batch]


		visual_sequences = torch.stack(sequences)

		return visual_sequences, audio_features, frameids, v_ids, v_lengths


if not os.path.isdir("SavedWeights"):
	os.makedirs("SavedWeights", exist_ok=True)

weight_save_path = "SavedWeights"

result_save_path ="save"
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)

### Loading audiovisual model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = 'ABAW2020TNT/model2/TSAV_Sub4_544k.pth.tar' # path to the model
model = TwoStreamAuralVisualModel(num_channels=4)
saved_model = torch.load(model_path)
model.load_state_dict(saved_model['state_dict'])

new_first_layer = nn.Conv3d(in_channels=3,
					out_channels=model.video_model.r2plus1d.stem[0].out_channels,
					kernel_size=model.video_model.r2plus1d.stem[0].kernel_size,
					stride=model.video_model.r2plus1d.stem[0].stride,
					padding=model.video_model.r2plus1d.stem[0].padding,
					bias=False)

new_first_layer.weight.data = model.video_model.r2plus1d.stem[0].weight.data[:, 0:3]
model.video_model.r2plus1d.stem[0] = new_first_layer
model = nn.DataParallel(model)
model = model.to(device)

### Freezing the model
for p in model.parameters():
	p.requires_grad = False
for p in model.children():
	p.train(False)
 
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Is CUDA available? ", torch.cuda.is_available())


fusion_model_name = args.fusion_model
fusion_model = Custom_CAModel()

print_model_name = fusion_model.__class__.__name__
print("Fusion Model : ", print_model_name)

fusion_model = fusion_model.to(device=device)

flag = configuration["Mode"]
print("flag : ", flag)

##########################################################################
if args.resume == 1:
	cam_model_path = f'SavedWeights/{args.resume_file}' # path to the model
	cam_saved_model = torch.load(cam_model_path)
	fusion_model.load_state_dict(cam_saved_model['net'])
	cammodel_accV = torch.load(cam_model_path)['best_Val_accV']
	cammodel_accA = torch.load(cam_model_path)['best_Val_accA']
	best_Val_acc = cammodel_accV + cammodel_accA
	best_Val_acc_epoch = cam_saved_model['best_Val_acc_epoch']
	print("Saved cammodel_accV : ", cammodel_accV)
	print("Saved cammodel_accA : ", cammodel_accA)
	print("Saved best_Val_acc : ", best_Val_acc)
	print("Saved best_epoch : ", best_Val_acc_epoch)
 
##########################################################################
if flag == "Testing":
	cam_model_path = 'SavedWeights/Val_model_valence_cnn_lstm_mil_64_new_fd_128.pt' # path to the model
	cam_saved_model = torch.load(cam_model_path)
	fusion_model.load_state_dict(cam_saved_model['net'])
	cammodel_accV = torch.load(cam_model_path)['best_Val_accV']
	cammodel_accA = torch.load(cam_model_path)['best_Val_accA']
	print(cammodel_accV)
	print(cammodel_accA)
	for param in fusion_model.parameters():  # children():
		param.requires_grad = False

print('==> Preparing data..')


def matching_files(root_path, anno_path):
	anno_list = []
	for f in os.listdir(anno_path):
		anno_list.append(f.split(".")[0])
  
	root_path_list = os.listdir(root_path)
	
	for f in os.listdir(root_path):
		if not f in anno_list:
			del root_path_list[root_path_list.index(f)]

	return root_path_list

def train_val_test_split(root_path, anno_path, seed=0):
	random.seed(seed)
	trial_data = matching_files(root_path, anno_path)
 
	fname_dict = {i:f for i,f in enumerate(trial_data)}
	length = len(fname_dict)
 
	print("full trial length: ", len(fname_dict))

	train_set = []
	valid_set = []
	test_set = []
 
	train_list_idx = random.sample(fname_dict.keys(), int(length*0.6))
	for i in train_list_idx:
		train_set.append(fname_dict[i]+".csv")
		del fname_dict[i]
		
	valid_list_idx = random.sample(fname_dict.keys(), int(length*0.2))
	for i in valid_list_idx:
		valid_set.append(fname_dict[i]+".csv")
		del fname_dict[i]

	test_list_idx = random.sample(fname_dict.keys(), int(length*0.2))    
	for i in test_list_idx:
		test_set.append(fname_dict[i]+".csv")
		del fname_dict[i]
  
	return train_set, valid_set, test_set
    
dataset_rootpath = configuration['dataset_rootpath']
dataset_wavspath = configuration['dataset_wavspath']
dataset_labelpath = configuration['labelpath']
# train_set, valid_set, test_set = train_val_test_split(dataset_rootpath, dataset_labelpath, SEED)

def load_partition_set(partition_path, seed):
	import json

	with open(partition_path, 'r') as f:    
		seed_data = json.load(f)

	seed_data_train = seed_data[f'seed_{seed}']['Train_Set']
	seed_data_valid = seed_data[f'seed_{seed}']['Validation_Set']
	seed_data_test  = seed_data[f'seed_{seed}']['Test_Set']
 
	seed_data_train = [fn + ".csv" for fn in seed_data_train]
	seed_data_valid = [fn + ".csv" for fn in seed_data_valid]
	seed_data_test  = [fn + ".csv" for fn in seed_data_test ]

	return seed_data_train, seed_data_valid, seed_data_test


partition_path = "../data/Affwild2/seed_data.json"
 
train_set, valid_set, test_set = load_partition_set(partition_path, SEED)

init_time = datetime.now()
init_time = init_time.strftime('%m%d_%H%M')

root_time_chk_dir = "time_chk"

if is_time_chk:
	time_chk_path = os.path.join(root_time_chk_dir, init_time)
    
	if not os.path.exists(time_chk_path):
		os.makedirs(time_chk_path)
	else:
		init_time = datetime.now()
		init_time = init_time + timedelta(minutes=1)  # 1분 더하기
		init_time = init_time.strftime('%m%d_%H%M')
		time_chk_path = os.path.join(root_time_chk_dir, init_time)
		os.makedirs(time_chk_path)
else:
    time_chk_path = None


if flag == "Training":
	print("Train Data")
	traindataset = ImageList(root=configuration['dataset_rootpath'], fileList=train_set, labelPath=dataset_labelpath,
							audList=configuration['dataset_wavspath'], length=configuration['train_params']['seq_length'],
							flag='train', stride=configuration['train_params']['stride'], dilation = configuration['train_params']['dilation'],
							subseq_length = configuration['train_params']['subseq_length'], time_chk_path=time_chk_path)
	trainloader = torch.utils.data.DataLoader(
					traindataset, collate_fn=TrainPadSequence(),
      				**configuration['train_params']['loader_params'])

	print("Val Data")
	valdataset = ImageList_val(root=configuration['dataset_rootpath'], fileList=valid_set, labelPath=dataset_labelpath,
							audList=configuration['dataset_wavspath'], length=configuration['val_params']['seq_length'],
							flag='val', stride=configuration['val_params']['stride'], dilation = configuration['val_params']['dilation'],
							subseq_length = configuration['val_params']['subseq_length'])
	valloader = torch.utils.data.DataLoader(
					valdataset, collate_fn=ValPadSequence(),
     				**configuration['val_params']['loader_params'])
					 
	print("Number of Train samples:" + str(len(traindataset)))
	print("Number of Val samples:" + str(len(valdataset)))
else:
	print("Testing")
	testdataset = ImageList_test(root=configuration['dataset_rootpath'], fileList=test_set, labelPath=dataset_labelpath,
						audList=configuration['dataset_wavspath'], length=configuration['test_params']['seq_length'],
						flag='Test', stride=configuration['test_params']['stride'], dilation = configuration['test_params']['dilation'],
						subseq_length = configuration['test_params']['subseq_length'])

	testloader = torch.utils.data.DataLoader(
				testdataset, collate_fn=TestPadSequence(),
				**configuration['test_params']['loader_params'])
	print("Number of Test samples:" + str(len(testdataset)))
	test_tic = time.time()
	Valid_vacc, Valid_aacc = Test(testloader, model, fusion_model)
	test_toc = time.time()
	print("Test phase took {:.1f} seconds".format(test_toc - test_tic))
	sys.exit()

criterion = CCCLoss(digitize_num=1).cuda()
optimizer = torch.optim.Adam(fusion_model.parameters(),# filter(lambda p: p.requires_grad, multimedia_model.parameters()),
								configuration['model_params']['lr'])

scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

cnt = 0

columns = [	"time",
			"epoch",
			"best_epoch",
			"best_Val_acc",
			"Training_loss",
			"Valid_loss",
			"Training_vacc",
			"Training_aacc",
			"Valid_vacc",
			"Valid_aacc"]

if args.resume == 0:
	init_df = pd.DataFrame(columns=columns)
	csv_name = f'{result_save_path}/{init_time}_seed_{SEED}_{fusion_model_name.lower()}_output.csv'
	save_model_path = f'{weight_save_path}/{init_time}_seed_{SEED}_{fusion_model_name.lower()}_model.pt'
	print("save csv_name : ", csv_name)
	init_df.to_csv(csv_name, index=False)
	chkpt_path = f'{weight_save_path}/Checkpoints/{init_time}_seed_{SEED}_{fusion_model_name.lower()}_chkpt.pth'
else:
	csv_name = os.path.join(result_save_path, args.resume_file.replace("model.pt", "output.csv"))
	print("csv_name : ", csv_name)
	save_model_path = os.path.join(weight_save_path, args.resume_file)
	print("save_model_path : ", save_model_path)
 
	last_epoch_csv = pd.read_csv(csv_name)
	start_epoch = last_epoch_csv['epoch'].values[-1] + 1

	chkpt_path = f"SavedWeights/Checkpoints/{args.resume_file.replace('model.pt','chkpt.pth')}"


if args.check_point == 0:
	lr = 0.0001
else:
	checkpoint = torch.load(chkpt_path)
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']

for epoch in range(start_epoch, total_epoch):
	epoch_tic = time.time()
	logging.info("Epoch")
	logging.info(epoch)

	# train for one epoch
	train_tic = time.time()
	Training_vacc, Training_aacc, Training_loss = train(trainloader, model, criterion, optimizer, scheduler, epoch, lr, fusion_model, time_chk_path=time_chk_path)
	train_toc = time.time()
	print("Train phase took {:.1f} seconds".format(train_toc - train_tic))
	logging.info("Train phase took {:.1f} seconds".format(train_toc - train_tic))

	val_tic = time.time()
	Valid_vacc, Valid_aacc, Valid_loss = validate(valloader, model, criterion, epoch, fusion_model)
	val_toc = time.time()
	print("Val phase took {:.1f} seconds".format(val_toc - val_tic))
	logging.info("Val phase took {:.1f} seconds".format(val_toc - val_tic))

	gc.collect()
	TrainingAccuracy_V.append(Training_vacc)
	TrainingAccuracy_A.append(Training_aacc)
	ValidationAccuracy_V.append(Valid_vacc)
	ValidationAccuracy_A.append(Valid_aacc)

	logging.info('TrainingAccuracy:')
	logging.info(TrainingAccuracy_V)
	logging.info(TrainingAccuracy_A)

	logging.info('ValidationAccuracy:')
	logging.info(ValidationAccuracy_V)
	logging.info(ValidationAccuracy_A)
 
	if (Valid_vacc + Valid_aacc) > np.float32(best_Val_acc):
		print('Saving..')
		print("best_Val_accV: %0.3f" % Valid_vacc)
		print("best_Val_accA: %0.3f" % Valid_aacc)
		state = {
			'net': fusion_model.state_dict() ,
			'best_Val_accV': Valid_vacc,
			'best_Val_accA': Valid_aacc,
			'best_Val_acc_epoch': epoch,
		}
		if not os.path.isdir(weight_save_path):
			os.mkdir(weight_save_path)

		torch.save(state, save_model_path)
		best_Val_acc = Valid_vacc + Valid_aacc
		best_Val_acc_epoch = epoch
	
		checkpoint = {
			'model_state_dict': fusion_model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'epoch': epoch,
			'loss': Training_loss
		}
		torch.save(checkpoint, chkpt_path)
  
	epoch_toc = time.time()
	print("Epoch {}/{} took {:.1f} seconds".format(epoch, total_epoch, epoch_toc - epoch_tic))

	print("best_PrivateTest_acc: %0.3f" % best_Val_acc)
	print("best_PrivateTest_acc_epoch: %d" % best_Val_acc_epoch)	

	now = datetime.now() 
	csv_record_time = now.strftime('%Y%m%d_%H%M%S')
	csv_epoch = epoch
	csv_best_epoch = best_Val_acc_epoch
	csv_best_Val_acc = f"{(best_Val_acc):.4f}"
	csv_Training_loss = f"{Training_loss:.4f}"
	csv_Valid_loss = f"{Valid_loss:.4f}"
	csv_Training_vacc = f"{Training_vacc:.4f}"
	csv_Training_aacc = f"{Training_aacc:.4f}"
	csv_Valid_vacc = f"{Valid_vacc:.4f}"
	csv_Valid_aacc = f"{Valid_aacc:.4f}"
	
	csv_data = [csv_record_time, 
				csv_epoch,
				csv_best_epoch,
				csv_best_Val_acc,
				csv_Training_loss,
				csv_Valid_loss,
				csv_Training_vacc,
				csv_Training_aacc,
				csv_Valid_vacc,
				csv_Valid_aacc]
	
	df = pd.DataFrame([csv_data], columns=columns)
	df.to_csv(csv_name, mode='a', header=False, index=False)
