import torch.utils.data as data
from PIL import Image
import os
import os.path
import sys
import matplotlib.pyplot as plt
import random
import numpy as np
import torchaudio
from torchvision import transforms
import torch
from scipy import signal
from .spec_transform import *
from .clip_transforms import *
import bisect
import cv2
import pandas as pd
import utils.videotransforms as videotransforms
import re
from models.vggish_pytorch import vggish_input
import csv
import math

def get_filename(n):
	filename, ext = os.path.splitext(os.path.basename(n))
	return filename

def default_seq_reader(videoslist, label_path, win_length, stride, dilation, wavs_list):
	shift_length = stride #length-1
	sequences = []
	# csv_data_list = os.listdir(videoslist)#[:2]
	csv_data_list = videoslist
	print("Number of Sequences: " + str(len(set(csv_data_list))))
	for video in csv_data_list:
		#video = 'video74_right.csv'
		if video.startswith('.'):
			continue
		vid_data = pd.read_csv(os.path.join(label_path, video))
		video_data = vid_data.to_dict("list")
		images = video_data['img']
		labels_V = video_data['V']
		labels_A = video_data['A']
		label_arrayV = np.asarray(labels_V, dtype = np.float32)
		label_arrayA = np.asarray(labels_A, dtype = np.float32)
		#medfiltered_labels = signal.medfilt(label_array)
		frame_ids = video_data['frame_id']
		frameid_array = np.asarray(frame_ids, dtype=np.int32)
		#timestamp_file = os.path.join(time_list, get_filename(video) +'_video_ts.txt')
		f_name = get_filename(video)
		if f_name.endswith('_left'):
			wav_file_path = os.path.join(wavs_list, f_name[:-5])
			vidname = f_name[:-5]
		elif f_name.endswith('_right'):
			wav_file_path = os.path.join(wavs_list, f_name[:-6])
			vidname = f_name[:-6]
		else:
			wav_file_path = os.path.join(wavs_list, f_name)
			vidname = f_name

		vid = np.asarray(list(zip(images, label_arrayV, label_arrayA, frameid_array)))
		time_filename = os.path.join('../data/realtimestamps', vidname) + '_video_ts.txt'
		f = open(os.path.join(time_filename))
		lines = f.readlines()[1:]
		length = len(lines) 

		end = 481
		start = end -win_length
		counter = 0
		cnt = 0
		result = []
		while end < length + 482:
			avail_seq_length = end -start
			count = 15
			num_samples = 0
			vis_subsequnces = []
			aud_subsequnces = []
			for i in range(16):
				#subseq_indices.append(np.where((frameid_array>=((i-1)*32)+1) & (frameid_array<=32*i))[0])
				sub_indices = np.where((frameid_array>=(start+(i*32))+1) & (frameid_array<=(end -(count*32))))[0]
				wav_file = os.path.join(wav_file_path, str(end -(count*32))) +'.wav'

				#print(sub_indices)
				if (end -(count*32)) <= length:
					result.append(end -(count*32))
				if ((start+(i*32))+1) <0 and (end -(count*32)) <0:
					vis_subsequnces.append([])
				if len(sub_indices)>=8 and len(sub_indices)<16:
					subseq_indices = sub_indices[-8:]
					vis_subsequnces.append([vid[subseq_indices], (end -(count*32)), f_name, length])
					aud_subsequnces.append(wav_file)
				elif len(sub_indices)>=16 and len(sub_indices)<24:
					subseq_indices = np.flip(np.flip(sub_indices)[::2])
					subseq_indices = subseq_indices[-8:]
					vis_subsequnces.append([vid[subseq_indices], (end -(count*32)), f_name, length])
					aud_subsequnces.append(wav_file)
				elif len(sub_indices)>=24 and len(sub_indices)<32:
					subseq_indices = np.flip(np.flip(sub_indices)[::3])
					subseq_indices = subseq_indices[-8:]
					vis_subsequnces.append([vid[subseq_indices], (end -(count*32)), f_name, length])
					aud_subsequnces.append(wav_file)
				elif len(sub_indices) == 32:
					subseq_indices = np.flip(np.flip(sub_indices)[::4])
					vis_subsequnces.append([vid[subseq_indices], (end -(count*32)), f_name, length])
					aud_subsequnces.append(wav_file)
				elif len(sub_indices) > 0 and len(sub_indices) < 8:
					newList = [sub_indices[-1]]* (8-len(sub_indices))
					sub_indices = np.append(sub_indices, np.array(newList), 0)
					vis_subsequnces.append([vid[sub_indices], (end -(count*32)), f_name, length])
					aud_subsequnces.append(wav_file)
				else:
					vis_subsequnces.append([[], (end -(count*32)),f_name, length])
					aud_subsequnces.append(wav_file)

				count = count - 1

			if (len(aud_subsequnces) < 16):
				print(end -(count*32))
				print(aud_subsequnces)
				sys.exit()
			start_frame_id = start +1
			#wav_file = os.path.join(wav_file_path, str(end)) +'.wav'

			#if len(vis_subsequnces) == 16:
			#sequences.append([subsequnces, wav_file, start_frame_id])
			sequences.append([vis_subsequnces, aud_subsequnces])
			#else:
			#	sequences.append([])
			if avail_seq_length>512:
				print("Wrong Sequence")
				sys.exit()
			counter = counter + 1
			if counter > 31:
				end = end + 480 + shift_length
				start = end - win_length
				#start = start + 224 + shift_length
				#end = start + win_length
				counter = 0
			else:
				end = end + shift_length
				start = end - win_length

		result.sort()
		#print(result)
		#print("-----------")
		#print(len(set(result)))
		#print(length)
		#print(len(sequences))
		#return sequences
		if len(set(result)) == length:
			continue
		else:
			print(video)
			print(len(set(result)))
			print(length)
			print("Seq lengths are wrong")
			sys.exit()
		#	#sequences.append([seq, wav_file, start_frame_id])
	return sequences

def default_list_reader(fileList):
	with open(fileList, 'r') as file:
		#print(fileList)
		video_length = 0
		videos = []
		lines = list(file)
		#print(len(lines))
		for i in range(9):
			line = lines[video_length]
			#print(line)
			#line = file.readlines()[video_length + i]
			imgPath, label = line.strip().split(' ')
			find_str = os.path.dirname(imgPath)
			#print(find_str)
			new_video_length = 0
			for line in lines:
				if find_str in line:
					new_video_length = new_video_length + 1
			#print(new_video_length)Visualmodel_for_Afwild2_bestworkingcode_avail_lab_img_videolevel_perf
			videos.append(lines[video_length:video_length + new_video_length])
			video_length = video_length + new_video_length
			#print(video_length)
	return videos

class ImageList_val(data.Dataset):
	def __init__(self, root, fileList, labelPath, audList, length, flag, stride, dilation, subseq_length, list_reader=default_list_reader, seq_reader=default_seq_reader):
		self.root = root
		#self.label_path = label_path
		self.videoslist = fileList #list_reader(fileList)
		self.label_path = labelPath
		self.win_length = length
		self.num_subseqs = int(self.win_length / subseq_length)
		self.wavs_list = audList
		self.stride = stride
		self.dilation = dilation
		self.subseq_length = int(subseq_length / self.dilation)
		self.sequence_list = seq_reader(self.videoslist, self.label_path, self.win_length, self.stride, self.dilation, self.wavs_list)
		self.sample_rate = 44100
		self.window_size = 20e-3
		self.window_stride = 10e-3
		self.sample_len_secs = 1
		self.sample_len_clipframes = int(self.sample_len_secs * self.sample_rate * self.num_subseqs)
		self.sample_len_frames = int(self.sample_len_secs * self.sample_rate)
		self.audio_shift_sec = 1
		self.audio_shift_samples = int(self.audio_shift_sec * self.sample_rate)

		self.flag = flag

	def __getitem__(self, index):
		seq_path, wav_file = self.sequence_list[index]
		seq, fr_ids, video, vid_lengths, labelV, labelA = self.load_vis_data(self.root, seq_path, self.flag, self.subseq_length)
		aud_data = self.load_aud_data(wav_file, self.num_subseqs, self.flag)
		return seq, aud_data, fr_ids, video, vid_lengths, labelV, labelA#_index

	def __len__(self):
		return len(self.sequence_list)

	def load_vis_data(self, root, SeqPath, flag, subseq_len):
		clip_transform = ComposeWithInvert([NumpyToTensor(),
												 Normalize(mean=[0.43216, 0.394666, 0.37645],
														   std=[0.22803, 0.22145, 0.216989])])
		labV = []
		labA = []
		frame_ids = []
		v_names = []
		v_lengths = []
		seq_length = math.ceil(self.win_length / self.dilation)
		seqs = []
		for clip in SeqPath:
			seq_clip = clip[0]
			frame_id = clip[1]
			v_name = clip[2]
			v_length = clip[3]
			images = np.zeros((8, 112, 112, 3), dtype=np.uint8)
			labelV = -5.0
			labelA = -5.0

			for im_index, image in enumerate(seq_clip):
				imgPath = image[0]
				labelV = image[1]
				labelA = image[2]

				try:
					img = np.array(Image.open(os.path.join(root , imgPath)))
					images[im_index, :, :, 0:3] = img
				except:
					pass

			imgs = clip_transform(images)
			seqs.append(imgs)
			v_names.append(v_name)
			frame_ids.append(frame_id)
			v_lengths.append(v_length)
			labV.append(float(labelV))
			labA.append(float(labelA))

		targetsV = torch.FloatTensor(labV)
		targetsA = torch.FloatTensor(labA)
		imgframe_ids = np.stack(np.asarray(frame_ids))#.permute(4,0,1,2,3)
		vid_seqs = torch.stack(seqs)#.permute(4,0,1,2,3)

		return vid_seqs, imgframe_ids, v_names, v_lengths, targetsV, targetsA # vid_seqs,

	def load_aud_data(self, wav_file, num_subseqs, flag):
		audio_spec_transform = ComposeWithInvert([AmpToDB(), Normalize(mean=[-14.8], std=[19.895])])

		spectrograms = []
		max_spec_shape = []
		if(len(wav_file) < 16):
			print(wav_file)
			sys.exit()
		for wave in wav_file:
			if wave == []:
				audio = torch.zeros((1, 45599))
			elif not os.path.isfile(wave):
				audio = torch.zeros((1, 45599))
			else:
				try:
					audio, sr = torchaudio.load(wave) #,
				except:
					audio, sr = torchaudio.load(wave) #,
			if audio.shape[1] <= 45599:
				_audio = torch.zeros((1, 45599))
				_audio[:, -audio.shape[1]:] = audio
				audio = _audio
			audiofeatures = torchaudio.transforms.MelSpectrogram(sample_rate=44100, win_length=882, hop_length=441, n_mels=64,
												   n_fft=1024, window_fn=torch.hann_window)(audio)

			max_spec_shape.append(audiofeatures.shape[2])

			audio_feature = audio_spec_transform(audiofeatures)

			spectrograms.append(audio_feature)
		spec_dim = max(max_spec_shape)

		audio_features = torch.zeros(len(max_spec_shape), 1, 64, spec_dim)
		for batch_idx, spectrogram in enumerate(spectrograms):
			if spectrogram.shape[2] < spec_dim:
				audio_features[batch_idx, :, :, -spectrogram.shape[2]:] = spectrogram
			else:
				audio_features[batch_idx, :,:, :] = spectrogram
		return audio_features # melspecs_scaled
