########################################################################################
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import torchaudio
from torchvision import transforms
import torch
from datasets.spec_transform import *
from datasets.clip_transforms import *
import pandas as pd
import utils.videotransforms as videotransforms
import math
import gc
import time
import psutil


def get_filename(n):
	filename, ext = os.path.splitext(os.path.basename(n))
	return filename

# def default_seq_reader(videoslist, label_path, win_length, stride, dilation, wavs_list):
def create_jca_seq_data(videoslist, label_path, win_length, stride, dilation, wavs_list):
	'''
	오디오 데이터에 대해서는 _left, _right가 어차피 같은 소리니까 제거해서 파일 접근

	sequnece 길이: 512
	sub-sequence: 16개의 서브 시퀀스로 이루어지며, 최대 32개의 프레임으로 이루어지게 됨 (16 * 32 == 512)
	multi window -> ((16 * n) * (32 / n) == 512)
	clip: 16개로 나뉜 서브 시퀀스 내에서 선택되는 8개의 프레임

	샘플링 방식:
	서브시퀀스별로 8개의 프레임을 선택. 프레임의 개수에 따라 샘플링 방식이 달라짐:
	- 서브시퀀스에 프레임이 32개 -> 4개씩 건너뛰어 8개의 프레임을 선택 (::4)
	- 서브시퀀스에 프레임이 24개 이상 32개 미만 -> 3개씩 건너뛰어 8개의 프레임을 선택 (::3)
	- 서브시퀀스에 프레임이 16개 이상 24개 미만 -> 2개씩 건너뛰어 8개의 프레임을 선택 (::2)
	- 서브시퀀스에 프레임이  8개 이상 16개 미만 -> 마지막 8개의 프레임을 그대로 선택
	- 서브시퀀스에 프레임이  8개 미만이면, 부족한 프레임을 마지막 프레임으로 복사하여 8개 보충

	시퀀스 간의 이동을 처리:
	- avail_seq_length가 512를 초과하면, 시퀀스가 잘못 생성된 경우를 의미하므로 “Wrong Sequence” 출력(비정상적으로 큰 시퀀스는 오류)
	- counter: 시퀀스가 32개의 서브시퀀스로 나누어진 후 몇 번째 서브시퀀스를 처리 중인지 확인하기 위함. 총 16개의 서브시퀀스를 만들면 한 번의 시퀀스가 완성됩니다.	
	- counter가 31을 넘으면, 480 프레임 만큼 시퀀스를 이동. 즉, 시퀀스가 480 + shift_length 프레임 만큼 앞으로 이동 후, start와 end가 재설정. (뒤의 모든 프레임을 보기 위함)
	- counter가 31 이하일 경우, shift_length만큼만 시퀀스를 이동 후, 시퀀스의 시작과 끝을 재설정. (총 32개를 맞추기 위함, collate_fn이라고 생각하면 될듯)
	'''
	shift_length = stride #length-1
	# sub_seq_num * sub_seq_len = 512 (한번의 배치에서 보고자 하는 최대 프레임 개수(시퀀스 길이))
	sub_seq_num = 16
	sub_seq_len = 32
	sequences = []
	csv_data_list = videoslist
	skip_vids = ['313.csv', '212.csv', '303.csv', '171.csv', '40-30-1280x720.csv', '286.csv', '270.csv', '234.csv', '239.csv', '266.csv']

	print("Number of Sequences: " + str(len(set(csv_data_list))))
	for video in csv_data_list:
		if video.startswith('.'):
			continue
		if video in skip_vids:
			continue
		vid_data = pd.read_csv(os.path.join(label_path,video))
  
				# wav file 에러나는 파일만 디렉토리 돌려서 
		if video=="420.csv" or video=="110.csv" or video=="318.csv":
			wavs_list = "../data/Affwild2/SegmendtedAudioFiles/Shift_2_win_32/"
		else:
			wavs_list = "../data/Affwild2/SegmendtedAudioFiles/Shift_1_win_32/"
  
		video_data = vid_data.to_dict("list")
		images = video_data['img']
		labels_V = video_data['V']
		labels_A = video_data['A']
		label_arrayV = np.asarray(labels_V, dtype = np.float32)
		label_arrayA = np.asarray(labels_A, dtype = np.float32)
		frame_ids = video_data['frame_id']
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
		vid = np.asarray(list(zip(images, label_arrayV, label_arrayA)))
		frameid_array = np.asarray(frame_ids, dtype=np.int32)
  
		time_filename = os.path.join('../data/realtimestamps', vidname) + '_video_ts.txt'
		f = open(os.path.join(time_filename))
		lines = f.readlines()[1:]
		length = len(lines) #len(os.listdir(wav_file_path))
  
		end = 481
		start = end -win_length
		counter = 0
		result = []
		while end < length + 481:
			avail_seq_length = end - start
			count = 15
			num_samples = 0
			vis_subsequnces = []
			aud_subsequnces = []
   
			for i in range(sub_seq_num):
				sub_indices = np.where((frameid_array>=(start+(i*sub_seq_len))+1) & (frameid_array<=(end -(count*sub_seq_len))))[0]
				wav_file = os.path.join(wav_file_path, str(end -(count*sub_seq_len))) +'.wav'
				if (end -(count*sub_seq_len)) <= length:
					result.append(end -(count*sub_seq_len))
					# 프레임 8개(clip) 뽑기
					if len(sub_indices)>=8 and len(sub_indices)<16:
						subseq_indices = sub_indices[-8:]
						vis_subsequnces.append(vid[subseq_indices])
						aud_subsequnces.append(wav_file)
					elif len(sub_indices)>=16 and len(sub_indices)<24:
						subseq_indices = np.flip(np.flip(sub_indices)[::2])
						subseq_indices = subseq_indices[-8:]
						vis_subsequnces.append(vid[subseq_indices])
						aud_subsequnces.append(wav_file)
					elif len(sub_indices)>=24 and len(sub_indices)<sub_seq_len:
						subseq_indices = np.flip(np.flip(sub_indices)[::3])
						subseq_indices = subseq_indices[-8:]
						vis_subsequnces.append(vid[subseq_indices])
						aud_subsequnces.append(wav_file)
					elif len(sub_indices) == sub_seq_len:
						subseq_indices = np.flip(np.flip(sub_indices)[::4])
						vis_subsequnces.append(vid[subseq_indices])
						aud_subsequnces.append(wav_file)
					elif len(sub_indices) > 0 and len(sub_indices) < 8:
						newList = [sub_indices[-1]]* (8-len(sub_indices))
						sub_indices = np.append(sub_indices, np.array(newList), 0)
						vis_subsequnces.append(vid[sub_indices])
						aud_subsequnces.append(wav_file)
				count = count - 1

			if len(vis_subsequnces) == 16:
				sequences.append([vis_subsequnces, aud_subsequnces])
			if avail_seq_length>512:
				print("Wrong Sequence")
			
			counter = counter + 1
			if counter > 31:
				end = end + 480 + shift_length
				start = end - win_length
				counter = 0
			else:
				end = end + shift_length
				start = end - win_length

		result.sort()
		if len(set(result)) == length:
			continue
		else:
			print(video)
			print(len(set(result)))
			print(length)
	return sequences

def default_list_reader(fileList):
	with open(fileList, 'r') as file:
		video_length = 0
		videos = []
		lines = list(file)
		for i in range(9):
			line = lines[video_length]
			imgPath, label = line.strip().split(' ')
			find_str = os.path.dirname(imgPath)
			new_video_length = 0
			for line in lines:
				if find_str in line:
					new_video_length = new_video_length + 1
			videos.append(lines[video_length:video_length + new_video_length])
			video_length = video_length + new_video_length
	return videos

class ImageList(data.Dataset):
	# def __init__(self, root, fileList, labelPath, audList, length, flag, stride, dilation, subseq_length, list_reader=default_list_reader, seq_reader=default_seq_reader):
	def __init__(self, root, fileList, labelPath, audList, length, flag, stride, dilation, subseq_length, list_reader=default_list_reader, seq_reader=create_jca_seq_data, time_chk_path=None):
		self.root = root
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
		self.time_chk_path=time_chk_path

	def __getitem__(self, index):
		seq_path, wav_file = self.sequence_list[index]
		st1 = time.time()
		seq, label_V, label_A = self.load_vis_data(self.root, seq_path, self.flag)
		ed1 = time.time()
		st2 = time.time()
		aud_data = self.load_aud_data(wav_file, self.num_subseqs, self.flag)
		ed2 = time.time()

		vis_data_time = ed1 - st1
		aud_data_time = ed2 - st2
		if self.time_chk_path:
			time_chk_file = os.path.join(self.time_chk_path, "time_chk.txt")
   
			current_thread = psutil.Process().threads()[0]
			core_number = psutil.Process().cpu_num()
  
			with open(time_chk_file, 'a') as f:
				if vis_data_time > 2:
					seq_list = []
					for se in seq_path:
						seq_list = [s[0] for s in se]
					f.write(f"Time load vis_data: {vis_data_time:.4f}\t core: {core_number}\t seq_path: {seq_list}\n")
				else:
					f.write(f"Time load vis_data: {vis_data_time:.4f}\n")       
				if aud_data_time > 2:
					t_wav = [wf.split("/")[-2] + '/' + wf.split("/")[-1] for wf in wav_file]
					f.write(f"Time load aud_data: {aud_data_time:.4f}\t core: {core_number}\t wav_file: {t_wav}\n")       
				else:
					f.write(f"Time load aud_data: {aud_data_time:.4f}\n")
			f.close()
		return seq, aud_data, label_V, label_A #_index

	def __len__(self):
		return len(self.sequence_list)

	def load_vis_data(self, root, SeqPath, flag):
		clip_transform = ComposeWithInvert([NumpyToTensor(),
												 Normalize(mean=[0.43216, 0.394666, 0.37645],
														   std=[0.22803, 0.22145, 0.216989])])
		if (flag == 'train'):
			data_transforms = transforms.Compose([videotransforms.RandomCrop(224),
										   videotransforms.RandomHorizontalFlip()])
		else:
			data_transforms=transforms.Compose([videotransforms.CenterCrop(224)])
		labV = []
		labA = []
		seqs = []
		for clip in SeqPath:
			images = np.zeros((8, 112, 112, 3), dtype=np.uint8)
			labelV = -5.0
			labelA = -5.0
			for im_index, image in enumerate(clip):
				imgPath = image[0]
				labelV = image[1]
				labelA = image[2]

				try:
					img = np.array(Image.open(os.path.join(root , imgPath)))
					images[im_index, :, :, 0:3] = img
				except:
					pass

			imgs = clip_transform(RandomColorAugmentation(images))
			seqs.append(imgs)
			labV.append(float(labelV))
			labA.append(float(labelA))

		targetsV = torch.FloatTensor(labV)
		targetsA = torch.FloatTensor(labA)
		vid_seqs = torch.stack(seqs)#.permute(4,0,1,2,3)
		gc.collect()

		return vid_seqs, targetsV, targetsA # vid_seqs,

	def load_aud_data(self, wav_file, num_subseqs, flag):
		transform_spectra = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomVerticalFlip(1),
			transforms.ToTensor(),
		])
		audio_spec_transform = ComposeWithInvert([AmpToDB(), Normalize(mean=[-14.8], std=[19.895])])

		spectrograms = []
		max_spec_shape = []
		for wave in wav_file:
			try:
				audio, sr = torchaudio.load(wave) #,
			except:
				audio, sr = torchaudio.load(wave) #,
			if audio.shape[1] <= 45599:
				_audio = torch.zeros((1, 45599))
				_audio[:, -audio.shape[1]:] = audio
				audio = _audio
			audiofeatures = torchaudio.transforms.MelSpectrogram(sample_rate=sr, win_length=882, hop_length=441, n_mels=64,
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

