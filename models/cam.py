from __future__ import absolute_import
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import functional as F
import sys
from .layer import LSTM
from torch.nn import init

from .audguide_att import BottomUpExtract

class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        # self.corr_weights = torch.nn.Parameter(torch.empty(
        #        1024, 1024, requires_grad=True).type(torch.cuda.FloatTensor))
        self.encoder1 = nn.Linear(512, 128)
        self.encoder2 = nn.Linear(512, 128)
        self.reduce_visual = nn.Linear(25088, 512)

        self.affine_a = nn.Linear(16, 16, bias=False)
        self.affine_v = nn.Linear(16, 16, bias=False)

        self.W_a = nn.Linear(16, 32, bias=False)
        self.W_v = nn.Linear(16, 32, bias=False)
        self.W_ca = nn.Linear(256, 32, bias=False)
        self.W_cv = nn.Linear(256, 32, bias=False)

        self.W_ha = nn.Linear(32, 16, bias=False)
        self.W_hv = nn.Linear(32, 16, bias=False)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        # self.vregressor = nn.Sequential(nn.Linear(640, 128),
        self.vregressor = nn.Sequential(nn.Linear(256, 128),
                                     nn.Dropout(0.6),
                                 nn.Linear(128, 1))

        # self.aregressor = nn.Sequential(nn.Linear(640, 128),
        self.aregressor = nn.Sequential(nn.Linear(256, 128),
                                     nn.Dropout(0.6),
                                 nn.Linear(128, 1))
        
        self.init_weights()


    def init_weights(net, init_type='xavier', init_gain=1):

        if torch.cuda.is_available():
            net.cuda()

        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.uniform_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_uniform_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>


    def forward(self, f1_norm, f2_norm):
        fin_audio_features = []
        fin_visual_features = []
        vsequence_outs = []
        asequence_outs = []

        for i in range(f1_norm.shape[0]):
            audfts = f1_norm[i,:,:] #.transpose(0,1)
            visfts = f2_norm[i,:,:] #.transpose(0,1)
            
            aud_fts = self.encoder1(audfts)
            vis_fts = self.encoder2(self.reduce_visual(visfts))

            aud_vis_fts = torch.cat((aud_fts, vis_fts), 1) # J
            
            a_t = self.affine_a(aud_fts.transpose(0,1))
            v_t = self.affine_v(vis_fts.transpose(0,1))
            
            att_aud = torch.mm(aud_vis_fts.transpose(0,1), a_t.transpose(0,1)) # X_a^T W_ja J
            att_vis = torch.mm(aud_vis_fts.transpose(0,1), v_t.transpose(0,1)) # X_v^T W_jv J
            
            audio_att = self.tanh(torch.div(att_aud, math.sqrt(aud_vis_fts.shape[1]))) # C_a
            vis_att = self.tanh(torch.div(att_vis, math.sqrt(aud_vis_fts.shape[1]))) # C_v

            H_a = self.relu(self.W_ca(audio_att.transpose(0,1)) + self.W_a(aud_fts.transpose(0,1)))
            H_v = self.relu(self.W_ca(vis_att.transpose(0,1)) + self.W_a(vis_fts.transpose(0,1)))

            att_audio_features = self.W_ha(H_a).transpose(0,1) + aud_fts
            att_visual_features = self.W_hv(H_v).transpose(0,1) + vis_fts
            
            audiovisualfeatures = torch.cat((att_audio_features, att_visual_features), -1)
            audiovisualfeatures = audiovisualfeatures.unsqueeze(1)

            vouts = self.vregressor(audiovisualfeatures) #.transpose(0,1))
            aouts = self.aregressor(audiovisualfeatures) #.transpose(0,1))
            
            vsequence_outs.append(vouts)
            asequence_outs.append(aouts)
            fin_audio_features.append(att_audio_features)
            fin_visual_features.append(att_visual_features)
            
        vfinal_outs = torch.stack(vsequence_outs)
        afinal_outs = torch.stack(asequence_outs)
        
        return vfinal_outs.squeeze(2).squeeze(2), afinal_outs.squeeze(2).squeeze(2)
        # return vfinal_outs.squeeze(2), afinal_outs.squeeze(2)
        

class LSTM_CAM(nn.Module):
    def __init__(self):
        super(LSTM_CAM, self).__init__()
        # self.corr_weights = torch.nn.Parameter(torch.empty(
        #        1024, 1024, requires_grad=True).type(torch.cuda.FloatTensor))
        self.audio_extract = LSTM(512, 512, 2, 0.1, residual_embeddings=True)
        self.video_extract = LSTM(512, 512, 2, 0.1, residual_embeddings=True)
        self.video_attn = BottomUpExtract(512, 512) 
        

        # self.encoder1 = nn.Linear(512, 128)
        # self.encoder2 = nn.Linear(512, 128)
        # self.reduce_visual = nn.Linear(25088, 512)
        self.encoder1 = nn.Linear(512, 256)
        self.encoder2 = nn.Linear(512, 256)
        self.reduce_visual = nn.Linear(25088, 512)


        self.affine_a = nn.Linear(16, 16, bias=False)
        self.affine_v = nn.Linear(16, 16, bias=False)

        self.W_a = nn.Linear(16, 32, bias=False)
        self.W_v = nn.Linear(16, 32, bias=False)
        # self.W_ca = nn.Linear(256, 32, bias=False)
        # self.W_cv = nn.Linear(256, 32, bias=False)
        self.W_ca = nn.Linear(512, 32, bias=False)
        self.W_cv = nn.Linear(512, 32, bias=False)

        self.W_ha = nn.Linear(32, 16, bias=False)
        self.W_hv = nn.Linear(32, 16, bias=False)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        # self.vregressor = nn.Sequential(nn.Linear(640, 128),
        # self.vregressor = nn.Sequential(nn.Linear(256, 128),
        #                              nn.Dropout(0.6),
        #                          nn.Linear(128, 1))
        self.vregressor = nn.Sequential(nn.Linear(512, 256),
                                     nn.Dropout(0.6),
                                 nn.Linear(256, 1))

        # self.aregressor = nn.Sequential(nn.Linear(640, 128),
        # self.aregressor = nn.Sequential(nn.Linear(256, 128),
        #                              nn.Dropout(0.6),
        #                          nn.Linear(128, 1))
        self.aregressor = nn.Sequential(nn.Linear(512, 256),
                                     nn.Dropout(0.6),
                                 nn.Linear(256, 1))
        
        self.Joint = LSTM(512, 256, 2, dropout=0, residual_embeddings=True)

        self.init_weights()


    def init_weights(net, init_type='xavier', init_gain=1):

        if torch.cuda.is_available():
            net.cuda()

        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.uniform_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_uniform_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>


    def forward(self, f1_norm, f2_norm):
        fin_audio_features = []
        fin_visual_features = []
        vsequence_outs = []
        asequence_outs = []

        for i in range(f1_norm.shape[0]):
            audfts = f1_norm[i,:,:].unsqueeze(1) #.transpose(0,1)
            visfts = f2_norm[i,:,:].unsqueeze(1) #.transpose(0,1)            
            
            audfts = self.audio_extract(audfts)
            visfts = self.video_attn(visfts, audfts)
            visfts = self.video_extract(visfts)  
            
            audfts = audfts.squeeze(1)  # (sequence_length, input_size)
            visfts = visfts.squeeze(1)  # (sequence_length, input_size)
            # audfts = f1_norm[i,:,:]
            # visfts = f2_norm[i,:,:]
            
            # print("visfts : ", visfts.shape)
            
            aud_fts = self.encoder1(audfts)
            vis_fts = self.encoder2(visfts)
            # vis_fts = self.encoder2(self.reduce_visual(visfts))

            aud_vis_fts = torch.cat((aud_fts, vis_fts), 1) # J
            
            # affine이 왜 필요한거지?
            a_t = self.affine_a(aud_fts.transpose(0,1))
            v_t = self.affine_v(vis_fts.transpose(0,1))
            
            att_aud = torch.mm(aud_vis_fts.transpose(0,1), a_t.transpose(0,1)) # X_a^T W_ja J
            att_vis = torch.mm(aud_vis_fts.transpose(0,1), v_t.transpose(0,1)) # X_v^T W_jv J
            
            audio_att = self.tanh(torch.div(att_aud, math.sqrt(aud_vis_fts.shape[1]))) # C_a
            vis_att = self.tanh(torch.div(att_vis, math.sqrt(aud_vis_fts.shape[1]))) # C_v

            H_a = self.relu(self.W_ca(audio_att.transpose(0,1)) + self.W_a(aud_fts.transpose(0,1)))
            H_v = self.relu(self.W_ca(vis_att.transpose(0,1)) + self.W_a(vis_fts.transpose(0,1)))

            att_audio_features = self.W_ha(H_a).transpose(0,1) + aud_fts
            att_visual_features = self.W_hv(H_v).transpose(0,1) + vis_fts
            
            audiovisualfeatures = torch.cat((att_audio_features, att_visual_features), -1) #(seq_len, batch_size, 2 * feature_size)
            audiovisualfeatures = audiovisualfeatures.unsqueeze(1)
            
            audiovisualfeatures = self.Joint(audiovisualfeatures)
   
            fin_audio_features.append(att_audio_features)
            fin_visual_features.append(att_visual_features)
        
        # Stack features to get (batch_size, seq_len, feature_size)
        att_audio_features = torch.stack(fin_audio_features, dim=1)
        att_visual_features = torch.stack(fin_visual_features, dim=1)
        
        audiovisualfeatures = torch.cat((att_audio_features, att_visual_features), -1) # (seq_len, batch_size, 2 * feature_size)
        # audiovisualfeatures = self.Joint(audiovisualfeatures)

        # print("audiovisualfeatures : ", audiovisualfeatures.shape)
        
        vouts = self.vregressor(audiovisualfeatures) # (seq_len, batch_size, 1)
        aouts = self.aregressor(audiovisualfeatures) # (seq_len, batch_size, 1)
        
        return vouts.squeeze(2), aouts.squeeze(2)

#################################################################

# class LSTM_CAM(nn.Module):
#     def __init__(self):
#         super(LSTM_CAM, self).__init__()
#         # self.corr_weights = torch.nn.Parameter(torch.empty(
#         #        1024, 1024, requires_grad=True).type(torch.cuda.FloatTensor))
#         self.audio_extract = LSTM(512, 512, 2, 0.1, residual_embeddings=True)
#         self.video_extract = LSTM(512, 512, 2, 0.1, residual_embeddings=True)       
#         self.video_attn = BottomUpExtract(512, 512) 
        

#         # self.encoder1 = nn.Linear(512, 128)
#         # self.encoder2 = nn.Linear(512, 128)
#         # self.reduce_visual = nn.Linear(25088, 512)
#         self.encoder1 = nn.Linear(512, 256)
#         self.encoder2 = nn.Linear(512, 256)
#         self.reduce_visual = nn.Linear(25088, 512)


#         self.affine_a = nn.Linear(16, 16, bias=False)
#         self.affine_v = nn.Linear(16, 16, bias=False)

#         self.W_a = nn.Linear(16, 32, bias=False)
#         self.W_v = nn.Linear(16, 32, bias=False)
#         # self.W_ca = nn.Linear(256, 32, bias=False)
#         # self.W_cv = nn.Linear(256, 32, bias=False)
#         self.W_ca = nn.Linear(512, 32, bias=False)
#         self.W_cv = nn.Linear(512, 32, bias=False)

#         self.W_ha = nn.Linear(32, 16, bias=False)
#         self.W_hv = nn.Linear(32, 16, bias=False)

#         self.tanh = nn.Tanh()
#         self.relu = nn.ReLU()
#         # self.vregressor = nn.Sequential(nn.Linear(640, 128),
#         self.vregressor = nn.Sequential(nn.Linear(256, 128),
#                                      nn.Dropout(0.6),
#                                  nn.Linear(128, 1))
#         # self.vregressor = nn.Sequential(nn.Linear(512, 256),
#         #                              nn.Dropout(0.6),
#         #                          nn.Linear(256, 1))

#         # self.aregressor = nn.Sequential(nn.Linear(640, 128),
#         self.aregressor = nn.Sequential(nn.Linear(256, 128),
#                                      nn.Dropout(0.6),
#                                  nn.Linear(128, 1))
#         # self.aregressor = nn.Sequential(nn.Linear(512, 256),
#         #                              nn.Dropout(0.6),
#         #                          nn.Linear(256, 1))
        
#         self.Joint = LSTM(512, 256, 2, dropout=0, residual_embeddings=True)

#         self.init_weights()


#     def init_weights(net, init_type='xavier', init_gain=1):

#         if torch.cuda.is_available():
#             net.cuda()

#         def init_func(m):  # define the initialization function
#             classname = m.__class__.__name__
#             if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#                 if init_type == 'normal':
#                     init.uniform_(m.weight.data, 0.0, init_gain)
#                 elif init_type == 'xavier':
#                     init.xavier_uniform_(m.weight.data, gain=init_gain)
#                 elif init_type == 'kaiming':
#                     init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
#                 elif init_type == 'orthogonal':
#                     init.orthogonal_(m.weight.data, gain=init_gain)
#                 else:
#                     raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#                 if hasattr(m, 'bias') and m.bias is not None:
#                     init.constant_(m.bias.data, 0.0)

#         print('initialize network with %s' % init_type)
#         net.apply(init_func)  # apply the initialization function <init_func>


#     def forward(self, f1_norm, f2_norm):
#         fin_audio_features = []
#         fin_visual_features = []
#         vsequence_outs = []
#         asequence_outs = []
        
#         audfts = f1_norm #.transpose(0,1)
#         visfts = f2_norm #.transpose(0,1)      
        
#         audfts = self.audio_extract(audfts)
#         visfts = self.video_attn(visfts, audfts)
#         visfts = self.video_extract(visfts)        

#         for i in range(f1_norm.shape[0]):
#             # audfts = f1_norm[i,:,:].unsqueeze(1) #.transpose(0,1)
#             # visfts = f2_norm[i,:,:].unsqueeze(1) #.transpose(0,1)            
            
#             # audfts = audfts.squeeze(1)  # (sequence_length, input_size)
#             # visfts = visfts.squeeze(1)  # (sequence_length, input_size)
#             audfts = f1_norm[i,:,:]
#             visfts = f2_norm[i,:,:]
            
#             aud_fts = self.encoder1(audfts)
#             # vis_fts = self.encoder2(visfts)
#             vis_fts = self.encoder2(self.reduce_visual(visfts))

#             aud_vis_fts = torch.cat((aud_fts, vis_fts), 1) # J
            
#             # affine이 왜 필요한거지?
#             a_t = self.affine_a(aud_fts.transpose(0,1))
#             v_t = self.affine_v(vis_fts.transpose(0,1))
            
#             att_aud = torch.mm(aud_vis_fts.transpose(0,1), a_t.transpose(0,1)) # X_a^T W_ja J
#             att_vis = torch.mm(aud_vis_fts.transpose(0,1), v_t.transpose(0,1)) # X_v^T W_jv J
            
#             audio_att = self.tanh(torch.div(att_aud, math.sqrt(aud_vis_fts.shape[1]))) # C_a
#             vis_att = self.tanh(torch.div(att_vis, math.sqrt(aud_vis_fts.shape[1]))) # C_v

#             H_a = self.relu(self.W_ca(audio_att.transpose(0,1)) + self.W_a(aud_fts.transpose(0,1)))
#             H_v = self.relu(self.W_ca(vis_att.transpose(0,1)) + self.W_a(vis_fts.transpose(0,1)))

#             att_audio_features = self.W_ha(H_a).transpose(0,1) + aud_fts
#             att_visual_features = self.W_hv(H_v).transpose(0,1) + vis_fts
   
#             fin_audio_features.append(att_audio_features)
#             fin_visual_features.append(att_visual_features)
        
#         # Stack features to get (batch_size, seq_len, feature_size)
#         att_audio_features = torch.stack(fin_audio_features, dim=1)
#         att_visual_features = torch.stack(fin_visual_features, dim=1)
        
#         audiovisualfeatures = torch.cat((att_audio_features, att_visual_features), -1) # (seq_len, batch_size, 2 * feature_size)
#         audiovisualfeatures = self.Joint(audiovisualfeatures)
        
        
#         vouts = self.vregressor(audiovisualfeatures) # (seq_len, batch_size, 1)
#         aouts = self.aregressor(audiovisualfeatures) # (seq_len, batch_size, 1)
        
#         return vouts.squeeze(2), aouts.squeeze(2)