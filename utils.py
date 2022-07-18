from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import shutil
import numpy as np
import torch

import os
import shutil
import numpy as np
import torch
import os
import sys
import time
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dst

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val   = 0
		self.avg   = 0
		self.sum   = 0
		self.count = 0

	def update(self, val, n=1):
		self.val   = val
		self.sum   += val * n
		self.count += n
		self.avg   = self.sum / self.count


def count_parameters_in_MB(model):
	return sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6


def create_exp_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)
	print('Experiment dir : {}'.format(path))


def load_pretrained_model(model, pretrained_dict):
	model_dict = model.state_dict()
	# 1. filter out unnecessary keys
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict) 
	# 3. load the new state dict
	model.load_state_dict(model_dict)


def transform_time(s):
	m, s = divmod(int(s), 60)
	h, m = divmod(m, 60)
	return h,m,s


def save_checkpoint(state, is_best, save_root):
	save_path = os.path.join(save_root, 'checkpoint.pth.tar')
	torch.save(state, save_path)
	if is_best:
		best_save_path = os.path.join(save_root, 'model_best.pth.tar')
		shutil.copyfile(save_path, best_save_path)


class SoftTarget(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self):
		super(SoftTarget, self).__init__()
		self.T = 4

	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss
   
def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred    = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


def L2Loss(y,yhead):
    return torch.mean((y-yhead)**2)

def ICKDLoss(f_s, f_t):
    bsz, ch = f_s.shape[0], f_s.shape[1]

    f_s = f_s.view(bsz, ch, -1)
    f_t = f_t.view(bsz, ch, -1)

    emd_s = torch.bmm(f_s, f_s.permute(0, 2, 1))
    emd_s = torch.nn.functional.normalize(emd_s, dim=2)

    emd_t = torch.bmm(f_t, f_t.permute(0, 2, 1))
    emd_t = torch.nn.functional.normalize(emd_t, dim=2)

    G_diff = emd_s - emd_t
    loss = (G_diff * G_diff).view(bsz, -1).sum() / (ch * bsz)
    return loss



def intra_fd(f_s, f_t):
    s_H, t_H = f_s.shape[2], f_t.shape[2]
    s_C, t_C = f_s.shape[1], f_t.shape[1]
    if s_H > t_H:
        f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    elif s_H < t_H:
        f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
    else:
        pass
    return L2Loss(f_s[:, 0:min(s_C,t_C), :, :], f_t[:, 0:min(s_C,t_C), :, :])

def L2Loss(y,yhead):
    return torch.mean((y-yhead)**2)

def inter_fd(f_s):
    sorted_s, indices_s = torch.sort(torch.nn.functional.normalize(f_s, p=2, dim=(2,3)).mean([0, 2, 3]), dim=0, descending=True)
    # sorted_s, indices_s = torch.sort(f_s.mean([0, 2, 3]), dim=0, descending=True)
    f_s = torch.index_select(f_s, 1, indices_s)
    loss = L2Loss(f_s[:, 0:f_s.shape[1]//2, :, :].detach(), f_s[:, f_s.shape[1]//2: f_s.shape[1], :, :])
    return loss


class NST(nn.Module):
    '''
    Like What You Like: Knowledge Distill via Neuron Selectivity Transfer
    https://arxiv.org/pdf/1707.01219.pdf
    '''

    def __init__(self):
        super(NST, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), fm_s.size(1), -1)
        fm_s = F.normalize(fm_s, dim=2)

        fm_t = fm_t.view(fm_t.size(0), fm_t.size(1), -1)
        fm_t = F.normalize(fm_t, dim=2)

        loss = self.poly_kernel(fm_t, fm_t).mean() \
               + self.poly_kernel(fm_s, fm_s).mean() \
               - 2 * self.poly_kernel(fm_s, fm_t).mean()

        return loss

    def poly_kernel(self, fm1, fm2):
        fm1 = fm1.unsqueeze(1)
        fm2 = fm2.unsqueeze(2)
        out = (fm1 * fm2).sum(-1).pow(2)

        return out


class PKTCosSim(nn.Module):
    '''
    Learning Deep Representations with Probabilistic Knowledge Transfer
    http://openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.pdf
    '''

    def __init__(self):
        super(PKTCosSim, self).__init__()

    def forward(self, feat_s, feat_t, eps=1e-6):
        # Normalize each vector by its norm
        feat_s_norm = torch.sqrt(torch.sum(feat_s ** 2, dim=1, keepdim=True))
        feat_s = feat_s / (feat_s_norm + eps)
        feat_s[feat_s != feat_s] = 0

        feat_t_norm = torch.sqrt(torch.sum(feat_t ** 2, dim=1, keepdim=True))
        feat_t = feat_t / (feat_t_norm + eps)
        feat_t[feat_t != feat_t] = 0

        # Calculate the cosine similarity
        feat_s_cos_sim = torch.mm(feat_s, feat_s.transpose(0, 1))
        feat_t_cos_sim = torch.mm(feat_t, feat_t.transpose(0, 1))

        # Scale cosine similarity to [0,1]
        feat_s_cos_sim = (feat_s_cos_sim + 1.0) / 2.0
        feat_t_cos_sim = (feat_t_cos_sim + 1.0) / 2.0

        # Transform them into probabilities
        feat_s_cond_prob = feat_s_cos_sim / torch.sum(feat_s_cos_sim, dim=1, keepdim=True)
        feat_t_cond_prob = feat_t_cos_sim / torch.sum(feat_t_cos_sim, dim=1, keepdim=True)

        # Calculate the KL-divergence
        loss = torch.mean(feat_t_cond_prob * torch.log((feat_t_cond_prob + eps) / (feat_s_cond_prob + eps)))

        return loss


'''
AT with sum of absolute values with power p
'''


class AT(nn.Module):
    '''
    Paying More Attention to Attention: Improving the Performance of Convolutional
    Neural Netkworks wia Attention Transfer
    https://arxiv.org/pdf/1612.03928.pdf
    '''

    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(F.normalize(fm_s.pow(2).mean(1).view(fm_s.size(0), -1)),
                          F.normalize(fm_t.pow(2).mean(1).view(fm_t.size(0), -1)))

        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)

        return am





class Hint(nn.Module):
    '''
    FitNets: Hints for Thin Deep Nets
    https://arxiv.org/pdf/1412.6550.pdf
    '''

    def __init__(self):
        super(Hint, self).__init__()

    def forward(self, fm_s, fm_t):
        # fm_s, fm_t =  dropout(fm_s, fm_t, 0.1)
        loss = F.mse_loss(fm_s, fm_t)
        return loss


def loss_kd(outputs, labels, teacher_outputs, params):
    """
    loss function for Knowledge Distillation (KD)
    """
    alpha = params.alpha
    T = params.temperature

    loss_CE = F.cross_entropy(outputs, labels)
    D_KL = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1)) * (T * T)
    KD_loss = (1. - alpha) * loss_CE + alpha * D_KL

    return KD_loss


def loss_kd_self(outputs, labels, teacher_outputs, params):
    """
    loss function for self training: Tf-KD_{self}
    """
    alpha = params.alpha
    T = params.temperature

    loss_CE = F.cross_entropy(outputs, labels)
    D_KL = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1)) * (
                T * T) * params.multiplier  # multiple is 1.0 in most of cases, some cases are 10 or 50
    KD_loss = (1. - alpha) * loss_CE + alpha * D_KL

    return KD_loss


def loss_kd_regularization(outputs, labels, params):
    """
    loss function for mannually-designed regularization: Tf-KD_{reg}
    """
    alpha = params.reg_alpha
    T = params.reg_temperature
    correct_prob = 0.99  # the probability for correct class in u(k)
    loss_CE = F.cross_entropy(outputs, labels)
    K = outputs.size(1)

    teacher_soft = torch.ones_like(outputs).cuda()
    teacher_soft = teacher_soft * (1 - correct_prob) / (K - 1)  # p^d(k)
    for i in range(outputs.shape[0]):
        teacher_soft[i, labels[i]] = correct_prob
    loss_soft_regu = nn.KLDivLoss()(F.log_softmax(outputs, dim=1),
                                    F.softmax(teacher_soft / T, dim=1)) * params.multiplier

    KD_loss = (1. - alpha) * loss_CE + alpha * loss_soft_regu

    return KD_loss


def loss_label_smoothing(outputs, labels):
    """
    loss function for label smoothing regularization
    """
    alpha = 0.1
    N = outputs.size(0)  # batch_size
    C = outputs.size(1)  # number of classes
    smoothed_labels = torch.full(size=(N, C), fill_value=alpha / (C - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1 - alpha)

    log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / N

    return loss


class SP(nn.Module):
    '''
    Similarity-Preserving Knowledge Distillation
    https://arxiv.org/pdf/1907.09682.pdf
    '''

    def __init__(self):
        super(SP, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        G_s = torch.mm(fm_s, fm_s.t())
        norm_G_s = F.normalize(G_s, p=2, dim=1)

        fm_t = fm_t.view(fm_t.size(0), -1)
        G_t = torch.mm(fm_t, fm_t.t())
        norm_G_t = F.normalize(G_t, p=2, dim=1)

        loss = F.mse_loss(norm_G_s, norm_G_t)

        return loss


class AB(nn.Module):
    '''
    Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons
    https://arxiv.org/pdf/1811.03233.pdf
    '''

    def __init__(self, margin):
        super(AB, self).__init__()

        self.margin = margin

    def forward(self, fm_s, fm_t):
        # fm befor activation
        loss = ((fm_s + self.margin).pow(2) * ((fm_s > -self.margin) & (fm_t <= 0)).float() +
                (fm_s - self.margin).pow(2) * ((fm_s <= self.margin) & (fm_t > 0)).float())
        loss = loss.mean()

        return loss


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val   = 0
		self.avg   = 0
		self.sum   = 0
		self.count = 0

	def update(self, val, n=1):
		self.val   = val
		self.sum   += val * n
		self.count += n
		self.avg   = self.sum / self.count


def count_parameters_in_MB(model):
	return sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6


def create_exp_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)
	print('Experiment dir : {}'.format(path))


def load_pretrained_model(model, pretrained_dict):
	model_dict = model.state_dict()
	# 1. filter out unnecessary keys
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict) 
	# 3. load the new state dict
	model.load_state_dict(model_dict)


def transform_time(s):
	m, s = divmod(int(s), 60)
	h, m = divmod(m, 60)
	return h,m,s


def save_checkpoint(state, is_best, save_root):
	save_path = os.path.join(save_root, 'checkpoint.pth.tar')
	torch.save(state, save_path)
	if is_best:
		best_save_path = os.path.join(save_root, 'model_best.pth.tar')
		shutil.copyfile(save_path, best_save_path)


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred    = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].contiguous().view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res



def ICKDLoss(f_s, f_t):
    bsz, ch = f_s.shape[0], f_s.shape[1]

    f_s = f_s.view(bsz, ch, -1)
    f_t = f_t.view(bsz, ch, -1)

    emd_s = torch.bmm(f_s, f_s.permute(0, 2, 1))
    emd_s = torch.nn.functional.normalize(emd_s, dim=2)

    emd_t = torch.bmm(f_t, f_t.permute(0, 2, 1))
    emd_t = torch.nn.functional.normalize(emd_t, dim=2)

    G_diff = emd_s - emd_t
    loss = (G_diff * G_diff).view(bsz, -1).sum() / (ch * bsz)
    return torch.mean((f_s-f_t)**2)



def inter_fd(f_s, f_t):
    s_H, t_H = f_s.shape[2], f_t.shape[2]
    if s_H > t_H:
        f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    elif s_H < t_H:
        f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
    else:
        pass
    s_C, t_C = f_s.shape[1], f_t.shape[1]
    if s_C > t_C:
        loss = L2Loss(f_s[:, 0:t_C, :, :], f_t)
    elif s_C < t_C:
        loss = L2Loss(f_s, f_t[:, 0:s_C, :, :])
    else:
        pass
    return loss

def L2Loss(y,yhead):
    return torch.mean((y-yhead)**2)

def intra_fd(f_s):
    sorted_s, indices_s = torch.sort(torch.nn.functional.normalize(f_s, p=2, dim=(2,3)).mean([0, 2, 3]), dim=0, descending=True)
    f_s = torch.index_select(f_s, 1, indices_s)
    loss = L2Loss(f_s[:, 0:f_s.shape[1]//2, :, :], f_s[:, f_s.shape[1]//2: f_s.shape[1], :, :])
    return loss

