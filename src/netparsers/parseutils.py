'''All the parser functions is implemented here'''
import torch.nn as nn
import math
from src.layers.klmodules import *
from src.layers.Initializers import *

def parse_layer_opts(layer_string):
	'''input is a layer description string with name|p1:[v1],p2[v2]... convention'''
	layer_string.rstrip(' ')
	temp = layer_string.split('|')
	layer_name_str = temp[0]
	if len(temp)>1:
		layer_opts_string = temp[1]
		layer_opts_list = layer_opts_string.split(',')
	else:
		layer_opts_list =[]
	layer_opts = {}
	for param_value in layer_opts_list:
		param_value_list = param_value.split(':')
		if len(param_value_list)<2:
			raise Exception(param_value_list[0] + 'is not initialized')
		layer_opts[param_value_list[0]] = param_value_list[1]
	return layer_name_str,layer_opts

def evalpad(pad, ksize):
	if pad == 'same':
		totalpad = ksize - 1
		padding = int(math.floor(totalpad / 2))
	else:
		padding = 0
	return padding

def get_init(initstring:str)->Parameterizer:
	if 'stoch' in initstring:
		isstoch = True
	else:
		isstoch = False
	if 'unif' in initstring:
		isuniform = True
	else:
		isuniform = False
	if 'dirich' in initstring:
		isdirichlet = True
	else:
		isdirichlet = False
	if 'log' in initstring:
		if 'proj' in initstring:
			init = LogParameterProjector(isstoch=isstoch, isuniform=isuniform, isdirichlet=isdirichlet)
		else:
			init = LogParameter(isstoch=isstoch,isuniform=isuniform,isdirichlet=isdirichlet)
	elif 'sphere' in initstring:
		init = SphereParameter(isstoch=isstoch, isuniform=isuniform, isdirichlet=isdirichlet)
	return init

def parse_layer_string(layer_string,in_n_channel,in_icnum,blockidx_dict):
	out_n_channel = -1
	out_icnum = in_icnum
	layer_name_str,layer_opts = parse_layer_opts(layer_string)
	if layer_name_str not in blockidx_dict.keys():
		blockidx_dict[layer_name_str] = 1
	blockidx = blockidx_dict[layer_name_str]
	if layer_name_str == 'fin':
		return None,in_n_channel,out_icnum


	elif layer_name_str == 'conv':
		ksize = int(layer_opts['r'])
		fnum = int(layer_opts['f'])
		stride = int(layer_opts['stride'])
		coef = float(layer_opts['coef'])
		isbiased = ((layer_opts['bias'] == '1')) if 'bias' in layer_opts.keys() else False
		pad = layer_opts['pad']
		layer = FConv(fnum=fnum,
		              input_ch=in_n_channel,
		              spsize=ksize,
		              stride=stride,
		              pad=pad,
		              init_coef=coef,
		              isbiased=isbiased,
		              blockidx=blockidx)
		out_n_channel = fnum

	elif layer_name_str == 'map':
		ksize = int(layer_opts['r'])
		fnum = int(layer_opts['f'])
		stride = int(layer_opts['stride'])
		out_icnum = int(layer_opts['icnum'])
		coef = float(layer_opts['coef'])
		isbiased = ((layer_opts['bias'] == '1')) if 'bias' in layer_opts.keys() else False
		pad = layer_opts['pad']
		layer = BayesFunc(fnum=fnum,
		              input_ch=in_n_channel,
		              icnum= out_icnum,
		              spsize=ksize,
		              stride=stride,
		              pad=pad,
		              init_coef=coef,
		              isbiased=isbiased,
		              blockidx=blockidx)
		out_n_channel = fnum
		out_icnum = 1

	elif layer_name_str == 'smap':
		ksize = int(layer_opts['r'])
		fnum = int(layer_opts['f'])
		stride = int(layer_opts['stride'])
		out_icnum = int(layer_opts['icnum'])
		coef = float(layer_opts['coef'])
		isbiased = ((layer_opts['bias'] == '1')) if 'bias' in layer_opts.keys() else False
		pad = layer_opts['pad']
		layer = SMAP(blockidx=blockidx,
		             out_state=fnum,
		             out_id_comp=out_icnum,
		             in_state=in_n_channel,
		             in_id_comp=in_icnum,
		             rec_field=ksize,
		             stride=stride,
		             pad=pad,
		             init_coef=coef,
		             isbiased=isbiased)
		out_n_channel = fnum
		out_icnum = out_icnum
	elif layer_name_str =='sample':
		layer = Sampler(blockidx=blockidx)
		out_n_channel = in_n_channel
		out_icnum = in_icnum
	elif layer_name_str =='tofin':
		layer = ToFiniteProb(blockidx=blockidx)
		out_n_channel = 2
		out_icnum = in_n_channel
	elif layer_name_str == 'xor':
		ksize = int(layer_opts['r'])
		fnum = int(layer_opts['f'])
		stride = int(layer_opts['stride'])
		out_icnum = int(layer_opts['icnum'])
		coef = float(layer_opts['coef'])
		isbiased = ((layer_opts['bias'] == '1')) if 'bias' in layer_opts.keys() else False
		pad = layer_opts['pad']
		layer = XOR(blockidx=blockidx,
		             out_state=fnum,
		             out_id_comp=out_icnum,
		             in_state=in_n_channel,
		             in_id_comp=in_icnum,
		             rec_field=ksize,
		             stride=stride,
		             pad=pad,
		             init_coef=coef,
		             isbiased=isbiased)
		out_n_channel = fnum
		out_icnum = out_icnum
	elif layer_name_str == 'normalsample':
		layer = NormalVar(in_n_channel,blockidx=blockidx)
		out_n_channel = in_n_channel
		out_icnum = in_icnum





	elif layer_name_str == 'relu':
		layer = FReLU(blockidx=blockidx)
		out_n_channel = in_n_channel
	elif layer_name_str == 'hlnorm':
		alpha = int(layer_opts['alpha'] if 'alpha' in layer_opts.keys() else 1)
		layer = HyperLNorm(alpha,blockidx=blockidx)
		out_n_channel = in_n_channel
	elif layer_name_str == 'sigmoid':
		layer = nn.Sigmoid()
		out_n_channel = in_n_channel
	elif layer_name_str == 'idreg':
		alpha = float(layer_opts['alpha'])
		layer = IDReg(alpha, blockidx=blockidx)
		out_n_channel = in_n_channel
	elif layer_name_str == 'lsigmoid':
		layer = nn.LogSigmoid()
		out_n_channel = in_n_channel
	elif layer_name_str == 'maxpool':
		ksize = int(layer_opts['r'])
		stride = int(layer_opts['stride'])
		pad = layer_opts['pad']
		pad = evalpad(pad,ksize)
		layer = nn.MaxPool2d(kernel_size=ksize, stride=stride, padding=pad)
		out_n_channel = in_n_channel
	elif layer_name_str == 'avgpool':
		ksize = int(layer_opts['r'])
		stride = int(layer_opts['stride'])
		pad = layer_opts['pad']
		pad = evalpad(pad,ksize)
		layer = nn.AvgPool2d(kernel_size=ksize, stride=stride, padding=pad)
		out_n_channel = in_n_channel

	elif layer_name_str == 'klavgpool':
		ksize = int(layer_opts['r'])
		stride = int(layer_opts['stride'])
		pad = layer_opts['pad']
		layer = KLAvgPool(ksize,stride, pad,blockidx= blockidx)
		out_n_channel = in_n_channel
		out_icnum= in_icnum
	elif layer_name_str =='dropout':
		prob = float(layer_opts['p'])
		exact = bool(layer_opts['exact']) if 'exact' in layer_opts.keys() else True
		layer = FDropOut(rate =prob,blockidx=blockidx,exact=exact)
		out_n_channel = in_n_channel
	elif layer_name_str =='bn':

		layer = nn.BatchNorm2d(in_n_channel,track_running_stats=True)
		out_n_channel = in_n_channel
	else:
		raise(Exception('Undefined Layer: ' + layer_name_str))
	if out_n_channel == -1:
		raise('Output Channel Num not assigned in :' + layer_name_str)

	blockidx_dict[layer_name_str] = blockidx_dict[layer_name_str]+1
	return layer, out_n_channel,out_icnum
