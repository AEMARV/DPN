from src.optstructs import DataOpts,NetOpts, OptimOpts
from src.trainvalid.lr_schedulers import *
from torch.nn.modules import NLLLoss
from typing import Tuple
import torchvision.transforms as transforms
''' Available Options for KLCONV('klconv'):
	r: receptive field,
	f: number of filters,
	pad: padding amount either 'same' or 'valid'
	bias: does it have bias (Bias not implemented yet)
	
	param: selects the parameterization methods including->
	
	 1. 'logstochu': log parameterization, stochastic gradient
	 and all dists are initialized to uniform
	 2. 'logstoch' : log parameterizaiton, stochastic gradient
	 3. 'log' : log parametrizaiton, exponential dists initialization  
	
	
	
	'''

def quick_cifar(data_opts:DataOpts) -> Tuple[NetOpts,OptimOpts]:
	model_string = ''
	nl = 'relu'
	d = '->'
	finish = 'fin'
	model_string += 'conv|r:5,f:32,pad:same,bias:1' + d
	model_string += 'maxpool|r:3,f:32,pad:same,stride:2,bias:1' + d
	model_string += 'conv|r:5,f:64,pad:same,bias:1' + d + nl + d
	model_string += 'avgpool|r:3,f:32,pad:same,stride:2,bias:1' + d
	model_string += 'conv|r:4,f:64,pad:same,bias:1' + d + nl + d
	model_string += 'conv|r:7,f:'+ str(data_opts.classnum) + ',pad:valid,bias:1' + d+ 'lnorm|s:0'+d
	model_string += finish


	'''Data OPTs'''
	data_transforms = [transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]

	'''LR SCHED'''
	initlr = .01
	lr_sched = exp_decay_lr(init_lr=initlr,step=20,exp_decay_perstep=1)

	''' Net Options'''
	opts_net = NetOpts(model_string,
	                   inputspatszvalidator= lambda x : x==32,
	                   data_transforms=data_transforms,
	                   classicNet=True,
	                   weightinit=lambda x : x.normal_(0,0.05),
	                   biasinit= lambda x: x.zero_(),
	                   )

	'''Optimizer Options'''
	opts_optim =OptimOpts(lr=initlr,
	                      lr_sched_lambda= lr_sched,
	                      type='SGD',
	                      momentum=0.9,
	                      weight_decay=1e-5,
	                      dampening=0,
	                      nestrov=False,
	                      loss=NLLLoss(reduce=True))

	return opts_net,opts_optim

def nin_caffe(data_opts:DataOpts) -> Tuple[NetOpts,OptimOpts]:
	model_string = ''
	nl = 'relu'
	d = '->'
	finish = 'fin'
	model_string += 'conv|r:5,f:192,pad:same,bias:1' + d + nl + d
	model_string += 'conv|r:1,f:160,pad:same,bias:1' + d + nl + d
	model_string += 'conv|r:1,f:96,pad:same,bias:1' + d + nl + d
	model_string += 'maxpool|r:3,f:32,pad:valid,stride:2,bias:1' + d
	model_string += 'dropout|p:0.5' + d
	model_string += 'conv|r:5,f:192,pad:same,bias:1' + d + nl + d
	model_string += 'conv|r:1,f:192,pad:same,bias:1' + d + nl + d
	model_string += 'conv|r:1,f:192,pad:same,bias:1' + d + nl + d
	model_string += 'avgpool|r:3,f:32,pad:valid,stride:2,bias:1' + d
	model_string += 'dropout|p:0.5' + d
	model_string += 'conv|r:3,f:192,pad:same,bias:1' + d + nl + d
	model_string += 'conv|r:1,f:192,pad:same,bias:1' + d + nl + d
	model_string += 'conv|r:1,f:'+ str(data_opts.classnum) + ',pad:valid,bias:1' + d+ nl+d
	model_string += 'avgpool|r:7,f:32,pad:valid,stride:2,bias:1' + d + 'lnorm|s:0' +d

	model_string += finish

	'''Data OPTs'''
	data_transforms = [transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]
	'''LR SCHED'''
	lr_sched = nin_caffe_lr

	''' Net Options'''
	opts_net = NetOpts(model_string,
	                   inputspatszvalidator= lambda x : x==32,
	                   data_transforms=data_transforms,
	                   classicNet=True,
	                   weightinit=lambda x : x.normal_(0,0.05),
	                   biasinit= lambda x: x.zero_(),
	                   )
	'''Optimizer Options'''
	opts_optim =OptimOpts(lr= 2e-3,
	                      lr_sched_lambda= lr_sched,
	                      type='SGD',
	                      momentum=0.9,
	                      weight_decay=1e-4,
	                      dampening=0,
	                      nestrov=False,
	                      loss=NLLLoss(reduce=True)
	                      )

	return opts_net, opts_optim

def vgg(data_opts:DataOpts) -> Tuple[NetOpts,OptimOpts]:
	model_string = ''
	nl = 'relu'
	d = '->'
	finish = 'fin'
	model_string += 'conv|r:3,f:64,pad:same,bias:1' + d + nl + d + 'bn' + d + 'dropout|p:0.3' + d
	model_string += 'conv|r:3,f:64,pad:same,bias:1' + d + nl + d
	model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1' + d
	model_string += 'conv|r:3,f:128,pad:same,bias:1' + d + nl + d + 'bn' + d + 'dropout|p:0.4' + d
	model_string += 'conv|r:3,f:128,pad:same,bias:1' + d + nl + d
	model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1' + d
	model_string += 'conv|r:3,f:256,pad:same,bias:1' + d + nl + d + 'bn' + d + 'dropout|p:0.4' + d
	model_string += 'conv|r:3,f:256,pad:same,bias:1' + d + nl + d + 'bn' + d + 'dropout|p:0.4' + d
	model_string += 'conv|r:3,f:256,pad:same,bias:1' + d + nl + d
	model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1' + d
	model_string += 'conv|r:3,f:512,pad:same. 1 +,bias:1' + d + nl + d + 'bn' + d + 'dropout|p:0.4' + d
	model_string += 'conv|r:3,f:512,pad:same,bias:1' + d + nl + d + 'bn' + d + 'dropout|p:0.4' + d
	model_string += 'conv|r:3,f:512,pad:same,bias:1' + d + nl + d
	model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1' + d
	model_string += 'conv|r:3,f:512,pad:same,bias:1' + d + nl + d + 'bn' + d + 'dropout|p:0.4' + d
	model_string += 'conv|r:3,f:512,pad:same,bias:1' + d + nl + d + 'bn' + d +'dropout|p:0.4' + d
	model_string += 'conv|r:3,f:512,pad:same,bias:1' + d + nl + d + 'bn' + d
	model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1' + d
	model_string += 'conv|r:1,f:512,pad:same,bias:1' + d + nl + d + 'dropout|p:0.5' + d
	model_string += 'conv|r:1,f:' + str(data_opts.classnum) + ',pad:valid,bias:1' + d
	model_string += 'lnorm|s:0' + d
	model_string += finish

	'''Data OPTs'''
	data_transforms = [transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]
	'''LR SCHED'''
	lr_sched = vgg_lr

	''' Net Options'''
	opts_net = NetOpts(model_string,
	                   inputspatszvalidator=lambda x: x == 32,
	                   data_transforms=data_transforms,
	                   classicNet=True,
	                   weightinit=lambda x : x.normal_(0,0.01),
	                   biasinit=lambda x : x.zero_)
	'''Optimizer Options'''
	opts_optim =OptimOpts(lr= .001,
	                      lr_sched_lambda= lr_sched,
	                      type='SGD',
	                      momentum=0.9,
	                      weight_decay=1e-4,
	                      dampening=0,
	                      nestrov=False,
	                      loss=NLLLoss(reduce=True))
	return opts_net,opts_optim