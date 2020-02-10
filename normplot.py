import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from src.netparsers.staticnets import StaticNet
import math
import os
def get_model_string():
	init_coef = 0.03
	scale = 1
	ceil = lambda x: str(math.ceil(x))
	''' With no regularization diverges'''
	model_string = ''
	d = '->'
	finish = 'fin'
	convparam = 'param:log,coef:{}'.format(str(1 * init_coef))
	convparam2 = 'param:log,coef:{}'.format(str(1 * init_coef))
	convparam3 = 'param:log,coef:{}'.format(str(1 * init_coef))
	convparam4 = 'param:log,coef:{}'.format(str(1 * init_coef))
	nl = 'relu'

	model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale * 64), convparam4) + d + nl + d
	model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale * 64), convparam3) + d + nl + d
	model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
	model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale * 128), convparam2) + d + nl + d
	model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale * 128), convparam2) + d + nl + d
	model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
	model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale * 256), convparam2) + d + nl + d
	model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale * 256), convparam2) + d + nl + d
	model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale * 256), convparam2) + d + nl + d
	model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
	model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale * 512), convparam2) + d + nl + d
	model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale * 512), convparam) + d + nl + d
	model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale * 512), convparam) + d + nl + d
	model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
	model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale * 512), convparam) + d + nl + d
	model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale * 512), convparam) + d + nl + d
	model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale * 512), convparam) + d + nl + d
	model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
	model_string += 'conv|r:1,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale * 512), convparam) + d + nl + d
	model_string += 'conv|r:1,f:' + str(100) + ',pad:valid,stride:1,bias:1,{}'.format(convparam) + d
	model_string += finish


def load_model(path):
	state_dict= torch.load(path)
	model =StaticNet(get_model_string(),3)
	model.load_state_dict(state_dict)
	return model
def norm_dict(state_dict):
	ret_list = []
	for key in state_dict:
		if(key.count('weight')>0):
			tensor = state_dict[key]
			norm = (tensor**2).sum().mean().item()
			ret_list = ret_list + [norm]
	return ret_list
join = os.path.join
from typing import *
def stringcomp(a:str):
	lst =a.split(":")
	alpha = lst[-1]
	if(alpha == "None"):
		return -1
	else:
		return int(alpha)


def plot_dict_norm_list(dict_norm_list):
	fix, axs = plt.subplots()
	keys= (list(dict_norm_list.keys()))
	keys.sort(key=stringcomp)
	for model in keys:
		norms = dict_norm_list[model]
		alpha = stringcomp(model)/32
		axs.plot(list(range(len(norms))),norms,label=model)
	fix.show()
	axs.legend()

if __name__ == '__main__':
	rslt_folder = './Results/VGG_VANILLA_CIFAR10_Try/cifar10'
	dict_norm_list = dict()
	for model_folder in os.listdir(rslt_folder):
		model_path = join(rslt_folder,model_folder)
		model_path = join(model_path,'0')
		state_dict_path = join(model_path,'final_model.model')
		state_dict = torch.load(state_dict_path)
		norm_list = norm_dict(state_dict['model_state'])
		dict_norm_list.update([(model_folder,norm_list)])

	plot_dict_norm_list(dict_norm_list)
	input()



