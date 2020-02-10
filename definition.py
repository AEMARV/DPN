import os
import torch
import sys

PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
PATH_DATA = os.path.join(PATH_ROOT,'data','datasets')
print(PATH_ROOT)

# OPTS NAMES
OPTS_DATASET = 'dataset'
OPTS_DATASET_CHANS = 'datachandim'
OPTS_DATASET_SPATIAL = 'datarcdim'
OPTS_OPTMIZER = 'optimizer'
OPTS_MODEL = 'model'
epsilon = 1e-14
concentration= 1

#Result defs
RESULT_ROOT_DIR = './results'
EXP_RESULT_ROOT_DIR = './experiment_conclude'
CODE_ROOT_DIR= './src'
def hasnan(t):
	if torch.isnan(t).sum()>0:
		print("HASSSNAN")
		return True
	return False
def hasinf(t):
	if (t == -float('inf')).sum() >0:
		print("HASSSINF")
		return True
	if (t == float('inf')).sum() >0:
		print("HASSSINF")
		return True
	return False
def boolprompt(question):
	answer=''
	while(answer.lower!='n' or answer.lower()!='y'):
		answer = input(question+' [y]/[n]')
		if answer[0].lower()=='y':
			return True
		elif answer[0].lower()=='n':
			return False
		else:
			print('please answer with y/n characters')
