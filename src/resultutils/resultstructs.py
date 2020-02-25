import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from typing import List,Dict,Tuple
np.random.seed(sum(map(ord, "aesthetics")))

class ResultStruct(object):
	def __init__(self,path:str, resultdict=None):
		if resultdict is None:
			self.resultdict = {}
		else:
			self.resultdict = resultdict
		self.writer = None
		self.trial = None
		self.path_str = path

	def set_result_dir(self):
		print('Deprecated !!!!!!!!!')
		input('Do not continue')
		# if os.path.exists(RESULT_ROOT_DIR):
		# 	print('res dir exists')
		# else:
		# 	print('Result dir not found: Creating Result directory')
		# 	self.mk_dir(RESULT_ROOT_DIR)
		# path = os.path.join(*self.path_list)
		# if not os.path.exists(path):
		# 	raise Exception('Path does not exist')
		# trial = 0
		# full_path=None
		# while (True):
		# 	full_path = os.path.join(path, str(trial))
		# 	if os.path.exists((full_path)):
		# 		trial += 1
		# 		continue
		# 	else:
		# 		os.mkdir(full_path)
		# 		break

		return None#full_path

	def add_epoch_res_dict(self,resdict:dict,epoch,write):
		for i,key in enumerate(resdict.keys()):
			if key in self.resultdict.keys():
				self.resultdict[key].append(resdict[key])
			else:
				self.resultdict[key] = [resdict[key]]
			if write:
				if self.writer is None:
					self.writer = SummaryWriter(self.path_str)
				self.writer.add_scalar(key,resdict[key],epoch)



	@staticmethod
	def write_res_dict(resdict:dict,path:str):
		writer = SummaryWriter(path)
		for key in resdict.keys():
			list_val = resdict[key]
			for epoch,val in enumerate(list_val):
				writer.add_scalar(key,val,epoch)














