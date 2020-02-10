from torch.nn import Module
from torch.nn import Parameter
from layers.klfunctions import *
from layers.Initializers import *
from layers.pmaputils import sample
import torch.autograd.gradcheck
import os
from torchvision.utils import save_image,make_grid
import math
from definition import concentration as C
from definition import *
from abc import abstractmethod



class MyModule(Module):
	def __init__(self, *args, blockidx=None, **kwargs):
		super(MyModule,self).__init__(*args, **kwargs)
		self.logprob = torch.zeros(1).to('cuda:0')
		self.regularizer = torch.zeros(1).to('cuda:0')
		self.scalar_dict = {}
		if blockidx is None:
			raise Exception('blockidx is None')
		self.blockidx = blockidx
		self.compact_name = type(self).__name__ + '({})'.format(blockidx)
		self.register_forward_hook(self.update_scalar_dict)

	def update_scalar_dict(self,self2,input,output):
		return
	def get_output_prior(self,inputprior):
		return inputprior
	def get_lrob_model(self, inputprior):
		return 0
	def get_log_prob(self):
		lprob = self.logprob
		for m in self.children():
			if isinstance(m,MyModule):
				lprob = lprob + m.get_log_prob()
		self.logprob = 0
		return lprob

	def get_reg_vals(self):
		reg = self.regularizer
		for m in self.children():
			if isinstance(m, MyModule):
				temp = m.get_reg_vals()
				if temp is not None:
					reg = reg + temp
		self.regularizer = 0
		return reg

	def print(self,inputs,epoch_num,batch_num):
		for m in self.children():
			inputs = m.forward(inputs)

			if isinstance(m, MyModule) or isinstance(m,Sampler):
				if isinstance(inputs, Tuple):
					inputs = inputs[0]
				m.print_output(inputs, epoch_num, batch_num)
				m.print_filts(epoch_num,batch_num)
			else:
				inputs= inputs[0]

	def get_scalar_dict(self):

		for m in self.children():
			if isinstance(m, MyModule):
				self.scalar_dict.update(m.get_scalar_dict())

		return self.scalar_dict

	def max_prob(self,mother,model,dim):
		lrate = (mother - model)
		max_lrate= lrate.min(dim=dim , keepdim=True)[0]
		max_logical_ind = (lrate == max_lrate).float()

		max_logical_ind = max_logical_ind / max_logical_ind.sum(dim=dim,keepdim=True)
		max_lprob = (max_logical_ind * lrate).sum(dim=dim,keepdim=True)
		return max_lprob

	def print_filts(self,epoch,batch):
		try:
			probkernel = self.get_prob_kernel()
		except:
			return
		sh = probkernel.shape
		probbias = self.get_log_bias().exp().view(sh[0], 1, 1, 1)
		probkernel = probkernel * probbias
		chans = probkernel.shape[1]
		probkernel = probkernel.view(
			[probkernel.shape[0] * probkernel.shape[1], 1, probkernel.shape[2], probkernel.shape[3]])
		dirpath = './GenImages/' + self.compact_name + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		dirpath = dirpath + 'Filters' + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		imagepath = dirpath + '/filt_' + str(epoch)+'_'+str(batch) + '.bmp'

		save_image(probkernel, imagepath, normalize=True, scale_each=False, nrow=chans)

	def print_output(self, y,epoch,batch):
		probkernel = y.exp()
		chans = probkernel.shape[1]
		factors =probkernel.shape[4]
		probkernel = probkernel.permute((0,1,4,2,3))
		probkernel = probkernel.contiguous().view(
			[probkernel.shape[0] * probkernel.shape[1]*probkernel.shape[2], 1, probkernel.shape[3], probkernel.shape[4]])
		dirpath = './GenImages/' + self.compact_name + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		dirpath = dirpath + 'Output' + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		imagepath = dirpath + '/output_' + str(epoch) + '_' + str(batch) + '.bmp'

		save_image(probkernel, imagepath, normalize=False, scale_each=False, nrow=chans*factors)

	def prop_prior(self,output_prior):
		return output_prior

	def forward(self, *inputs):
		raise Exception("Not Implemented")
		pass


class KLConv_Base(MyModule):
	def __init__(self,
	             *args,
	             fnum=None,
	             icnum=1,
	             inp_icnum,
	             kersize=None,
	             isbiased=False,
	             inp_chan_sz=0,
	             isrelu=True,
	             biasinit=None,
	             padding='same',
	             stride=1,
	             paraminit= None,
	             coefinit = 1,
	             isstoch=False,
	             **kwargs
	             ):
		super(KLConv_Base,self).__init__(*args,**kwargs)
		#TODO: Set isbinary switch in paraminit
		self.biasinit = biasinit
		self.padding = num_pad_from_symb_pad(padding,kersize)
		self.isbiased = isbiased
		self.isrelu = isrelu
		self.stride = stride
		self.isstoch= isstoch
		self.axisdim=-2
		self.icnum=icnum
		self.inp_icnum = inp_icnum
		self.spsz = kersize
		self.fnum = fnum
		self.chansz = inp_chan_sz
		self.coefinit  = coefinit
		self.kernel_shape = (fnum,self.icnum,)+(inp_chan_sz,inp_icnum,)+(kersize,kersize)
		self.paraminit = paraminit# type:Parameterizer
		self.input_is_binary= False
	'''Scalar Measurements'''
	def get_scalar_dict(self):
		#y = self.scalar_dict.copy()
		#self.scalar_dict = {}
		return self.scalar_dict

	def update_scalar_dict(self,self2,input,output):
		#Filter Entorpy
		if type(input) is tuple:
			input = input[0]

		temp_dict = {self.compact_name +'| Kernel Entropy' : self.expctd_ent_kernel().item(),
		             self.compact_name +'| Input Entropy' : self.expctd_ent_input(input).item(),
		             }
		for key in temp_dict.keys():
			if key in self.scalar_dict:
				self.scalar_dict[key] = self.scalar_dict[key]/2 + temp_dict[key]/2
			else:
				self.scalar_dict[key] = temp_dict[key]

	''' Build'''
	def build(self):
		self.paraminit.coef =  self.coefinit
		self.kernel = Parameter(data=self.paraminit(self.kernel_shape))
		self.register_parameter('weight', self.kernel)
		if self.isbiased:
			self.paraminit.coef= 0
			self.bias = Parameter(data=self.paraminit((1,self.fnum,1,1,self.icnum),isbias=True))
			self.register_parameter('bias', self.bias)

	'''Kernel/Bias Getters'''
	def get_log_kernel(self,kernel=None,index=0):
		if kernel is None:
			k = self.kernel
		sp1 = k.shape[4]
		sp2 = k.shape[5]
		k = k.reshape((self.fnum*self.icnum,self.chansz,self.inp_icnum,sp1,sp2))
		k = self.paraminit.get_log_kernel(k)
		return k
	def get_log_kernel_conv(self,kernel=None):
		k = self.get_log_kernel(kernel=kernel)
		k = k.reshape((self.fnum * self.icnum, self.inp_icnum * self.chansz, self.kernel.shape[2], self.kernel.shape[3]))
		return k

	def get_kernel(self):
		return self.kernel

	def get_prob_kernel(self)-> torch.Tensor:

		return self.get_log_kernel().exp()

	def get_log_bias(self,index=0):
		return self.paraminit.get_log_kernel(self.bias)

	def get_prob_bias(self)-> torch.Tensor:
		return self.biasinit.get_prob_bias(self.bias)

	def convwrap(self,x:Tensor,w:Parameter):
		y = F.conv2d(x, w, bias=None,
		             stride=self.stride,
		             padding=self.padding)
		return y
	def reshape_input_for_conv(self,x:Tensor):
		if x.ndimension()<5:
			return x

		x=(x.permute(0,1,4,2,3))
		x = x.reshape((x.shape[0],x.shape[1]*x.shape[2],x.shape[3],x.shape[4]))
		return x
	def reshape_input_for_nxt_layer(self,ly):
		ly = ly.view((ly.shape[0],self.fnum,self.icnum,ly.shape[2],ly.shape[3]))
		ly = ly.permute(0,1,3,4,2)
		return ly
	def add_ker_ent(self,y:torch.Tensor,x,pker,lker,mask=None):
		H = self.ent_per_spat(pker,lker)

		if mask is not None:
			H = self.convwrap((x[0:, 0:1, 0:, 0:]*0 + 1)*mask, H)
		else:
			H = self.convwrap(x[0:,0:1,0:,0:]*0 +1,H)
		hall = H.mean()
		return y + H,hall
	def project_params(self):
		self.kernel = self.paraminit.projectKernel(self.kernel)


	def get_log_prob(self):
		lprob = self.logprob
		self.logprob= 0
		return lprob
	'''Entorpy Functions'''
	def ent_per_spat(self,pker,lker):
		raise Exception(self.__class__.__name__ + ":Implement this function")
	def ent_kernel(self):
		return self.ent_per_spat(self.get_prob_kernel(),self.get_log_kernel()).sum()
	def expctd_ent_kernel(self):
		return self.ent_per_spat(self.get_log_kernel().exp(),self.get_log_kernel()).mean()/self.get_log_symbols()
	def ent_input_per_spat(self,x):
		ent = -x * x.exp()
		ent = ent.sum(dim=1, keepdim=True)
		return ent
	def expctd_ent_input(self,x):
		ent = self.ent_input_per_spat(x)
		return ent.mean()/self.get_log_symbols()
	def ent_input(self,x):
		ent = self.ent_input_per_spat(x)#type:Tensor
		e = ent.sum(dim=1,keepdim=False).sum(dim=1,keepdim=False).sum(dim=1,keepdim=False)
		return e
	def get_log_symbols(self):
		if self.input_is_binary:
			syms = self.chansz*math.log(2)
		else:
			syms = math.log(self.chansz)
		return syms


class BayesFunc(KLConv_Base):
	''' self.kernel_shape = (fnum,self.icnum,)+(inp_chan_sz,inp_icnum,)+(kersize,kersize) '''
	def __init__(self,
	             *args,
	             paraminit=None,
	             islast=False,
	             write_images=False,
	             samplingtype=2,
	             exact=False,
	             **kwargs):

		super(BayesFunc, self).__init__(*args, **kwargs)
		self.paraminit = paraminit
		self.paraminit.isbinary = False  # DO NOT Move these lines after super
		self.axisdim = 1
		self.build()
		self.exact = exact
		if exact and self.inp_icnum[2] >1:
			raise Exception("Exact Calculation of gradient is not possible with receptive filed >1")
		self.useless_counter = 0
		self.write_images = write_images
		self.samplingtype = samplingtype  # Mile Sampling is 1, Rejection Sampling is 0#
		if write_images:
			self.register_backward_hook(BayesFunc.print_grad_out)
			self.register_backward_hook(BayesFunc.print_grad_filt)
		self.islast = islast

	def build(self):
		self.inputshape=None
		self.paraminit.coef = self.coefinit
		self.kernel = Parameter(data=self.paraminit(self.kernel_shape))
		self.paraminit.coef = 1
		self.isuniform = False
		self.mixkernel = Parameter(data=self.paraminit((self.fnum*self.icnum,self.fnum*self.icnum,1,1)))
		self.requires_grad =True
		self.register_parameter('mixkernel', self.mixkernel)
		self.mixkernel.requires_grad=True
		# self.kernel = Parameter(data=torch.normal(torch.zeros(self.kernel_shape),self.coefinit))
		self.kernel.requires_grad = True
		self.register_parameter('weight', self.kernel)
		self.paraminit.coef = 0
		self.bias = Parameter(data=self.paraminit((1, self.fnum, 1, 1,self.icnum), isbias=True))
		self.bias.requires_grad = self.isbiased
		self.register_parameter('bias', self.bias)

	#### Parameter Gets

	def get_stochastic_mat(self):
		if self.inp_icnum !=1 or self.icnum !=1:
			raise Exception("Stochastic matrix is too large: Module is factorized")
		k = self.get_log_kernel_conv()[0].detach()
		b = self.get_log_bias()[0].detach()
		b = b.transpose(0,1).squeeze().unsqueeze(1)
		k = k.squeeze() + b
		k = k - k.logsumexp(dim=0)
		return k

	def get_log_kernel_conv(self,k=None):
		'''Returns a tensor of size (self.fnum * self.icnum, self.inp_icnum * self.chansz, self.kernel.shape[2], self.kernel.shape[3])'''
		k,norm = self.get_log_kernel()
		k = k.reshape((self.fnum * self.icnum, -1, k.shape[3], k.shape[4]))
		return k, norm

	def get_kernel_expanded_format(self,k):
		'''Returns a tensor of size (self.fnum, self.icnum, self.chansz, self.inp_icnum, sp_sz_1, sp_sz_2)'''
		#sp1 = k.shape[2]
		#sp2 = k.shape[3]
		k = self.get_log_kernel()
		k = k.reshape((self.fnum, self.icnum, self.chansz, self.inp_icnum, k.shape[3], k.shape[4]))
		return k

	def reshape_input_for_conv(self,x:Tensor):
		idp_dim=4
		chand_dim=1
		if x.ndimension()<5:
			return x
		x = (x.permute(0,chand_dim,idp_dim,2,3))
		x = x.reshape((x.shape[0],x.shape[1]*x.shape[2],x.shape[3],x.shape[4]))
		return x

	def reshape_input_for_nxt_layer(self,ly):
		ly = ly.view((ly.shape[0],self.fnum,self.icnum,ly.shape[2],ly.shape[3]))
		ly = ly.permute(0,1,3,4,2)
		return ly

	def reshape_input_for_prev_layer(self,ly):
		ly = ly.view((ly.shape[0],self.chansz,self.inp_icnum,ly.shape[2],ly.shape[3]))
		ly = ly.permute(0,1,3,4,2)
		return ly

	def update_scalar_dict(self, self2, input, output):
		return

	def print_filts(self,epoch,batch):
		probkernel,_ = self.get_log_kernel_conv()
		sh = probkernel.shape
		probbias = self.get_log_bias()[0].exp().view(sh[0], 1, 1, 1)
		probkernel = probkernel * probbias
		chans = probkernel.shape[1]
		probkernel = probkernel.view(
			[probkernel.shape[0] * probkernel.shape[1], 1, probkernel.shape[2], probkernel.shape[3]])
		dirpath = './GenImages/' + self.compact_name + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		dirpath = dirpath + 'Filters' + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		imagepath = dirpath + '/filt_' + str(epoch)+'_'+str(batch) + '.bmp'

		save_image(probkernel, imagepath, normalize=True, scale_each=False, nrow=chans)

	@staticmethod
	def print_grad_out(self, grad_input, grad_output):
		grad_output = -grad_output[0]
		sh = grad_output.shape
		grad_output = grad_output.view(sh[0] * sh[1], 1, sh[2], sh[3])
		dirpath = './GenImages/' + self.compact_name + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		dirpath = dirpath + 'GradOutput' + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		imagepath = dirpath + '/gradout_' + str(self.useless_counter) + '.bmp'

		save_image(grad_output, imagepath, normalize=True, scale_each=False, nrow=sh[1])

	@staticmethod
	def print_grad_filt(self, grad_input, grad_output):
		grad_output = -grad_input[1]
		sh = grad_output.shape
		grad_output = grad_output.view(sh[0] * sh[1], 1, sh[2], sh[3])
		dirpath = './GenImages/' + self.compact_name + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		dirpath = dirpath + 'GradFilt' + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		imagepath = dirpath + '/gradfilt_' + str(self.useless_counter) + '.bmp'

		save_image(grad_output, imagepath, normalize=True, scale_each=False, nrow=sh[1])

	def testGrad(self, x):
		input = torch.rand(1, 2, 2, 2).to('cuda:0')
		input = (input / input.sum(dim=1, keepdim=True)).log()

		checks = torch.autograd.gradcheck(ConvBayesMap.apply,
		                                  [input, self.get_log_kernel(), self.get_log_bias(), 100, self.padding[0],
		                                   self.stride], eps=1e-2, atol=0.1, rtol=1e-1)
		if checks:
			print("Yes")
		else:
			print("Oh no")
		return checks

	def calcent(self, y: Tensor):
		ent = -y.clamp(-10000, None) * y.exp()
		ent = ent.sum(dim=1, keepdim=True)
		ent = ent.mean()

		return ent

	def ent_filts(self, y):
		b = self.get_log_bias()
		self.get_log_kernel() + self.get_log_bias().view([])
		ent = -y.clamp(-10000, None) * y.exp()
		ent = ent.sum(dim=1, keepdim=True)
		ent = ent.mean()

	def get_model_lprob(self):
		b = self.get_log_bias()
		k = self.get_log_kernel()
		blunif = -math.log(b.shape[1])
		klunif = -math.log(k.shape[1])

		bias_divg = blunif - b
		kernel_divg = klunif - k

		bias_divg = bias_divg.min(dim=1,keepdim=True)[0].sum()
		kernel_divg = kernel_divg.min(dim=1,keepdim=True)[0].sum()

		return bias_divg+kernel_divg

	def get_log_kernel(self,kernel=None,index=0):
		'''Output shape is (self.fnum*self.icnum,self.chansz,self.inp_icnum,sp1,sp2)'''

		norm = None
		if kernel is None:
			k = self.get_kernel()
		sp1 = k.shape[4]
		sp2 = k.shape[5]
		k = k.reshape((self.fnum*self.icnum,self.chansz,self.inp_icnum,sp1,sp2))
		k, norm = self.paraminit.get_log_kernel(k)
		# k = k - ((k*10).logsumexp(dim=1,keepdim=True)/10)
		return k,norm

	def get_log_bias(self,index=0):
		norm=None
		b = self.bias
		b = b - b.logsumexp(dim=1,keepdim=True)

		return b, norm

	def get_mix_kernel(self):
		mix_kernel = self.mixkernel
		mix_kernel,_ = self.paraminit.get_log_kernel(mix_kernel)

		return mix_kernel

	def prop_prior(self,output_prior):
		''' Takes a prior for output with size (1,fnum,1,1,out_intdpt_cmpts) and return a stattionary prior on the input
		representation of size (1,in_chz,1,1,in_indpt_compts.
		The inconsisitency of the idpt components are not taken care of.
		'''
		# Output prior is of size (1, fnum, 1, 1, out_ic)
		output_prior = output_prior.permute([1,0,2,3,4])
		# Output prior is of size (fnum, 1, 1, 1, out_ic)
		output_prior = output_prior.unsqueeze(dim=5)
		# Output prior is of size (fnum, 1, 1, 1, out_ic,1)
		kernel = self.get_kernel_expanded_format(self.get_kernel())
		indptcompts = kernel.shape[1]* kernel.shape[4]*kernel.shape[5]
		# kernle is now ( fnum:0,out_ic:1,in_ch:2,in_ic:3, sp_sz_1:4, sp_sz_2:5)
		kernel = kernel.permute((0,2,4,5,1,3))
		# kernel is now of size (fnum, in_ch, sp1,sp2, out_ic ,in_ic)
		# output prior is now   (fnum, 1    , 1  , 1 , out_ic ,1    )
		mixture = output_prior + kernel
		# mixture is of size (fnum:0, in_ch:1, sp1:2,sp2:3, out_ic:4,in_ic:5)
		mixture = mixture.logsumexp(dim=0,keepdim=True).logsumexp(dim=2,keepdim=True).logsumexp(dim=3,keepdim=True).logsumexp(dim=4,keepdim=True) - math.log(indptcompts)
		mixture = mixture.squeeze(dim=4)

		# output is of sz (1,in_ch, 1,1, in_ic)
		return mixture

	def p_invert(self, y):
		y = self.reshape_input_for_conv(y)
		x = F.conv_transpose2d(y,self.get_log_kernel_conv(),stride=self.stride
		                       ,padding=self.padding[0],
		                       output_padding= 0)
		x = self.reshape_input_for_prev_layer(x)
		x = x- x.logsumexp(dim=1,keepdim=True)
		return x

	def get_lrob_model(self, inputprior):
		## Input prior dim is (1,inp_ch,1,1,inp_icnum)
		## kernel shape is (fnum,icnum,inp_ch,inp_icnum,kersize,kersize)
		## bias shape is (1,fnum, 1,1, icnum)
		k = self.get_kernel()
		k = k - k.logsumexp(dim=2,keepdim=True)
		conc = 1000
		sp_dim1= 4
		sp_dim2= 5
		in_ch_dim= 2
		in_comp_dim= 3
		out_ic_dim= 1
		out_filt_dim = 0
		if inputprior is None:
			inputprior = -math.log(k.shape[2])
		else:
			inputprior = inputprior.unsqueeze(0)
			## Input prior dim is (1,1,inp_ch,1,1,inp_icnum)
			inputprior = inputprior.transpose(5, 3)
		## Input prior dim is (1,1,inp_ch,inp_icnum,1,1)
		## kernel shape is (fnum,icnum,inp_ch,inp_icnum,kersize,kersize)
		lrob = -(-(inputprior - k)*conc).logsumexp(dim=in_ch_dim,keepdim=True)/conc
		lrob = lrob.sum(dim=(3,4,5),keepdim=True)
		## lrob shape is (fnum,icnum,1,1,1,1)
		lrob = lrob.squeeze(-1).permute((4,0,3,2,1))
		## lrob shape is (1,fnum,1,1,icnum)
		b = self.get_log_bias()[0]
		lb = - math.log(b.shape[1]) - b
		lrob = -(-(lrob + lb)*conc).logsumexp(dim=1,keepdim=True)/conc

		return lrob - math.log(2)

	def forward(self,lx,prior=None,
	            inputprior=None,
	            isuniform=False,
	            isinput=False,
	            mode=None,
	            manualrand = None,
	            concentration=None):
		self.inputshape= lx.shape
		if isinput:
			pass
			# lx= (lx.exp()>0.5).float().log()
			lx = lx*10
			lx = lx - lx.logsumexp(dim=1,keepdim=True)
		x = lx.exp()
		x = self.reshape_input_for_conv(x)
		k,_ = self.get_log_kernel_conv()
		ly = F.conv2d(x,k,stride=self.stride,padding=self.padding[0])
		useMixer= False
		if useMixer:
			mixkernel = self.get_mix_kernel()
			mixkernel = mixkernel.unsqueeze(4).transpose(0,4)
			ly= ly.unsqueeze(4)
			ly= ly + mixkernel
			# ly = ly.logsumexp(dim=1,keepdim=True)
			ly = LogSumExpStoch.apply(ly,1,0)
			ly = ly.transpose(1,4)
			ly = ly.squeeze(4)

		ly = self.reshape_input_for_nxt_layer(ly)
		if prior is None:
			b, _ = self.get_log_bias()  # type:Tensor
			ly += b
		else:
			ly += prior
		lp = None
		if isinput:
			lp = None
		ly = alpha_lnorm(ly,1,16)
		return ly, lp

	def forward_intersect(self,lx,logprob, prior=None,isuniform=False,isinput=False, mode=None, manualrand = None):
		self.inputshape= lx.shape
		model_prob_out = None
		if isinput:
			pass
			lx= (lx.exp()>0.5).float().log()
		x = lx.exp()
		x = self.reshape_input_for_conv(x)

		k,_ = self.get_log_kernel_conv()
		ly = F.conv2d(x,k,stride=self.stride,padding=self.padding[0])
		if logprob is not None:
			logprob = self.reshape_input_for_conv(logprob)
			# model_prob_out = F.conv2d(logprob,(k.exp()*0+1)[0:1,0:(logprob.shape[1]),0:,0:],stride=self.stride,padding=self.padding[0])
			# model_prob_out = -F.max_pool2d(-logprob,(k.shape[2],k.shape[3]),stride=self.stride,padding=self.padding)
			# model_prob_out = torch.min(model_prob_out,dim=1,keepdim=True)[0]
			model_prob_out = logprob
			# model_prob_out = model_prob_out.unsqueeze(4)
		useMixer= False
		if useMixer:
			mixkernel = self.get_mix_kernel()
			mixkernel = mixkernel.unsqueeze(4).transpose(0,4)
			ly= ly.unsqueeze(4)
			ly= ly + mixkernel
			# ly = ly.logsumexp(dim=1,keepdim=True)
			ly = LogSumExpStoch.apply(ly,1,0)
			ly = ly.transpose(1,4)
			ly = ly.squeeze(4)

		ly = self.reshape_input_for_nxt_layer(ly)
		if prior is None:
			b, _ = self.get_log_bias()  # type:Tensor
			ly += b
		else:
			ly += prior
		ly = (ly*C - (ly*C).logsumexp(dim=1,keepdim=True))/C
		return ly, model_prob_out


class BayesFuncJ(BayesFunc):
	def build(self):
		super(BayesFuncJ,self).build()
		self.bias_model = Parameter(data=torch.zeros(1))
		self.bias.requires_grad = True
		self.register_parameter('model_bias', self.bias_model)

	# def get_log_kernel(self,kernel=None,index=0):
	# 	return self.kernel
	# def get_log_bias(self,index=0):
	# 	return self.bias
	def get_kernel(self):
		ker = torch.cat([self.kernel,-self.kernel],dim=2)
		m = ker.clamp_min(0)
		ker = ker- m
		ker = -(m + ((-m).exp()+ ker.exp()).log())
		ker = ker - math.log(self.kernel.shape[2])
		return ker
	def get_lrob_model(self, inputprior):
		## Input prior dim is (1,inp_ch,1,1,inp_icnum)
		## kernel shape is (fnum,icnum,inp_ch,inp_icnum,kersize,kersize)
		## bias shape is (1,fnum, 1,1, icnum)
		k = self.get_kernel()
		# k = k - k.logsumexp(dim=2,keepdim=True)
		conc = 1000
		sp_dim1= 4
		sp_dim2= 5
		in_ch_dim= 2
		in_comp_dim= 3
		out_ic_dim= 1
		out_filt_dim = 0
		if inputprior is None:
			inputprior = k*0-math.log(k.shape[2])
		else:
			inputprior = inputprior.unsqueeze(0)
			## Input prior dim is (1,1,inp_ch,1,1,inp_icnum)
			inputprior = inputprior.transpose(5, 3)
		## Input prior dim is (1,1,inp_ch,inp_icnum,1,1)
		## kernel shape is (fnum,icnum,inp_ch,inp_icnum,kersize,kersize)
		lrob = ((inputprior - k)).min(dim=in_ch_dim,keepdim=True)[0]
		lrob = lrob.sum(dim=(3,4,5),keepdim=True)
		## lrob shape is (fnum,icnum,1,1,1,1)
		lrob = lrob.squeeze(-1).permute((4,0,3,2,1))
		## lrob shape is (1,fnum,1,1,icnum)
		b = self.get_log_bias()[0]
		lb = - math.log(b.shape[1]) - b
		lrob = ((lrob + lb)).min(dim=1,keepdim=True)[0]
		temp =  self.bias_model.clamp_min(0)
		logbias_model = temp  + ((-temp).exp() + (self.bias_model-temp).exp()).log()
		return lrob - logbias_model

	def get_log_kernel(self,kernel=None,index=0):
		'''Output shape is (self.fnum*self.icnum,self.chansz,self.inp_icnum,sp1,sp2)'''

		norm = None
		if kernel is None:
			k = self.get_kernel()
		sp1 = k.shape[4]
		sp2 = k.shape[5]
		k = k.reshape((self.fnum*self.icnum,k.shape[2],self.inp_icnum,sp1,sp2))
		k, norm = self.paraminit.get_log_kernel(k)
		# k = k - ((k*10).logsumexp(dim=1,keepdim=True)/10)
		return k,norm


	def get_log_kernel_ratio_conv(self, inputprior):
		''' Input prior is of size (1,ch,1,1,cmp)

		kernel original shape is (fnum 0, icnum 1, ch 2, cmp 3, kersize 4,  kersize 5)
		'''
		kernel = self.get_kernel()
		fnum,icnum,ch,cmp,kersz1,kersz2 = kernel.shape
		if inputprior is None:
			inputprior = kernel*0 - math.log(ch)
		else:
			inputprior = inputprior.unsqueeze(0)
			# inputprior shape is (1-0,1-1,ch-2,1-3,1-4,cmp-5)
			inputprior= inputprior.transpose(3,5)
		# k = self.get_kernel() - self.get_kernel().logsumexp(dim=2,keepdim=True)
		kernel = kernel - inputprior
		k = kernel.reshape((fnum*icnum,ch*cmp,kersz1,kersz2))
		return k

	def get_output_prior(self,inputprior):
		''' Returns P(Y,M)'''
		b = self.get_lrob_model(inputprior)+ self.get_log_bias()[0]

		return b

	def forward(self,x,prior=None,
	            inputprior=None,
	            isuniform=False,
	            isinput=False,
	            mode=None,
	            manualrand = None,
	            concentration=None):
		self.inputshape= x.shape
		if isinput:
			pass
			x= (x.exp()>0.5).float()
			# lx = lx*10
			# lx = lx - lx.logsumexp(dim=1,keepdim=True)
		x = self.reshape_input_for_conv(x)
		k = self.get_log_kernel_conv(self.get_kernel())[0]
		ly = F.conv2d(x,k,stride=self.stride,padding=self.padding[0])
		ly = self.reshape_input_for_nxt_layer(ly)
		ly = ly + self.get_log_bias()[0]
		ly = ly - math.log(ly.shape[1]) -  max_correction(ly,1)
		if inputprior is not None and (hasinf(inputprior)or hasnan(inputprior)):
			print("nan")
		if hasinf(k) or hasnan(k):
			print("nan")
		if hasinf(x)or hasnan(x):
			print("nan")
		if hasinf(ly)or hasnan(ly):
			print("nan")
		# returns probability of y,m|x
		if (ly> -math.log(ly.shape[1])).sum()>0:
			# print("dude what?")
			pass
		return ly, None

	def forwardv2(self,lx,prior=None,
	            inputprior=None,
	            isuniform=False,
	            isinput=False,
	            mode=None,
	            manualrand = None,
	            concentration=None):
		self.inputshape= lx.shape
		if isinput:
			pass
			lx= (lx.exp()>0.5).float().log()
			# lx = lx*10
			# lx = lx - lx.logsumexp(dim=1,keepdim=True)
		x = lx.exp()
		x = self.reshape_input_for_conv(x)
		k = self.get_log_kernel_ratio_conv(inputprior)
		ly = F.conv2d(x,k,stride=self.stride,padding=self.padding[0])
		ly = self.reshape_input_for_nxt_layer(ly)
		lm = self.get_lrob_model(inputprior)
		ly = ly + lm
		ly = ly + self.get_log_bias()[0]

		ly = alpha_lnorm(ly,1,16)
		return ly, None

class BayesFuncD(BayesFunc):
	def build(self):
		self.kernel_shape = (self.fnum*self.icnum,self.chansz*self.inp_icnum,self.spsz,self.spsz)
		super(BayesFuncD,self).build()
	def get_log_kernel_conv(self,k=None):
		return self.kernel,None
	def get_log_bias(self,index=0):
		return self.bias,None
	def get_output_prior(self,inputprior):
		return self.bias
	def forward(self,x,prior=None,
	            inputprior=None,
	            isuniform=False,
	            isinput=False,
	            mode=None,
	            manualrand = None,
	            concentration=None):
		if isinput:
			pass
			x= (x.exp()>0.5).float()
			# lx = lx*10
			# lx = lx - lx.logsumexp(dim=1,keepdim=True)

		x = self.reshape_input_for_conv(x)
		# k = torch.cat((self.kernel,-self.kernel),dim=1)
		k = self.kernel
		lratey = F.conv2d(x,k,stride=self.stride,padding=self.padding[0])
		lratey = self.reshape_input_for_nxt_layer(lratey)
		lratey = lratey + self.bias
		return lratey,None

""" Samplers"""
class Sampler(MyModule):
	def __init__(self, *args, **kwargs):
		super(Sampler,self).__init__(*args, **kwargs)
		self.axis=1
		self.prior = 'uniform'
		self.conc = Parameter(data=torch.ones(1).to(device='cuda:0'))
		self.register_parameter('concentrate',self.conc)
		self.conc.requires_grad= True

	def sample_liklihood(self,lp, axis=1, numsamples= 1,manualrand=None):
		lastaxis = lp.ndimension() - 1
		lporig = lp
		lpunif = torch.zeros_like(lp)
		lpunif = lp.exp() * 0 - (lp.exp() * 0).logsumexp(dim=1, keepdim=True)
		samplinglp = lpunif
		lpt = samplinglp.transpose(lastaxis, axis)
		M = Multinomial(total_count=numsamples, logits=lpt)
		samps = M.sample().detach()
		samps = samps.transpose(lastaxis, axis) / numsamples
		logprob = lporig*samps
		logprob[logprob != logprob] = float('Inf')
		logprob = logprob.sum(dim=axis, keepdim=True)

		lpmodel = min_correction(lpunif - lporig, axis)

		return samps.detach(), logprob

	def sample_manual(self, lp: Tensor, axis=1, manualrand=None, concentration=1):
		lnorm = lp.logsumexp(dim=axis,keepdim=True).detach()
		lp = lp-lnorm
		lp = lp.transpose(0,axis)
		p = lp.exp()
		cumprob = p.cumsum(dim=0)

		if manualrand is not None:
			rand = manualrand
			rand = rand.transpose(axis,0)
		else:
			rand = torch.rand_like(p[0:1])
		samps = cumprob >= rand
		samps[1:] = samps[1:] ^ samps[0:-1]
		samps = samps.type_as(p).detach()
		logprob = samps*lp
		logprob[logprob != logprob] = 0

		logprob = logprob.sum(dim=0,keepdim=True)
		samps = samps.transpose(0,axis)
		logprob = logprob.transpose(0,axis)
		return samps.detach(), logprob

	def sample_concentrated(self, lp: Tensor, axis=1, manualrand=None,concentration=1.0):
		# lnorm = lp.logsumexp(dim=axis,keepdim=True)
		# lp = lp-(lnorm)

		lpsamp = lp #* concentration
		# lpsamp = lpsamp - lpsamp.logsumexp(dim=axis, keepdim=True)
		lpsamp = lpsamp.transpose(0,axis)
		lp = lp.transpose(0, axis)

		p = lpsamp.exp()
		cumprob = p.cumsum(dim=0)

		if manualrand is not None:
			rand = manualrand
			rand = rand.transpose(axis,0)
		else:
			rand = torch.rand_like(p[0:1])
		samps = cumprob >= rand
		samps[1:] = samps[1:] ^ samps[0:-1]
		samps = samps.type_as(p).detach()
		logprob = samps*lp
		logprob[logprob != logprob] = 0
		logprob = logprob.sum(dim=0,keepdim=True)
		samps = samps.transpose(0,axis)
		logprob = logprob.transpose(0,axis)
		logprob = logprob
		return samps.detach(), logprob/concentration

	def sample_prob_space(self,lp: Tensor, axis=1, manualrand=None):
		lp_prior = torch.zeros_like(lp)
		lp_prior = lp_prior - lp_prior.logsumexp(dim=axis,keepdim=True)
		samp_h, lp_h = self.sample_manual(lp_prior,axis=axis,manualrand=manualrand)
		lp_model = (lp_prior - lp).min(dim=axis,keepdim=True)[0]
		# lp_model = softmin(lp_prior-lp,axis=1)
		lp_h_given_model = lp * samp_h
		lp_h_given_model[lp_h_given_model !=  lp_h_given_model] = 0
		lp_h_given_model = lp_h_given_model.sum(dim=axis,keepdim=True)
		lp_model_given_h = lp_h_given_model + lp_model - lp_h

		return samp_h, lp_model_given_h

	def sample_variational_prior(self,lp: Tensor, axis=1, manualrand=None):
		lp_prior = torch.zeros_like(lp)
		lp_prior = lp_prior - lp_prior.logsumexp(dim=axis,keepdim=True)
		samp_h, lp_h = self.sample_manual(lp,axis=axis,manualrand=manualrand)
		lp_model = (lp_prior - lp_h)*samp_h
		lp_model = lp_model.sum(dim=axis,keepdim=True)
		# lp_model = softmin(lp_prior-lp,axis=1)

		return samp_h, lp_model

	def sample_cross_unif(self, lp: Tensor, axis=1, manualrand=None):

		lp = lp.transpose(0,axis)
		p = lp.exp()
		lpunif = p * 0 - (p * 0).logsumexp(dim=0, keepdim=True)
		p= lpunif.exp()
		cumprob = p.cumsum(dim=0)

		if manualrand is not None:
			rand = manualrand
			rand = rand.transpose(axis,0)
		else:
			rand = torch.rand_like(p[0:1])
		samps = cumprob >= rand
		samps[1:] = samps[1:] ^ samps[0:-1]
		samps = samps.type_as(p).detach()
		logprob = samps*lp
		logprob[logprob != logprob] = 0
		logprob = logprob.sum(dim=0,keepdim=True)
		samps = samps.transpose(0,axis)
		logprob = logprob.transpose(0,axis)
		return samps.detach(), logprob

	def prob_model(self,lp,axis=1,manualrand=None):
		samp, logprob = self.sample_manual(lp,manualrand=manualrand)
		logprob = (logprob - lp)
		# logprob = logprob.min(dim=axis,keepdim=True)[0]
		logprob = softmin(logprob,axis)
		return samp, logprob

	def prob_model_full(self,lp,axis=1,manualrand=None):
		samp, logprob = self.sample_maximum(lp,manualrand=manualrand)
		logprob = (-logprob)
		return samp, logprob

	def p_invert(self,x,state):
		logprob = (state.exp()*x).sum(dim=1,keepdim=True)
		return state.exp(),logprob

	def forward(self, inputs,manualrand=None, mode='likelihood',logprob_accumulate=None,concentration=1.0):
		if mode == 'likelihood' or mode==0:
			samps, logprob = self.sample_concentrated(inputs, manualrand=manualrand,concentration=concentration)
		elif mode == 'concentrated_likelihood' or mode==7:
			samps, logprob = self.sample_concentrated(inputs, manualrand=manualrand,concentration=1)
		elif mode == 'sumprob' or mode == 6:
			samps, logprob = self.sample_manual(inputs, manualrand=manualrand)
			logprob= -logprob
		elif mode=='mdivg' or mode ==1:
			samps, logprob = self.prob_model_full(inputs, manualrand=manualrand)
		elif mode=='entropy' or mode ==2:
			print("Not Implemented")
		elif mode=='intersect' or mode ==2:
			samps, logprob = self.sample_manual(inputs, manualrand=manualrand)
			logprob = logprob - (inputs*C).logsumexp(dim=1,keepdim=True)/C
			logprob = logprob.sum(dim=(1,2,3,4),keepdim=True).squeeze()
			if logprob_accumulate is not None:
				logprob = logprob + logprob_accumulate
		elif mode=='cross_entropy_unif' or mode==3:
			samps, logprob = self.sample_cross_unif(inputs, manualrand=manualrand)
		elif mode == 'delta' or mode == 4:
			samps, logprob = self.sample_prob_space(inputs, manualrand=manualrand)
		elif mode == 'variational_prior' or mode == 5:
			samps, logprob = self.sample_variational_prior(inputs, manualrand=manualrand)
		else:
			raise(Exception("unavailable sampling mode \"{}\". "
			                "Available Sampling Modes:\n"
			                "mode 0: likelihood\n"
			                "mode 1: mdivg\n"
			                "mode 2: entropy\n"
			                "mode 3: cross_entropy_unif\n"
			                "mode 4: delta\n "
			                "mode 5: variational_prior".format(mode)))

		return samps.log(),logprob


class FullSampler(Sampler):
	def get_output_prior(self, inputprior=None):
		unif = 1/inputprior.shape[1]
		if (inputprior.exp()> unif).sum()>0:
			print("shit")
		pymnot = (torch.max(unif - inputprior.exp(),inputprior.new_zeros(1))).log()
		outp = torch.cat([inputprior,pymnot],dim=1)
		return outp
	def forward(self, inputs:Tensor,manualrand=None, mode='likelihood',logprob_accumulate=None,concentration=1.0):
		unif = 1/inputs.shape[1]
		inputcomp = (torch.max(unif - inputs.exp(),inputs.new_zeros(1))).log()
		if hasnan(inputcomp):
			print("negative?")
		inputs = torch.cat([inputs,inputcomp],dim=1)
		samps, lp = self.sample_manual(inputs,axis=1)
		return samps.log().detach(),lp

class RateSampler(Sampler):
	def get_output_prior(self,inputprior):
		return inputprior
	def forward(self, inputs:Tensor,manualrand=None, mode='likelihood',logprob_accumulate=None,concentration=1.0):
		if not self.training :
			return self.forward_test(inputs)
		model_lp = logsigmoid(inputs)
		model_lp = model_lp#-  max_correction(model_lp,1)
		samps, lp_sample = self.sample_manual(inputs*0)
		lp_sample = (model_lp*samps).sum(dim=1,keepdim=True)
		return (samps).float().detach(), lp_sample
	def forward_test(self,inputs):
		model_lp = logsigmoid(inputs)
		norm = model_lp.logsumexp(dim=1,keepdim=True)
		samps,lp = self.sample_manual(model_lp)
		return samps.float(), lp
	def forward2(self, inputs:Tensor,manualrand=None, mode='likelihood',logprob_accumulate=None,concentration=1.0):
		lp1 = -softplus(-inputs)
		lp2 = -softplus(inputs)
		lp = torch.cat([lp1,lp2],dim=1)
		lp = lp - math.log(inputs.shape[1])
		samps, lp_sample = self.sample_manual(lp)
		return samps.log().detach(), lp_sample
class PriorSampler(Sampler):

	def forward(self, inputs,manualrand=None, mode='likelihood',logprob_accumulate=None,concentration=1.0):
		unif_lrob = -math.log(inputs.shape[1])
		model_lrob = (unif_lrob - inputs).min(dim=1,keepdim=True)[0]
		inputs = inputs + model_lrob
		reject = ((1/float(inputs.shape[1])) - inputs.exp()+1e-15).log()
		inputs = torch.cat((inputs,reject),dim=1)
		samp,lp = self.sample_manual(inputs,axis=1)
		return samp.log(),lp


class RejectSampler(Sampler):

	def forward(self, inputs,manualrand=None, mode='likelihood',logprob_accumulate=None,concentration=1.0):
		inputs = inputs - max_correction(inputs,1)
		inputs = inputs - math.log(inputs.shape[1])
		reject = (1- inputs.logsumexp(dim=1,keepdim=True).exp()+epsilon).log()
		samp,lp = self.sample_manual(inputs,axis=1)
		rej_samps =1-samp.sum(dim=1,keepdim=True)
		lp = lp*(1-rej_samps)# + reject*(rej_samps)
		return samp.log(),lp


class KLAvgPool(MyModule):
	def __init__(self,spsize,stride,pad,isstoch=True, **kwargs):
		super(KLAvgPool,self).__init__(**kwargs)
		self.spsize= spsize
		self.stride = stride
		self.pad = num_pad_from_symb_pad(pad,spsize)
		self.isstoch = isstoch

	def forward(self, x:Tensor,isinput=None,isuniform=False):
		log_prob= torch.zeros(1).to(x.device).squeeze()
		chans = x.shape[1]
		icnum = x.shape[4]
		#einput, log_prob = sample(x, 1, 1)
		einput = x.exp()
		einput = einput.permute(0,1,4,2,3)
		einput = einput.reshape((einput.shape[0],einput.shape[1]*einput.shape[2],einput.shape[3],einput.shape[4]))

		out = F.avg_pool2d(einput,
		                   self.spsize,
		                   stride=self.stride,
		                   padding=self.pad,
		                   count_include_pad=False)
		out = out.reshape((out.shape[0],chans,icnum,out.shape[2],out.shape[3]))
		out = out.permute(0,1,3,4,2)
		out = out.clamp(epsilon,None)
		out = out.log()
		return out
	def generate(self,y:Tensor):
		#y = LogSumExpStoch.sample(y,1)
		y = y.exp()
		x = F.upsample(y,scale_factor=self.stride,mode='bilinear')
		x = x.log()
		return x
	def print_filts(self,epoch,batch):
		pass
	def prop_prior(self,output_prior):
		return output_prior


class KLAvgPoolGL(MyModule):
	def __init__(self,*args,isstoch=False,**kwargs):
		super(KLAvgPoolGL,self).__init__(*args,**kwargs)
		self.isstoch = isstoch
		self.inputshape=None
	def p_invert(self,y):
		x = y.repeat((1,1,self.inputshape[2],self.inputshape[3],self.inputshape[4]))
		return x
	def forward(self, x:Tensor,isuniform=False,isinput=None):
		self.inputshape= x.shape
		out= x.logsumexp(dim=(2,3,4),keepdim=True)- math.log(x.shape[2]*x.shape[3]*x.shape[4])
		return out
	def print_filts(self,epoch,batch):
		pass


class KLConv(KLConv_Base):
	def __init__(self,*args,
	             drop_rate=1,
	             **kwargs):
		'''      fnum=None,\n
	             kersize=None,\n
	             isbiased=False,\n
	             isrelu=True,\n
	             paraminit=None,\n
	             biasinit=None,\n
	             padding=None,\n
	             stride=1'''
		super(KLConv,self).__init__(*args,**kwargs)
		self.paraminit.isbinary = False  # DO NOT Move these lines after super
		self.axisdim= 1
		self.drop_rate = drop_rate
		self.conc_par= Parameter(data= torch.zeros(1))
		self.register_parameter('conc',self.conc_par)
		self.build()


	''' Ent Functions'''
	def ent_per_spat(self,pker,lker):
		# Entropy Per spatial Position
		H = pker*lker
		H = -H.sum(dim=1,keepdim=True)
		return H

	'''Conv Functions'''
	def kl_xl_kp_stoch(self,x:torch.Tensor):
		lkernel = self.get_log_kernel()
		sample_kernel,logprob = ExpStoch.apply(lkernel,1)
		self.logprob = logprob.sum()
		cross_y = self.convwrap(x, sample_kernel)
		kl,n = self.add_ker_ent(cross_y,x, sample_kernel, lkernel)
		return kl

	def kl_xl_kp(self,x:torch.Tensor,mask=1):
		''' Relu KL Conv '''
		lkernel = self.get_log_kernel_conv()
		pkernel = lkernel.exp()
		y = self.convwrap(x,pkernel)
		y,ent = self.add_ker_ent(y, x, pkernel, lkernel,mask=mask)
		return y,ent

	def cross_xl_kp(self,x):
		lkernel = self.get_log_kernel()
		pkernel = lkernel.exp()
		y = self.convwrap(x, pkernel)
		g, ent = self.add_ker_ent(y, x, pkernel, lkernel)
		return y, ent

	def kl_xp_kl(self,xl):
		#TODO : not implemented yet
		xp = xl.exp()
		ent = -xp * xl #type: Tensor
		ent = ent.sum(dim=1, keepdim=True)
		cross = self.convwrap(xp,self.get_log_kernel_conv())
		ent = self.convwrap(ent,self.get_log_kernel_conv().exp()[0:1,0:1,0:,0:]*0 + 1)
		y = cross + ent
		ent = ent.sum(dim=1, keepdim=False).sum(dim=1, keepdim=False).sum(dim=1, keepdim=False)
		return y,ent

	def kl_xp_kp(self,xl):
		xp = xl.exp()
		kp = self.get_log_kernel().exp()
		y = self.convwrap(xp,kp)
		return y.log()

	def kl_xp_kl_stoch(self,xl):
		''' Stoch Sigmoid KL CONV'''
		if self.training:
			xp,lprob = ExpStoch.apply(xl, 1)
			self.logprob = lprob.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
			ent = 0

		else:
			xp= xl.exp()
			ent = -xp * xl  # type: Tensor
			ent = ent.sum(dim=1, keepdim=True)
			ent = self.convwrap(ent, self.get_prob_kernel()[0:1, 0:1, 0:, 0:] * 0 + 1)

		cross = self.convwrap(xp,self.get_log_kernel())

		return cross+ ent

	def cross_xp_kl(self,xl):
		xp = xl.exp()
		ent = -xp * xl  # type: Tensor
		ent = ent.sum(dim=1, keepdim=True)
		cross = self.convwrap(xp, self.get_log_kernel())
		ent = self.convwrap(ent, self.get_prob_kernel()[0:1, 0:1, 0:, 0:] * 0 + 1)
		ent = ent.sum(dim=1,keepdim=False).sum(dim=1,keepdim=False).sum(dim=1,keepdim=False)
		return cross,ent

	def cross_xp_kl_stoch(self,xl):
		xp, lprob = ExpStoch.apply(xl,1)
		self.logprob= lprob.sum(dim=1,keepdim=True).sum(dim=2,keepdim=True).sum(dim=3,keepdim=True)
		return self.convwrap(xp,self.get_log_kernel())

	def get_mask(self,x):
		if self.drop_rate == 0:
			return 1
		else:
			if self.training:

				y = torch.rand_like(x[0:, 0:1, 0:, 0:,0:])
				y = (y < self.drop_rate).to(dtype=torch.float32)
				return y
			else:
				return 1

	def generate(self,y):
		y.detach_()
		#y = LogSumExpStoch.sample(y,1)
		y = y.exp()
		ysum = y.sum(dim=1,keepdim=True)
		y = (y+epsilon)/(ysum+epsilon)
		y = sampleprob(y,1,1)[0]
		lkernel = self.get_log_kernel()
		lkernel.detach_()
		x = F.conv_transpose2d(y,self.get_log_kernel(),stride=self.stride, padding=self.padding)
		#x=  (x-LogSumExp.apply(x,1,1)).exp()
		#x = x/(x+definition.epsilon).sum(dim=1,keepdim=True)
		return x

	def forward(self, x:torch.Tensor,MAP=False,prior=None):
		#dummy = x[0:,0:1,0:,0:,0:]*0 +1
		#dummy = F.dropout(dummy,0.5,self.training)
		#x = x * dummy
		x= self.reshape_input_for_conv(x)
		#self.project_params()
		if not self.isrelu:
			# Sigmoid Activated

			y,g = self.kl_xp_kl(x)
				#self.logprob = self.ent_kernel()
		else:
			# ReLU Activated

			y,ent = self.kl_xl_kp(x)

		y = self.reshape_input_for_nxt_layer(y)
		#y = y*math.log(self.fnum)
		#y = y *((self.conc_par.exp()).exp())
		if self.isbiased:
			pass
			y = y+  self.get_log_bias()

		#y = y + self.get_log_bias()
		return y,0,0


class KLConvB(KLConv_Base):
	def __init__(self,
	             *args,
	             **kwargs
	             ):
		super(KLConvB,self).__init__(*args,**kwargs)
		self.paraminit.isbinary = True # DO NOT Move these lines after super
		self.axisdim=4
		self.input_is_binary = True
		self.build()

	def ent_per_spat(self,pker,lker):
		# Entropy Per Spatial Position
		lker0,lker1 = self.seperate_kernels(lker)
		pker0,pker1 = self.seperate_kernels(pker)
		H = (pker0*lker0) + (pker1*lker1)
		H = -H.sum(dim=1,keepdim=True)
		return H

	def ent_input_per_spat(self,x:Tensor):
		xl = x.clamp(epsilon,None).log()
		ent =-x*xl
		ent = ent.sum(dim=1,keepdim=True)
		return ent

	def seperate_kernels(self,k):
		return k[0:,0:,0:,0:,0], k[0:,0:,0:,0:,1]

	'''KL Conv Functions'''
	def cross_xl_kp_standalone(self,x):
		k0, k1 = self.seperate_kernels(self.get_prob_kernel())
		xp1 = x
		xp0 = 1 - x
		xp1.clamp_(epsilon, 1)
		xp0.clamp_(epsilon, 1)
		xlog1 = xp1.log()
		xlog0 = xp0.log()

		y0 = self.convwrap(xlog0, k0)
		y1 = self.convwrap(xlog1, k1)
		y = y0 + y1
		return y,[]

	def cross_xl_kp(self,x:Tensor,k0:Tensor,k1:Tensor):
		xp1 = x
		xp0 = 1- x
		xp1 = xp1.clamp(epsilon,1)
		xp0 = xp0.clamp(epsilon,1)
		xlog1 = xp1.log()
		xlog0 = xp0.log()

		y0 = self.convwrap(xlog0,k0)
		y1 = self.convwrap(xlog1,k1)
		y = y0 + y1
		return y

	def cross_xp_kl(self,xp1):
		xp0 = 1-xp1
		ent = - xp0*(xp0.clamp(epsilon,None).log()) - xp1*(xp1.clamp(epsilon,None).log()) #type:Tensor
		ent = ent.sum(dim=1,keepdim=True)
		lk = self.get_log_kernel()
		lk0 , lk1  = self.seperate_kernels(lk)
		y0 = self.convwrap(xp0,lk0)
		y1 = self.convwrap(xp1,lk1)
		h = self.convwrap(ent, lk0[0:1, 0:1, 0:, 0:] * 0 + 1)
		h = h.sum(dim=1,keepdim=False).sum(dim=1,keepdim=False).sum(dim=1,keepdim=False)
		return y0 + y1,h

	def kl_xl_kp(self,x:torch.Tensor):
		'''conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor'''
		lk =self.get_log_kernel()
		pk = lk.exp()
		pk0,pk1 = self.seperate_kernels(pk)
		y = self.cross_xl_kp(x,pk0,pk1)
		y,ent = self.add_ker_ent(y,x,pk,lk)
		return y,ent

	def kl_xl_kp_stoch(self,x:torch.Tensor):
		return None

	def kl_xp_kl(self,xp1):
		xp0 = 1-xp1
		ent = - xp0*(xp0.clamp(epsilon,None).log()) - xp1*(xp1.clamp(epsilon,None).log()) #type:Tensor
		ent = ent.sum(dim=1,keepdim=True)
		lk = self.get_log_kernel()
		lk0 , lk1 = self.seperate_kernels(lk)
		y0 = self.convwrap(xp0,lk0)
		y1 = self.convwrap(xp1,lk1)
		h = self.convwrap(ent,lk0[0:1,0:1,0:,0:]*0 + 1)
		y = y0+ y1 + h
		h = h.sum(dim=1, keepdim=False).sum(dim=1, keepdim=False).sum(dim=1, keepdim=False)
		return y,h

	def kl_xp_kl_stoch(self, xp1):
		xp0 = 1 - xp1
		ent = - xp0 * (xp0.clamp(epsilon, None).log()) - xp1 * (xp1.clamp(epsilon, None).log())
		lk = self.get_log_kernel()
		lk0, lk1 = self.seperate_kernels(lk)
		y0 = self.convwrap(xp0, lk0)
		y1 = self.convwrap(xp1, lk1)
		h = self.convwrap(ent, lk0[0:1, 0:1, 0:, 0:] * 0 + 1)
		return y0 + y1 + h
	def generate(self,y):
		#y = LogSumExpStoch.sample(y, 1)
		#y= y.exp()
		#ysum = y.sum(dim=1, keepdim=True)
		#y = (y + epsilon) / (ysum + epsilon)
		#y = sampleprob(y, 1, 1)[0]
		#sampleker0,sampleker1 = self.seperate_kernels(sample(self.get_log_kernel(),4,50)[0])
		lker0,lker1 = self.seperate_kernels(self.get_log_kernel())
		x1 = F.conv_transpose2d(y,lker0.exp(),stride = self.stride, padding=self.padding)
		x0 = F.conv_transpose2d(y, lker1.exp(), stride=self.stride, padding = self.padding)
		x = x1 -x0
		#x = (x1/x0).log()
		#x0 = F.conv_transpose2d(y, lker0, stride=self.stride, output_padding=0)
		#x  = F.sigmoid(x-x0)
		return x
	def forward(self, x:torch.Tensor):
		if self.isrelu:
			y,ent = self.kl_xl_kp(x)
		else:
			y,ent = self.kl_xp_kl(x)

			# self.logprob = ent
		y = y + self.get_log_bias()

		return y


class BayesFuncI(BayesFunc):
	def __init__(self,*args,**kwargs):
		super(BayesFuncI,self).__init__(*args,**kwargs)
		self.sampler = Sampler(blockidx=-1)

	def get_log_kernel_conv(self,concentrate=1.0):
		k,norm = self.get_log_kernel()
		k, logprob= self.sampler.sample_concentrated(k,concentration=concentrate)
		k = k.reshape((self.fnum * self.icnum, self.inp_icnum * self.chansz, k.shape[3], k.shape[4]))

		return k, logprob
	def forward(self,lx,prior=None,isuniform=False,isinput=False, mode=None, manualrand = None,concentration=1.0):
		self.inputshape= lx.shape
		x = self.reshape_input_for_conv(lx)
		filtersamp,logprob = self.get_log_kernel_conv(concentrate=concentration)
		ly = F.conv2d(x,filtersamp,stride=self.stride,padding=self.padding[0])
		ly = self.reshape_input_for_nxt_layer(ly)
		if prior is None:
			b, _ = self.get_log_bias()  # type:Tensor
			ly += b
		else:
			ly += prior
		ly = ly - (ly).logsumexp(dim=1,keepdim=True)
		return ly, logprob.sum().expand(lx.shape[0],1)
""" InActive Pool of Modules"""

"""Archive ______________________________________________________________________________________________________"""
class NonFactMarkov(MyModule):
	def __init__(self,*args,fnum=-1,kersize=-1,inp_chan_sz=-1,**kwargs):
		super(NonFactMarkov,self).__init__(*args,**kwargs)
		if True:
			self.kernel = Parameter(data= (0*torch.rand(fnum,inp_chan_sz,kersize,kersize)+1).log())

		else:
			self.kernel = Parameter(data=(torch.rand(fnum, inp_chan_sz, kersize, kersize)).log()/math.log(self.kernel.shape[1]))
		self.register_parameter('kernel', self.kernel)
	def get_log_kernel(self):
		k = self.get_kernel() * math.log(self.get_kernel().shape[1])
		return k - LogSumExp.apply(k,0,1)
	def get_prob_kernel(self):
		return self.get_log_kernel().exp()
	def forward(self, x):
		x = x.exp()
		ker = self.get_prob_kernel()
		y = F.conv2d(x,ker,padding=0)
		y = y.log()
		return y


class LNorm(MyModule):
	def __init__(self,isstoch,*args,isregulated=False,**kwargs):
		super(LNorm,self).__init__(*args,**kwargs)
		self.isstoch= isstoch
		self.isregulated= isregulated
		if not(self.isstoch):
			self.implicit_layer = nn.LogSoftmax(dim=1)
			self.add_module('logsoft',self.implicit_layer)
	def forward(self, x,MAP=False):
		if self.isstoch:
			m = LogSumExpStoch.apply(x, 1,1)
			out = x - m
		else:
			#out = self.implicit_layer(x)
			out = x - x.logsumexp(dim=1,keepdim=True)
			if definition.hasnan(out):
				print("DUDE")
				pass

		return out,0,0
	def generate(self,y):
		y = (y - LogSumExp.apply(y,1,1))
		return y


class Inp2Log(MyModule):
	def __init__(self):
		super(Inp2Log,self).__init__()
	def forward(self, x:Tensor):
		x= x.clamp(epsilon,None) # type: Tensor
		x = x/x.sum(dim=1,keepdim=True)
		out = x.log()

		return out


class Mixer(KLConv_Base):
	def __init__(self,
	             *args,
	             paraminit=None,
	             **kwargs):
		'''      fnum=None,\n
	             kersize=None,\n
	             isbiased=False,\n
	             isrelu=True,\n
	             paraminit=None,\n
	             biasinit=None,\n
	             padding=None,\n
	             stride=1'''
		super(Mixer,self).__init__(*args,**kwargs)
		paraminit.coef = 10
		self.paraminit = paraminit
		self.paraminit.isbinary = False # DO NOT Move these lines after super
		self.axisdim= 1
		self.use_conv = True

		if not self.use_conv:
			self.kernel = Parameter(data=self.paraminit((1,self.kernel_shape[1],1,1,self.kernel.size()[0])))
			self.register_parameter('kernel', self.kernel)
		else:
			self.kernel = Parameter(data=self.paraminit(self.kernel_shape))
			self.register_parameter('kernel', self.kernel)
		if self.isbiased:
			self.paraminit.coef = 0
			self.biasinit = self.paraminit
			self.bias = Parameter(data=self.paraminit((1,self.kernel_shape[0],1,1)))
			self.register_parameter('bias', self.bias)

	def mix(self,x,lk:Tensor):
		ker_sz = lk.size()
		if ker_sz[2]*ker_sz[3] !=1:
			raise(Exception('Kernel size is not 1'))

		if  not  self.use_conv:
			x = x.unsqueeze(dim=4)
			if self.isstoch:
				y = LogSumExpStoch.apply(x+lk,1,1) #type:Tensor
				y = y.transpose(1,4)
				y = y.squeeze(dim=4)
			else:
				y = LogSumExp.apply(x+lk,1) # type: Tensor
				y = y.transpose(1, 4)
				y = y.squeeze(dim=4)
			return y
		else:
			y = mix.apply(x,self.get_log_kernel())
			#y = y + self.get_log_bias()
			#y = mix.normal_mix(x, self.get_log_kernel())
			#kerp = self.get_log_kernel().exp()
			#m = (x.max(dim=1,keepdim=True))[0]
			#x = x - m
			#x = x.exp()
			#y = F.conv2d(x,kerp)
			#y = (y ).log() + m
		return y

	def forward(self, x:torch.Tensor):
		y = self.mix(x,self.get_log_kernel())
		if self.isbiased:
			y = y + self.get_log_bias()
		return y



class Transitioner(KLConv_Base):
	def __init__(self,
	             *args,
	             paraminit=None,
	             **kwargs):
		'''      fnum=None,\n
	             kersize=None,\n
	             isbiased=False,\n
	             isrelu=True,\n
	             paraminit=None,\n
	             biasinit=None,\n
	             padding=None,\n
	             stride=1'''
		super(Transitioner,self).__init__(*args,**kwargs)
		self.paraminit = paraminit
		self.paraminit.isbinary = False # DO NOT Move these lines after super
		self.axisdim= 1
		self.use_conv = True
		self.kernel = Parameter(data=self.paraminit(self.kernel_shape))
		self.register_parameter('kernel', self.kernel)
		self.build()
	def update_scalar_dict(self, self2, input, output):
		return

	def forward(self, x:torch.Tensor):
		lkernel = self.kernel
		lkernel = lkernel - LogSumExp.apply(lkernel,0)
		y= self.convwrap(x.exp(),lkernel.exp()).log()
		return y


class JSDivg(Module):
	def __init__(self,*args,**kwargs):
		super(JSDivg,self).__init__()
	def forward(self, x,target):
		target = target.unsqueeze(1)
		target_one_hot = x.new_zeros([x.size()[0], x.size()[1]])  # type: Tensor
		target_one_hot.scatter_(1, target, 1)
		probs = x.exp()
		mixed = (probs + target_one_hot) / 2
		divg1 = (mixed.log() - x)  # type: Tensor
		divg1, d = divg1.min(dim=1, keepdim=True)
		divg2 = (mixed.log() - target_one_hot.log())  # type: Tensor
		divg2, d = divg2.min(dim=1, keepdim=True)
		return -(divg1 + divg2)

class MDConv(KLConv_Base):
	def __init__(self,*args,
	             **kwargs):
		'''      fnum=None,\n
	             kersize=None,\n
	             isbiased=False,\n
	             isrelu=True,\n
	             paraminit=None,\n
	             biasinit=None,\n
	             padding=None,\n
	             stride=1'''
		super(MDConv,self).__init__(*args,**kwargs)
		self.paraminit.isbinary = False  # DO NOT Move these lines after super
		self.axisdim= 1
		self.build()
	def crop_out(self,x):
		if self.padding[0] !=0 :
			return x

		ksu = math.floor((self.kernel_shape[2]-1)/2)
		ksd = math.ceil((self.kernel_shape[2]-1)/2)
		return x[0:,0:,ksd:-ksu,ksd:-ksu]

	''' Ent Functions'''
	def ent_per_spat(self,pker,lker):
		# Entropy Per spatial Position
		H = pker*lker
		H = -H.sum(dim=1,keepdim=True)
		return H

	'''Conv Functions'''
	def kl_xl_kp_stoch(self,x:torch.Tensor):
		lkernel = self.get_log_kernel()
		sample_kernel,logprob = ExpStoch.apply(lkernel,1)
		self.logprob = logprob.sum()
		cross_y = self.convwrap(x, sample_kernel)
		kl,n = self.add_ker_ent(cross_y,x, sample_kernel, lkernel)
		return kl
	def kl_xl_kp(self,x:torch.Tensor):
		''' Relu KL Conv '''
		lkernel = self.get_log_kernel()
		pkernel = lkernel.exp()
		y = self.convwrap(x,pkernel)
		y,ent = self.add_ker_ent(y, x, pkernel, lkernel)
		return y,ent
	def cross_xl_kp(self,x):
		lkernel = self.get_log_kernel()
		pkernel = lkernel.exp()
		y = self.convwrap(x, pkernel)
		g, ent = self.add_ker_ent(y, x, pkernel, lkernel)
		return y, ent

	def kl_xp_kl(self,xl):
		#TODO : not implemented yet
		xp = xl.exp()
		ent = -xp * xl #type: Tensor
		ent = ent.sum(dim=1, keepdim=True)
		cross = self.convwrap(xp,self.get_log_kernel())
		ent = self.convwrap(ent,self.get_prob_kernel()[0:1,0:1,0:,0:]*0 + 1)
		y = cross + ent
		ent = ent.sum(dim=1, keepdim=False).sum(dim=1, keepdim=False).sum(dim=1, keepdim=False)
		return y,ent

	def kl_xp_kp(self,xl):
		xp = xl.exp()
		kp = self.get_log_kernel().exp()
		y = self.convwrap(xp,kp)
		return y.log()

	def kl_xp_kl_stoch(self,xl):
		''' Stoch Sigmoid KL CONV'''
		if self.training:
			xp,lprob = ExpStoch.apply(xl, 1)
			self.logprob = lprob.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
			ent = 0

		else:
			xp= xl.exp()
			ent = -xp * xl  # type: Tensor
			ent = ent.sum(dim=1, keepdim=True)
			ent = self.convwrap(ent, self.get_prob_kernel()[0:1, 0:1, 0:, 0:] * 0 + 1)

		cross = self.convwrap(xp,self.get_log_kernel())

		return cross+ ent

	def cross_xp_kl(self,xl):
		xp = xl.exp()
		ent = -xp * xl  # type: Tensor
		ent = ent.sum(dim=1, keepdim=True)
		cross = self.convwrap(xp, self.get_log_kernel())
		ent = self.convwrap(ent, self.get_prob_kernel()[0:1, 0:1, 0:, 0:] * 0 + 1)
		ent = ent.sum(dim=1,keepdim=False).sum(dim=1,keepdim=False).sum(dim=1,keepdim=False)
		return cross,ent

	def cross_xp_kl_stoch(self,xl):
		xp, lprob = ExpStoch.apply(xl,1)
		self.logprob= lprob.sum(dim=1,keepdim=True).sum(dim=2,keepdim=True).sum(dim=3,keepdim=True)
		return self.convwrap(xp,self.get_log_kernel())

	def get_reg_vals(self):
		return None
		#TODO: Calculate the md prob of the mixture model
		b = self.get_log_bias() #type: Tensor
		if type(b) is not float:
			b = b.permute((1,0,2,3))
		k = self.get_log_kernel()
		k = k+b # type: Tensor
		meank = LogSumExp.apply(k,0)  #type: Tensor
		return meank.max(dim=1,keepdim=True)[0].sum()
	def forward(self, x:torch.Tensor):
		#self.project_params()
		if not self.isrelu:
			# Sigmoid Activated
			y = md_conv.apply(x,self.get_log_kernel())
		else:
			# ReLU Activated
			y = mdi_conv.apply(x,self.get_log_kernel())
		y = self.crop_out(y)
		y = y + self.get_log_bias()

		return y


class Classifier(KLConv_Base):
	def __init__(self,*args,
	             **kwargs):
		'''      fnum=None,\n
	             kersize=None,\n
	             isbiased=False,\n
	             isrelu=True,\n
	             paraminit=None,\n
	             biasinit=None,\n
	             padding=None,\n
	             stride=1'''
		super(Classifier, self).__init__(*args,**kwargs)
		self.paraminit.isbinary = False  # DO NOT Move these lines after super
		self.axisdim= 1
		self.build()
	def crop_out(self,x):
		if self.padding[0] !=0 :
			return x

		ksu = math.floor((self.kernel_shape[2]-1)/2)
		ksd = math.ceil((self.kernel_shape[2]-1)/2)
		return x[0:,0:,ksd:-ksu,ksd:-ksu]

	''' Ent Functions'''
	def ent_per_spat(self,pker,lker):
		# Entropy Per spatial Position
		H = pker*lker
		H = -H.sum(dim=1,keepdim=True)
		return H

	'''Conv Functions'''
	def kl_xl_kp_stoch(self,x:torch.Tensor):
		lkernel = self.get_log_kernel()
		sample_kernel,logprob = ExpStoch.apply(lkernel,1)
		self.logprob = logprob.sum()
		cross_y = self.convwrap(x, sample_kernel)
		kl,n = self.add_ker_ent(cross_y,x, sample_kernel, lkernel)
		return kl

	def kl_xl_kp(self,x:torch.Tensor):
		''' Relu KL Conv '''
		lkernel = self.get_log_kernel()
		pkernel = lkernel.exp()
		y = self.convwrap(x,pkernel)
		y,ent = self.add_ker_ent(y, x, pkernel, lkernel)
		return y,ent

	def cross_xl_kp(self,x):
		lkernel = self.get_log_kernel()
		pkernel = lkernel.exp()
		y = self.convwrap(x, pkernel)
		g, ent = self.add_ker_ent(y, x, pkernel, lkernel)
		return y, ent

	def kl_xp_kl(self,xl):
		#TODO : not implemented yet
		xp = xl.exp()
		ent = -xp * xl #type: Tensor
		ent = ent.sum(dim=1, keepdim=True)
		cross = self.convwrap(xp,self.get_log_kernel())
		ent = self.convwrap(ent,self.get_prob_kernel()[0:1,0:1,0:,0:]*0 + 1)
		y = cross + ent
		ent = ent.sum(dim=1, keepdim=False).sum(dim=1, keepdim=False).sum(dim=1, keepdim=False)
		return y,ent

	def kl_xp_kp(self,xl):
		xp = xl.exp()
		kp = self.get_log_kernel().exp()
		y = self.convwrap(xp,kp)
		return y.log()

	def kl_xp_kl_stoch(self,xl):
		''' Stoch Sigmoid KL CONV'''
		if self.training:
			xp,lprob = ExpStoch.apply(xl, 1)
			self.logprob = lprob.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
			ent = 0

		else:
			xp= xl.exp()
			ent = -xp * xl  # type: Tensor
			ent = ent.sum(dim=1, keepdim=True)
			ent = self.convwrap(ent, self.get_prob_kernel()[0:1, 0:1, 0:, 0:] * 0 + 1)

		cross = self.convwrap(xp,self.get_log_kernel())

		return cross+ ent

	def cross_xp_kl(self,xl):
		xp = xl.exp()
		ent = -xp * xl  # type: Tensor
		ent = ent.sum(dim=1, keepdim=True)
		cross = self.convwrap(xp, self.get_log_kernel())
		ent = self.convwrap(ent, self.get_prob_kernel()[0:1, 0:1, 0:, 0:] * 0 + 1)
		ent = ent.sum(dim=1,keepdim=False).sum(dim=1,keepdim=False).sum(dim=1,keepdim=False)
		return cross,ent

	def cross_xp_kl_stoch(self,xl):
		xp, lprob = ExpStoch.apply(xl,1)
		self.logprob= lprob.sum(dim=1,keepdim=True).sum(dim=2,keepdim=True).sum(dim=3,keepdim=True)
		return self.convwrap(xp,self.get_log_kernel())

	def get_reg_vals(self):
		return None

	def forward(self, x:torch.Tensor):
		xsampled = ExpStoch.apply(x,1)
		if not self.isrelu:
			# Sigmoid Activated
			y = md_conv.apply(x,self.get_log_kernel())
		else:
			# ReLU Activated
			y = mdi_conv.apply(x,self.get_log_kernel())
		y = self.crop_out(y)
		y = y + self.get_log_bias()

		return y

class JConv(KLConv_Base):
	''' Implements the convolutional probabilistic matching of input-filter
		JConv calculates the probability that a single sample generated from the filter matches
		a single sample generated from the input distribution and outputs the log probabiltiy.
	'''
	def __init__(self,
	             *args,
	             paraminit=None,
	             **kwargs):
		'''      fnum=None,\n
	             kersize=None,\n
	             isbiased=False,\n
	             isrelu=True,\n
	             paraminit=None,\n
	             biasinit=None,\n
	             padding=None,\n
	             stride=1'''
		super(JConv,self).__init__(**kwargs)
		self.paraminit = paraminit
		self.coefinit = 1
		self.paraminit.isbinary = False  # DO NOT Move these lines after super
		self.axisdim= 1
		self.build()

	def crop_out(self,x):
		ksu = math.floor((self.kernel_shape[2]-1)/2)
		ksd = math.ceil((self.kernel_shape[2]-1)/2)
		return x[0:,0:,ksd:-ksu,ksd:-ksu]

	def forward(self, x:torch.Tensor):
		y = jointConv.apply(x,self.get_log_kernel())
		if self.padding[0]==0:
			y = self.crop_out(y)
		#y = self.convwrap(x.exp(),self.get_log_kernel().exp())
		return  y.log()# +self.get_log_bias()


class Log(MyModule):
	def __init__(self, **kwargs):
		super(Log, self).__init__(**kwargs)
	def forward(self, x:torch.Tensor):
		y = x.clamp(epsilon,None).log()
		return y


class L2LogProb(MyModule):
	def __init__(self, **kwargs):
		super(L2LogProb, self).__init__(**kwargs)
	def forward(self, x:torch.Tensor):
		y = x / ((x+epsilon).sum(dim=1, keepdim=True)) #type: Tensor
		y = y.clamp(epsilon,None)
		y = y.log()
		return y
	def generate(self,y):
		return y


class SpConv(KLConv_Base):
	''' When the Nonlinearity is the L2-Normalization, SPConv calculates the cosine similarity.'''
	def __init__(self,
	             *args,
	             paraminit=None,
	             **kwargs):
		'''      fnum=None,\n
	             kersize=None,\n
	             isbiased=False,\n
	             isrelu=True,\n
	             paraminit=None,\n
	             biasinit=None,\n
	             padding=None,\n
	             stride=1'''

		super(SpConv,self).__init__(**kwargs)
		self.paraminit = NormalParameter()
		self.isbiased = True
		self.build()

	def build(self):
		self.paraminit.coef = self.coefinit
		self.kernel = Parameter(data=self.paraminit(self.kernel_shape,1))
		self.register_parameter('kernel', self.kernel)
		if self.isbiased:
			self.bias = Parameter(data=self.paraminit((1,self.kernel_shape[0],1,1),0.0001))
			self.register_parameter('bias', self.bias)

	def forward(self, x:torch.Tensor):
		kernel = self.paraminit.get_kernel(self.get_kernel())
		y = self.convwrap(x,kernel)
		y = y  + self.paraminit.get_kernel(self.bias)
		return y


class L2Norm(MyModule):
	def __init__(self, **kwargs):
		super(L2Norm, self).__init__(**kwargs)
	def forward(self, x:torch.Tensor):
		y = x / ((x**2).sum(dim=1, keepdim=True)).sqrt()
		return y


def num_pad_from_symb_pad(pad:str,ksize:int)->Tuple[int]:
	if pad=='same':
		rsize = ksize
		csize = ksize
		padr = (rsize-1)/2
		padc = (csize-1)/2
		return (int(padr),int(padc))
	elif pad=='valid':
		padr=0
		padc=0
		return (int(padr),int(padc))
	elif type(pad) is tuple:
		return pad
	else:
		raise(Exception('Padding is unknown--Pad:',pad))
