from torch.nn import Module
from torch.nn import Parameter
from src.layers.klfunctions import *
from src.layers.Initializers import *
import torch.autograd.gradcheck
from torchvision.utils import save_image
import math
from definition import concentration as C
from definition import *


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


			if isinstance(m,Sampler):
				inputs = m.forward(inputs,0,None)
				if isinstance(inputs, Tuple):
					inputs = inputs[0]
				m.print_output(inputs, epoch_num, batch_num)
				# m.print_filts(epoch_num,batch_num)
			else:
				inputs = m.forward(inputs)
				m.print_output(inputs, epoch_num, batch_num)

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
			probkernel = self.weight
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

		save_image(probkernel, imagepath, normalize=False, scale_each=False, nrow=chans)

	def print_output(self, y,epoch,batch):
		probkernel = y
		probkernel= probkernel[0:,0:-1]
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

''' Deterministic Components'''

class FConv(MyModule):
	def __init__(self,*args,fnum=0,input_ch=0,spsize=0,stride=0,pad=0,init_coef= 1,isbiased= True, **kwargs):
		super(FConv,self).__init__(*args,**kwargs)
		self.fnum = fnum
		self.spsize= spsize
		self.input_ch = input_ch
		self.kernel = Parameter(torch.randn(fnum,input_ch,spsize,spsize)*init_coef)
		self.kernel.requires_grad=True
		self.register_parameter('weight',self.kernel)

		self.bias = Parameter(torch.zeros(fnum))
		self.bias.requires_grad = isbiased
		self.register_parameter('bias',self.bias)
		self.stride= stride
		self.pad = num_pad_from_symb_pad(pad,spsize)
	def print_output(self, y,epoch,batch):
		probkernel = y
		dirpath = './GenImages/' + self.compact_name + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		dirpath = dirpath + 'Output' + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		imagepath = dirpath + '/output_' + str(epoch) + '_' + str(batch) + '.bmp'

		save_image(probkernel, imagepath, normalize=True, scale_each=False, nrow=probkernel.shape[1])

		return
	def forward(self, input):
		y = F.conv2d(input,self.kernel,self.bias,stride=self.stride,padding=self.pad)
		return y

class FReLU(MyModule):
	def forward(self, inputs:Tensor):
		return inputs.relu()

class HyperLNorm(MyModule):
	def __init__(self,alpha,*args,**kwargs):
		super(HyperLNorm,self).__init__(*args,**kwargs)
		self.alpha = alpha
	def forward(self, inputs:Tensor):
		return alpha_lnorm(inputs,1,self.alpha).exp()

'''Stochastic Component'''
class KLConv_Base(MyModule):
	def __init__(self,
	             *args,
	             fnum=None,
	             icnum=1,
	             inp_icnum=None,
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
		ly = self.reshape_input_for_nxt_layer(ly)
		if prior is None:
			b, _ = self.get_log_bias()  # type:Tensor
			ly += b
		else:
			ly += prior
		lp = None
		if isinput:
			lp = None
		ly = alpha_lnorm(ly,1,10)
		return ly

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


class SMAP(MyModule):
	def __init__(self, *args,
	             out_state=None,
	             out_id_comp=None,
	             in_state=None,
	             in_id_comp=None,
	             rec_field=None,
	             stride=None,
	             pad=None,
	             init_coef=None,
	             isbiased=None,
	             **kwargs):
		super(SMAP,self).__init__(*args,**kwargs)
		self.out_state = out_state
		self.out_id_comp = out_id_comp
		self.in_state = in_state
		self.in_id_comp = in_id_comp
		self.rec_field = rec_field
		self.pad = num_pad_from_symb_pad(pad,rec_field)
		self.init_coef = init_coef
		self.isbiased = isbiased
		self.stride=stride

		self.weight = Parameter(torch.rand(out_state*out_id_comp,in_state,in_id_comp,rec_field,rec_field).log()*init_coef)
		self.mixer = Parameter(torch.rand(out_state*out_id_comp,out_state*out_id_comp,1,1).log()*init_coef)
		self.bias = Parameter(torch.zeros(out_state*out_id_comp))
		self.sign = Parameter(torch.ones(1,1,1,1,out_id_comp))

		self.weight.requires_grad = True
		self.weight.requires_grad = True
		self.mixer.requires_grad = True

		self.register_parameter('weight',self.weight)
		self.register_parameter('bias', self.bias)
		self.register_parameter('mixer', self.mixer)
	def print_output(self, y,epoch,batch):
		y = y.exp()
		probkernel = y
		probkernel= probkernel[0:,0:-1]
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
	@staticmethod
	def checkBackward(self):
		pass
		print('backward hook')
		return
	def forward(self, input):
		kernel = self.weight.log_softmax(dim=1)
		mixer = self.mixer
		mixer = mixer.log_softmax(dim=1)#.log_softmax(dim=1)
		# kernel = alpha_lnorm(self.weight,1,10)
		kernel = kernel.reshape(self.out_state*self.out_id_comp,self.in_state*self.in_id_comp,self.rec_field,self.rec_field)

		# bias = self.bias.reshape(1,self.out_state*self.out_id_comp,1,1).log_softmax(dim=1)
		# input = input - float(1/input.shape[1])

		input = input.permute([0,1,4,2,3])
		input = input.reshape(input.shape[0],-1,input.shape[3],input.shape[4])
		y = F.conv2d(input,kernel,padding=self.pad,stride=self.stride)
		# y = y + bias

		max_mixer = max_correction(mixer,1)
		max_y = max_correction(y, 1)
		mixer = mixer.unsqueeze(4)
		mixer= mixer.transpose(0,4)
		y = y.unsqueeze(4)
		y = y + mixer
		y =logsumexpstoch(y,1)
		y = y.transpose(1,4)


		y = y.reshape(y.shape[0],self.out_id_comp,self.out_state,y.shape[2],y.shape[3])
		y = y.permute([0,2,3,4,1])





		y = (y).log_softmax(dim=1)
		# y = y  -y.logsumexp(dim=1,keepdim=True).detach()

		return y




class KLAvgPool(MyModule):
	def __init__(self,spsize,stride,pad,isstoch=True, **kwargs):
		super(KLAvgPool,self).__init__(**kwargs)
		self.spsize= spsize
		self.stride = stride
		self.pad = num_pad_from_symb_pad(pad,spsize)
		self.isstoch = isstoch
	def print_output(self, y,epoch,batch):
		return
	def forward(self, x:Tensor,isinput=None,isuniform=False):

		chans = x.shape[1]
		icnum = x.shape[4]
		einput = x.exp()
		# einput = x.log_softmax(dim=1)
		einput = einput.permute(0,1,4,2,3)
		einput = einput.reshape((einput.shape[0],einput.shape[1]*einput.shape[2],einput.shape[3],einput.shape[4]))

		out = F.avg_pool2d(einput,
		                   self.spsize,
		                   stride=self.stride,
		                   padding=self.pad,
		                   count_include_pad=False)
		out = out.reshape((out.shape[0],chans,icnum,out.shape[2],out.shape[3]))
		out = out.permute(0,1,3,4,2)
		# out = out.log_softmax(dim=1)
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

class FDropOut(MyModule):
	def __init__(self,rate,*args,exact=False,**kwargs):
		super(FDropOut,self).__init__(*args,**kwargs)
		self.rate= rate
		self.exact = exact
	def forward(self, inputs,force_drop=True):
		outputs =F.dropout(inputs,p=self.rate,training=force_drop)
		return outputs


''' Inputs'''
class ToFiniteProb(MyModule):
	def __init__(self, *args, **kwargs):
		super(ToFiniteProb,self).__init__(*args,**kwargs)

	def forward(self, inputs):
		inputs = inputs.unsqueeze(4).permute([0,4,2,3,1])*100
		output = alpha_lnorm(torch.cat((inputs/2,-inputs/2),dim=1),1,1).exp()
		return output

class IDReg(MyModule):
	def __init__(self,alpha,*args,**kwargs):
		super(IDReg,self).__init__(*args,**kwargs)
		self.alpha = alpha
	def forward(self, inputs:Tensor):
		out = idreg(inputs, 1, math.log(2))
		out = out.relu()
		return out

""" Samplers"""

class Sampler(MyModule):
	def __init__(self, *args, **kwargs):
		super(Sampler,self).__init__(*args, **kwargs)
		self.axis=1
		self.prior = 'uniform'
		self.conc = Parameter(data=torch.ones(1).to(device='cuda:0'))
		self.register_parameter('concentrate',self.conc)
		self.conc.requires_grad= True

	def sample(self, lp: Tensor, axis=1, manualrand=None):
		norm = lp.logsumexp(dim=1,keepdim=True)
		lp = lp - (norm)
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
		samps = samps.detach()
		logprob = samps*lp
		logprob[logprob != logprob] = 0

		logprob = logprob.sum(dim=0,keepdim=True)
		samps = samps.transpose(0,axis)
		logprob = logprob.transpose(0,axis)
		return samps.detach(), logprob,norm

	def concentrated_sample(self, lp: Tensor, axis=1, manualrand=None, conc=1.0):
		norm = lp.logsumexp(dim=1, keepdim=True)
		lp = lp - norm
		lporig = lp
		lp = (lp*conc).log_softmax(dim=1)
		lp = lp.transpose(0,axis)
		p = lp.exp()
		cumprob = p.cumsum(dim=0)

		if manualrand is not None:
			rand = manualrand
			rand = rand.transpose(axis,0)
		else:
			rand = torch.rand_like(p[0:1,0:,0:,0:,0:])
		samps = cumprob >= rand
		samps[1:] = samps[1:] ^ samps[0:-1]
		samps = samps.type_as(p).detach()


		samps = samps.transpose(0,axis)
		samps = samps.detach()
		logprob = samps*lporig
		logprob[logprob != logprob] = 0
		logprob = logprob.sum(dim=axis,keepdim=True)


		return samps.detach(), logprob,norm
	def reject_sample(self,lp: Tensor,axis=1,conc=1.0):
		lpmgh = lp - (lp*conc).logsumexp(dim=axis,keepdim=True)/conc
		unif_sample, _, _ = self.sample(lp*0)
		lpmgh = lpmgh * unif_sample
		lpmgh[lpmgh !=lpmgh] = 0
		lpmgh= lpmgh.sum(dim=axis,keepdim=True)
		lrand = torch.rand_like(lpmgh).log()
		inmodel = (lrand< lpmgh).type_as(lpmgh)
		sample = inmodel * unif_sample
		lpmgh = inmodel * lpmgh


		return sample.type_as(lpmgh), lpmgh
	def accumulate_lprob(self,lp,usemin=False):

		if lp is None: return None
		if usemin:
			# minlp = lp.min(dim=1, keepdim=True)[0]\
			# 	.min(dim=2, keepdim=True)[0]\
			# 	.min(dim=3, keepdim=True)[0]\
			# 	.min(dim=4,keepdim=True)[0].squeeze()
			minlp = -(-lp).logsumexp(dim=(1,2,3,4),keepdim=True).squeeze()
			return minlp
		else:
			sumlp = lp.sum(dim=(1, 2, 3, 4), keepdim=True).squeeze()
			return sumlp
	def accumulate_lprob_pair(self,lp1,lp2,usemin=False):
		if lp1 is None: return lp2
		if lp2 is None: return lp1
		if usemin:
			# minlp = torch.min(lp1,lp2)
			minlp = softmin_pair(lp1,lp2)
			return minlp
		else:
			return  lp1+lp2

	def forward_v0(self, inputs,logprob,totalIntegral,manualrand=None,alpha=1.0,islast=False):
		samps, logprob_temp,norm = self.sample(inputs)
		logprob_temp = self.accumulate_lprob(logprob_temp)


		localIntegral = self.accumulate_lprob((inputs*alpha).logsumexp(dim=1,keepdim=True))
		if islast:
			Remainder= localIntegral
		else:
			# Remainder = localIntegral
			Remainder = logminusexp(localIntegral,logprob_temp*alpha)
		if totalIntegral is None:
			totalIntegral = alpha*logprob + Remainder
		else:
			totalIntegral= lsepair(totalIntegral,alpha*logprob + Remainder)

		logprob = self.accumulate_lprob_pair(logprob, logprob_temp)

		return samps,logprob,totalIntegral

	def forward_reject(self, inputs,logprob,totalIntegral,manualrand=None,alpha=1.0,islast=False):
		''' Rejection'''
		if self.training and not islast:
			samps, lpmgh = self.reject_sample(inputs,conc=alpha)
		else:
			samps, lpmgh, _ = self.sample(inputs)
		logprob_temp = self.accumulate_lprob(lpmgh)
		logprob = self.accumulate_lprob_pair(logprob, logprob_temp)

		return samps,logprob,None

	def forward(self, inputs,logprob,norm,manualrand=None,alpha=1.0,islast=False):
		''' adds the norm seperately'''
		samps, logprob_temp,_ = self.sample(inputs)
		logprob_temp = self.accumulate_lprob(logprob_temp)

		# _ , logprob_norm,_ = self.sample(inputs*alpha)

		maxnorm= self.accumulate_lprob((inputs*alpha).logsumexp(dim=1,keepdim=True)/alpha).squeeze()
		# maxnorm = self.accumulate_lprob(logprob_norm)/alpha
		norm = self.accumulate_lprob_pair(maxnorm, norm)
		logprob = self.accumulate_lprob_pair(logprob, logprob_temp)
		return samps,logprob,norm

	def forward2(self, inputs,logprob,totalIntegral,manualrand=None,alpha=1.0):
		samps, logprob_temp,norm = self.sample(inputs)
		logprob_temp = self.accumulate_lprob(logprob_temp)

		s1, conc_prob,norm = self.concentrated_sample(inputs,conc=alpha)
		# conc_prob = (inputs*alpha).logsumexp(dim=1,keepdim=True)
		localIntegral = self.accumulate_lprob(conc_prob*alpha)
		if totalIntegral is None:
			totalIntegral = localIntegral
		else:
			totalIntegral= localIntegral + totalIntegral

		logprob = self.accumulate_lprob_pair(logprob, logprob_temp)
		return samps,logprob,totalIntegral

	def max_integral(self, inputs,logprob,totalIntegral,manualrand=None,alpha=1.0):
		samps, logprob_temp,norm = self.concentrated_sample(inputs,conc=alpha)
		logprob_temp = self.accumulate_lprob(logprob_temp)

		localIntegral = self.accumulate_lprob((inputs * alpha).logsumexp(dim=1, keepdim=True))
		Remainder = logminusexp(localIntegral, logprob_temp * alpha)
		if totalIntegral is None:
			totalIntegral = (alpha*logprob) + Remainder
		else:
			totalIntegral = lsepair(totalIntegral, (alpha * logprob) + Remainder)

		logprob = self.accumulate_lprob_pair(logprob, logprob_temp)
		return samps, logprob, totalIntegral

class XOR(Sampler):
	def __init__(self, *args,
	             out_state=None,
	             out_id_comp=None,
	             in_state=None,
	             in_id_comp=None,
	             rec_field=None,
	             stride=None,
	             pad=None,
	             init_coef=None,
	             isbiased=None,
	             **kwargs):
		super(XOR,self).__init__(*args,**kwargs)
		self.out_state = out_state
		self.out_id_comp = out_id_comp
		self.in_state = in_state
		self.in_id_comp = in_id_comp
		self.rec_field = rec_field
		self.pad = num_pad_from_symb_pad(pad,rec_field)
		self.init_coef = init_coef
		self.isbiased = isbiased
		self.stride=stride

		self.weight = Parameter(torch.rand(out_id_comp,in_state,in_id_comp,rec_field,rec_field).log()*self.init_coef)
		self.bias = self.bias = Parameter(torch.zeros(out_id_comp,out_state))

		self.weight.requires_grad = True
		self.weight.requires_grad = self.isbiased

		self.register_parameter('weight',self.weight)
		self.register_parameter('bias', self.bias)
	def print_output(self, y,epoch,batch):
		probkernel = y
		probkernel= probkernel[0:,0:-1]
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

	def sample(self, lp: Tensor, axis=1, manualrand=None):
		norm = lp.logsumexp(dim=1,keepdim=True)
		lp = lp - (norm)
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
		samps = samps.detach()
		logprob = samps*lp
		logprob[logprob != logprob] = 0

		logprob = logprob.sum(dim=0,keepdim=True)
		samps = samps.transpose(0,axis)
		logprob = logprob.transpose(0,axis)
		return samps.detach(), logprob,norm
	def to_int(self,inputs:Tensor):
		return inputs.argmax(dim=1,keepdim=True)

	def forward(self, inputs, lprob_before,lprobmodel, alpha= 1):
		x_ind = inputs.argmax(dim=1,keepdim=True)
		x_ind = x_ind.permute([0,4,2,3,1])
		x_ind = x_ind.reshape(x_ind.shape[0:-1]).float()
		weight, lprob,_ = self.sample(self.weight)
		bias, lprob_bias,_ = self.sample(self.bias)
		bias = bias.argmax(dim=1,keepdim=True).reshape(1,-1,1,1).float()
		weight_ind = weight.argmax(dim=1,keepdim=True)
		weight_ind = weight_ind.reshape(weight_ind.shape[0],weight_ind.shape[2],weight_ind.shape[3],weight_ind.shape[4]).float()
		y_ind = F.conv2d(x_ind,weight_ind, padding=self.pad, stride=self.stride) + bias
		y_ind = y_ind.unsqueeze(4).transpose(1,4).long()
		y_ind = (y_ind)%self.out_state
		onehot_y = inputs.new_zeros(y_ind.shape[0],self.out_state,y_ind.shape[2],y_ind.shape[3],y_ind.shape[4])
		onehot_y.scatter_(1,y_ind,1)


		return onehot_y, lprob_bias.sum() + lprob.sum()+ lprob_before,lprobmodel



class NormalVar(Sampler):
	def __init__(self, input_channel, *args, init_logvar=1, **kwargs):
		super(NormalVar, self).__init__(*args,**kwargs)
		self.prec = Parameter(torch.ones(1,input_channel,1,1)*init_logvar)
		self.register_parameter('logvar',self.prec)
		self.prec.requires_grad=True
	def sample_normal(self,inputs,prec):
		noise = ((torch.randn_like(inputs) / ((prec).abs().sqrt()))).detach()
		return noise
	def sample_normal_nat(self, loc, scale):
		mean = loc/scale
		var = 1/scale
		output =  ((torch.randn_like(loc)* ((var).abs().sqrt()))).detach()
		return None
	def log_prob_normal(self,state,prec):
		logprob = -(((state) ** 2) * ((prec).abs()) / 2) + prec.abs().log() / 2
		return logprob
	def forward(self, inputs, concentration=1):
		inputs = inputs.detach()
		if not self.training:
			return inputs,0
		prec = self.prec.abs()
		conc_prec = prec*concentration
		noise = self.sample_normal(inputs,prec)
		noise_conc= self.sample_normal(inputs,conc_prec)

		logprob = self.log_prob_normal(noise,prec)
		logprob_conc = self.log_prob_normal(noise_conc,prec)

		output = inputs + noise
		logprob = (logprob-logprob_conc).sum(dim=(1,2,3),keepdim=True).squeeze()
		return output,logprob



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
