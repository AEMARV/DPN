from src.experiment.Experiment import Experiment_
from src.models.klmodels import *
from src.datautils.datasetutils import *


class VGG_Alpha_Prior(Experiment_):
	''' Experimenting stochastic gradients in all layers:

		Model category 1: KL ReLU models with stochastic LNORM
		Model category 2: KL Sigmoid models with stochastic LNORM
	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
								 epochnum=100,
								 batchsz=128,
								 shuffledata=True,
								 numworkers=1,
								 gpu=True)
		datasets = ['cifar10','cifar100']
		for dataset in datasets:
			dataopt = DataOpts(dataset)
			for model in [self.VGG]:
				''' Alpha Regularzation'''
				for alpha in [1,2,4,8,16,32]:
					model_opt, optim_opt = model(dataopt,alpha_prior=alpha, weight_decay= 0)
					experiment_name = model.__name__ + "|l2_coef=None, alpha:{}".format(str(alpha))
					opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
								  dataopts=dataopt)
					opt_list.append(opt)

				''' L2 Regularzation'''
				base = 1e-4
				for weight_decay in [base,2*base]:
					model_opt, optim_opt = model(dataopt,alpha_prior=1,weight_decay=weight_decay)
					experiment_name = model.__name__ + "|l2_coef:{}, alpha=None".format(str(weight_decay))
					opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
								  dataopts=dataopt)
					opt_list.append(opt)

		return opt_list


	def VGG_nobn_cifar10(self,data_opts: DataOpts,
							init_coef=0.1,
	                        alpha_prior= 1,
	                        weight_decay=1e-4,
							lr=0.01,
                         val_iterations=10,
							) -> Tuple[NetOpts, OptimOpts]:

		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		convparam = 'param:log,coef:{}'.format(str(init_coef))
		convparam2 = 'param:log,coef:{}'.format(str(1*init_coef))
		convparam3 = 'param:log,coef:{}'.format(str(2 * init_coef))
		nl = 'relu'

		model_string += 'conv|r:3,f:64,pad:same,stride:1,bias:1,{}'.format(convparam3) + d + nl  + d + 'dropout|p:0.3' + d
		model_string += 'conv|r:3,f:64,pad:same,stride:1,bias:1,{}'.format(convparam3) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:128,pad:same,stride:1,bias:1,{}'.format(convparam2) + d + nl  + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:128,pad:same,stride:1,bias:1,{}'.format(convparam2) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:256,pad:same,stride:1,bias:1,{}'.format(convparam2) + d + nl  + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:256,pad:same,stride:1,bias:1,{}'.format(convparam2) + d + nl  + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:256,pad:same,stride:1,bias:1,{}'.format(convparam2) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:512,pad:same,stride:1,bias:1,{}'.format(convparam2) + d + nl  + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:512,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl  + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:512,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:512,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl  + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:512,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl  + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:512,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl  + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:1,f:512,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl + d + 'dropout|p:0.5' + d
		model_string += 'conv|r:1,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1,{}'.format(convparam) + d



		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		# data_transforms = []
		lr_sched = vgg_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=alpha_prior, val_iters= val_iterations)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim

	def VGG_nobn_cifar100(self,data_opts: DataOpts,
							init_coef=0.1,
	                        alpha_prior= 1,
	                        weight_decay=1e-4,
							lr=0.01,
	                      val_iterations= 10,
							) -> Tuple[NetOpts, OptimOpts]:
		scale=1
		ceil = lambda x: str(math.ceil(x))
		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		convparam = 'param:log,coef:{}'.format(str(1*init_coef))
		convparam2 = 'param:log,coef:{}'.format(str(1*init_coef))
		convparam3 = 'param:log,coef:{}'.format(str(1* init_coef))
		convparam4 = 'param:log,coef:{}'.format(str(1 * init_coef))
		nl = 'relu'

		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*64),convparam4) + d + nl  + d + 'dropout|p:0.3' + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*64),convparam3) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*128),convparam2) + d + nl  + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*128),convparam2) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*256),convparam2) + d + nl  + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*256),convparam2) + d + nl  + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*256),convparam2) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam2) + d + nl  + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:1,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl + d + 'dropout|p:0.5' + d
		model_string += 'conv|r:1,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1,{}'.format(convparam) + d



		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		# data_transforms = []
		lr_sched = vgg_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=alpha_prior,val_iters=val_iterations)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim

	def VGG(self,data_opts: DataOpts,
	                        alpha_prior=1,
	                        weight_decay= 1e-4,
							) -> Tuple[NetOpts, OptimOpts]:


		model_string = ''
		d = '->'
		finish = 'fin'
		convparam = 'param:log,coef:{}'.format(str(0.05))
		nl = 'relu'
		batch_norm= 'bn'

		model_string += 'conv|r:3,f:64,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl + d + batch_norm + d + 'dropout|p:0.3' + d
		model_string += 'conv|r:3,f:64,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:128,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl + d + batch_norm + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:128,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:256,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl + d + batch_norm + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:256,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl + d + batch_norm + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:256,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:512,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl + d + batch_norm + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:512,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl + d + batch_norm + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:512,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:512,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl + d + batch_norm + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:512,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl + d + batch_norm + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:512,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl + d + batch_norm + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:1,f:512,pad:same,stride:1,bias:1,{}'.format(convparam) + d + nl + d + 'dropout|p:0.5' + d
		model_string += 'conv|r:1,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1,{}'.format(convparam) + d



		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		# data_transforms = []
		lr_sched = vgg_lr

		''' Net Options'''
		divgreg= True
		netdict = dict(exact=True, divgreg=divgreg, reg_coef=0, reg_mode=None, alphaPrior=alpha_prior)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=0.1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim


class VGG_NoBatchNorm(VGG_Alpha_Prior):
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
								 epochnum=100,
								 batchsz=128,
								 shuffledata=True,
								 numworkers=1,
								 gpu=True)
		datasets = ['cifar100']
		coef = 0.01
		for dataset in datasets:
			dataopt = DataOpts(dataset)
			for model in [self.VGG_nobn_cifar100]:

				''' L2 Regularzation'''
				base = 1e-4
				for weight_decay in [base,2*base]:
					model_opt, optim_opt = model(dataopt,alpha_prior=1,weight_decay=weight_decay,init_coef=coef,val_iterations=1)
					experiment_name = model.__name__ + "|l2_coef:{}, alpha=None".format(str(weight_decay))
					opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
								  dataopts=dataopt)
					opt_list.append(opt)

				''' Alpha Regularzation'''
				for alpha in [1,2,4,8,16,32]:
					model_opt, optim_opt = model(dataopt,alpha_prior=alpha, weight_decay= 0,init_coef=coef,val_iterations=1)
					experiment_name = model.__name__ + "|l2_coef=None, alpha:{}".format(str(alpha))
					opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
								  dataopts=dataopt)
					opt_list.append(opt)


		return opt_list


class VGG_NOBN_CIFAR100(VGG_Alpha_Prior):

	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
								 epochnum=300,
								 batchsz=128,
								 shuffledata=True,
								 numworkers=1,
								 gpu=True)
		datasets = ['cifar100']
		coef = 0.03
		for dataset in datasets:
			dataopt = DataOpts(dataset)
			for model in [self.VGG_nobn_cifar100]:

				''' L2 Regularzation'''
				base = 1e-4
				for weight_decay in [base]:
					model_opt, optim_opt = model(dataopt,alpha_prior=1, weight_decay=weight_decay, init_coef=coef, val_iterations=10, lr=.1)
					experiment_name = model.__name__ + "|l2_coef:{}, alpha=None".format(str(weight_decay))
					opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
								  dataopts=dataopt)
					opt_list.append(opt)

				''' Alpha Regularzation'''
				for alpha in reversed([1, 2, 4, 8, 16]):
					model_opt, optim_opt = model(dataopt,alpha_prior=alpha, weight_decay= 0,init_coef=coef,val_iterations=10, lr=.1)
					experiment_name = model.__name__ + "|l2_coef=None, alpha:{}".format(str(alpha))
					opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
								  dataopts=dataopt)
					opt_list.append(opt)


		return opt_list


class VGG_VANILLA_CIFAR100(VGG_NOBN_CIFAR100):
	''' The VGG network trained on CIFAR100 stripped of BN and Dropout
	The network is trained with variations of the skeleton parameter and compared with L2 regularization of weights


	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
		                         epochnum=300,
		                         batchsz=128,
		                         shuffledata=True,
		                         numworkers=1,
		                         gpu=True)
		datasets = ['cifar100']
		coef = 0.03
		for dataset in datasets:
			dataopt = DataOpts(dataset)
			for model in [self.VGG_cifar100_vanilla]:
				'''L2 Regularization'''
				model_opt, optim_opt = model(dataopt, alpha_prior=1, weight_decay=1e-4, init_coef=coef,
				                             val_iterations=1, lr=.1)
				experiment_name = model.__name__ + "|l2_coef=1e-4, alpha:None"
				opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
				              dataopts=dataopt)
				opt_list.append(opt)

				''' Alpha Regularzation'''
				for alpha in reversed([32,16,8,5,4,3,2,1]):
					model_opt, optim_opt = model(dataopt, alpha_prior=alpha, weight_decay=0, init_coef=coef,
					                             val_iterations=1, lr=.1)
					experiment_name = model.__name__ + "|l2_coef=None, alpha:{}".format(str(alpha))
					opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
					              dataopts=dataopt)
					opt_list.append(opt)


		return opt_list
	def VGG_cifar100_vanilla(self,data_opts: DataOpts,
							init_coef=0.1,
	                        alpha_prior= 1,
	                        weight_decay=1e-4,
							lr=0.01,
	                      val_iterations= 10,
							) -> Tuple[NetOpts, OptimOpts]:
		scale=1
		ceil = lambda x: str(math.ceil(x))
		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		convparam = 'param:log,coef:{}'.format(str(1*init_coef))
		convparam2 = 'param:log,coef:{}'.format(str(1*init_coef))
		convparam3 = 'param:log,coef:{}'.format(str(1* init_coef))
		convparam4 = 'param:log,coef:{}'.format(str(1 * init_coef))
		nl = 'relu'

		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*64),convparam4) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*64),convparam3) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*128),convparam2) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*128),convparam2) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*256),convparam2) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*256),convparam2) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*256),convparam2) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam2) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:1,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl + d
		model_string += 'conv|r:1,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1,{}'.format(convparam) + d



		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		# data_transforms = []
		lr_sched = vgg_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=alpha_prior,val_iters=val_iterations)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim


''' Current Exp'''


class VGG_VANILLA_CIFAR100_RERUN(VGG_NOBN_CIFAR100):
	''' The VGG network trained on CIFAR100 stripped of BN and Dropout
	The network is trained with variations of the skeleton parameter and compared with L2 regularization of weights


	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
		                         epochnum=300,
		                         batchsz=128,
		                         shuffledata=True,
		                         numworkers=1,
		                         gpu=True)
		datasets = ['cifar100']
		coef = 0.03
		for dataset in datasets:
			dataopt = DataOpts(dataset)
			for model in [self.VGG_cifar100_vanilla]:
				'''L2 Regularization'''
				for reg_coef in [1/4,1/2,1,2,4]:
					'''L2 Regularization'''
					decay = reg_coef*(1e-4)
					model_opt, optim_opt = model(dataopt, alpha_prior=1, weight_decay=reg_coef*(1e-4), init_coef=coef,
					                             val_iterations=1, lr=.1)
					experiment_name = model.__name__ + "|l2_coef={}, alpha:None".format(str(decay))
					opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
					              dataopts=dataopt)
					opt_list.append(opt)


				''' Alpha Regularzation'''
				for alpha in reversed([32,16,8,5,4,3,2,1]):
					model_opt, optim_opt = model(dataopt, alpha_prior=alpha, weight_decay=0, init_coef=coef,
					                             val_iterations=1, lr=.1)
					experiment_name = model.__name__ + "|l2_coef=None, alpha:{}".format(str(alpha))
					opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
					              dataopts=dataopt)
					opt_list.append(opt)


		return opt_list
	def VGG_cifar100_vanilla(self,data_opts: DataOpts,
							init_coef=0.1,
	                        alpha_prior= 1,
	                        weight_decay=1e-4,
							lr=0.01,
	                      val_iterations= 10,
							) -> Tuple[NetOpts, OptimOpts]:
		scale=1
		ceil = lambda x: str(math.ceil(x))
		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		convparam = 'param:log,coef:{}'.format(str(1*init_coef))
		convparam2 = 'param:log,coef:{}'.format(str(1*init_coef))
		convparam3 = 'param:log,coef:{}'.format(str(1* init_coef))
		convparam4 = 'param:log,coef:{}'.format(str(1 * init_coef))
		nl = 'relu'

		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*64),convparam4) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*64),convparam3) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*128),convparam2) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*128),convparam2) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*256),convparam2) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*256),convparam2) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*256),convparam2) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam2) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:1,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl + d
		model_string += 'conv|r:1,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1,{}'.format(convparam) + d



		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		# data_transforms = []
		lr_sched = vgg_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=alpha_prior,val_iters=val_iterations)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim


class VGG_VANILLA_CIFAR10_RERUN(VGG_NOBN_CIFAR100):
	''' The VGG network trained on CIFAR100 stripped of BN and Dropout
	The network is trained with variations of the skeleton parameter and compared with L2 regularization of weights


	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
		                         epochnum=300,
		                         batchsz=128,
		                         shuffledata=True,
		                         numworkers=1,
		                         gpu=True)
		datasets = ['cifar10']
		coef = 0.03
		for dataset in datasets:
			dataopt = DataOpts(dataset)
			for model in [self.VGG_cifar10_vanilla]:
				# for reg_coef in [1/4,1/2,1,2,4]:
				# 	'''L2 Regularization'''
				# 	decay = reg_coef*(1e-4)
				# 	model_opt, optim_opt = model(dataopt, alpha_prior=1, weight_decay=reg_coef*(1e-4), init_coef=coef,
				# 	                             val_iterations=1, lr=.1)
				# 	experiment_name = model.__name__ + "|l2_coef={}, alpha:None".format(str(decay))
				# 	opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
				# 	              dataopts=dataopt)
				# 	opt_list.append(opt)

				''' Alpha Regularzation'''
				for alpha in reversed([32,16,8,5,4,3]):
					model_opt, optim_opt = model(dataopt, alpha_prior=alpha, weight_decay=0, init_coef=coef,
					                             val_iterations=1, lr=.1)
					experiment_name = model.__name__ + "|l2_coef=0, alpha:{}".format(str(alpha))
					opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
					              dataopts=dataopt)
					opt_list.append(opt)


		return opt_list
	def VGG_cifar10_vanilla(self,data_opts: DataOpts,
							init_coef=0.1,
	                        alpha_prior= 1,
	                        weight_decay=1e-4,
							lr=0.01,
	                      val_iterations= 10,
							) -> Tuple[NetOpts, OptimOpts]:
		scale=1
		ceil = lambda x: str(math.ceil(x))
		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		convparam = 'param:log,coef:{}'.format(str(1*init_coef))
		convparam2 = 'param:log,coef:{}'.format(str(1*init_coef))
		convparam3 = 'param:log,coef:{}'.format(str(1* init_coef))
		convparam4 = 'param:log,coef:{}'.format(str(1 * init_coef))
		nl = 'relu'

		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*64),convparam4) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*64),convparam3) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*128),convparam2) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*128),convparam2) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*256),convparam2) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*256),convparam2) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*256),convparam2) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam2) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:1,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl + d
		model_string += 'conv|r:1,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1,{}'.format(convparam) + d



		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		# data_transforms = []
		lr_sched = vgg_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=alpha_prior,val_iters=val_iterations)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim


class VGG_VANILLA_CIFAR10_Try(VGG_NOBN_CIFAR100):
	''' The VGG network trained on CIFAR100 stripped of BN and Dropout
	The network is trained with variations of the skeleton parameter and compared with L2 regularization of weights


	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
		                         epochnum=300,
		                         batchsz=128,
		                         shuffledata=True,
		                         numworkers=1,
		                         gpu=True)
		datasets = ['cifar10']
		coef = 0.03
		for dataset in datasets:
			dataopt = DataOpts(dataset)
			for model in [self.VGG_cifar10_vanilla]:

				''' Alpha Regularzation'''
				for alpha in reversed([32,2,1,8,16,math.log(10)]):
					model_opt, optim_opt = model(dataopt, alpha_prior=alpha, weight_decay=0, init_coef=coef,
					                             val_iterations=1, lr=.1)
					experiment_name = model.__name__ + "|l2_coef=0, alpha:{}".format(str(alpha))
					opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
					              dataopts=dataopt)
					opt_list.append(opt)


		return opt_list
	def VGG_cifar10_vanilla(self,data_opts: DataOpts,
							init_coef=0.1,
	                        alpha_prior= 1,
	                        weight_decay=1e-4,
							lr=0.01,
	                      val_iterations= 10,
							) -> Tuple[NetOpts, OptimOpts]:
		scale=1
		ceil = lambda x: str(math.ceil(x))
		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		convparam = 'param:log,coef:{}'.format(str(1*init_coef))
		convparam2 = 'param:log,coef:{}'.format(str(1*init_coef))
		convparam3 = 'param:log,coef:{}'.format(str(1* init_coef))
		convparam4 = 'param:log,coef:{}'.format(str(1 * init_coef))
		nl = 'relu'
		nlreg = 'idreg|alpha:{}'.format(str(alpha_prior/10))
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*64),convparam4) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*64),convparam3) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*128),convparam2) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*128),convparam2) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*256),convparam2) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*256),convparam2) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*256),convparam2) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam2) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d
		model_string += 'conv|r:3,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl  + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d
		model_string += 'conv|r:1,f:{},pad:same,stride:1,bias:1,{}'.format(ceil(scale*512),convparam) + d + nl + d
		model_string += 'conv|r:1,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1,{}'.format(convparam) + d



		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		# data_transforms = []
		lr_sched = vgg_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=alpha_prior,val_iters=val_iterations)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim


''' Probabilistic Models'''
class VGG_PMAP_CIFAR10_Try(VGG_NOBN_CIFAR100):
	''' The VGG network trained on CIFAR100 stripped of BN and Dropout
	The network is trained with variations of the skeleton parameter and compared with L2 regularization of weights


	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
		                         epochnum=300,
		                         batchsz=128,
		                         shuffledata=True,
		                         numworkers=1,
		                         gpu=True)
		datasets = ['cifar10']
		coef = .1
		for dataset in datasets:
			dataopt = DataOpts(dataset)
			for model in [self.VGG_cifar10_vanilla]:

				''' Alpha Regularzation'''
				for alpha in reversed([1]):
					model_opt, optim_opt = model(dataopt, alpha_prior=alpha, weight_decay=0, init_coef=coef,
					                             val_iterations=1, lr=0.1)
					experiment_name = model.__name__ + "|l2_coef=0, alpha:{}".format(str(alpha))
					opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
					              dataopts=dataopt)
					opt_list.append(opt)


		return opt_list
	def VGG_cifar10_vanilla(self,data_opts: DataOpts,
							init_coef=0.1,
	                        alpha_prior= 1,
	                        weight_decay=1e-4,
							lr=0.01,
	                      val_iterations= 10,
							) -> Tuple[NetOpts, OptimOpts]:
		scale=1
		ceil = lambda x: str(math.ceil(x))

		def mapping(f,r=3,bias=1,icnum=1,param='log',coef=1.0,pad='same',stride=1):
			sep = ","
			string = "smap|"
			string += "r:{}".format(str(r)) + sep
			string += "f:{}".format(str(f)) + sep
			string += "bias:{}".format(str(bias)) + sep
			string += "stride:{}".format(str(stride)) + sep
			string += "param:{}".format(param) + sep
			string += "icnum:{}".format(str(icnum)) + sep
			string += "coef:{}".format(str(coef)) + sep
			string += "pad:{}".format(pad)
			return string

		def mapping_xor(f,r=3,bias=1,icnum=1,param='log',coef=1.0,pad='same',stride=1):
			sep = ","
			string = "xor|"
			string += "r:{}".format(str(r)) + sep
			string += "f:{}".format(str(f)) + sep
			string += "bias:{}".format(str(bias)) + sep
			string += "stride:{}".format(str(stride)) + sep
			string += "param:{}".format(param) + sep
			string += "icnum:{}".format(str(icnum)) + sep
			string += "coef:{}".format(str(coef)) + sep
			string += "pad:{}".format(pad)
			return string

		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		convparam = 'param:log,coef:{}'.format(str(1*init_coef))
		convparam2 = 'param:log,coef:{}'.format(str(1*init_coef))
		convparam3 = 'param:log,coef:{}'.format(str(1* init_coef))
		convparam4 = 'param:log,coef:{}'.format(str(1*init_coef))
		nl = 'sample'
		# model_string += 'conv|r:1,f:3,pad:same,stride:1,bias:1,{}'.format(convparam4) + d
		model_string += 'tofin' + d# + nl + d
		model_string += mapping(scale * 2 , r=3, bias=1, icnum=24, param='log', coef=init_coef, pad='same',stride=2) + d + nl + d
		# model_string += 'klavgpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d+ nl + d

		model_string += mapping(scale * 2, r=3, bias=1, icnum=24, param='log', coef=init_coef, pad='same',stride=2) + d+ nl + d
		# model_string += 'klavgpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d+ nl + d


		model_string += mapping(scale * 2, r=3, bias=1, icnum=24, param='log', coef=init_coef, pad='same',stride=2) + d+ nl + d
		# model_string += 'klavgpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d+ nl + d


		model_string += mapping(scale * 2, r=3, bias=1, icnum=24, param='log', coef=init_coef, pad='same',stride=2) + d+ nl + d
		# model_string += 'klavgpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d+ nl + d



		model_string += mapping(scale * 2, r=3, bias=1, icnum=24, param='log', coef=init_coef, pad='same',stride=2) + d+ nl + d
		# model_string += 'klavgpool|r:2,pad:valid,stride:2,bias:1,{}'.format(convparam) + d+ nl + d


		model_string += mapping(scale * 2, r=3, bias=1, icnum=24, param='log', coef=init_coef, pad='same',stride=1) + d + nl + d

		model_string += mapping(data_opts.classnum, r=1, bias=1, icnum=1, param='log', coef=init_coef, pad='valid') + d



		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		# data_transforms = []
		lr_sched = vgg_lr

		''' Net Options'''
		netdict = dict(exact=False, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=alpha_prior,val_iters=val_iterations)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.0,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim

class BaseLines(Experiment_):
	def collect_opts(self):
		opt_list=[]
		epocheropt = EpocherOpts(self.save_results,
								 epochnum=150,
								 batchsz=100,
								 shuffledata=True,
								 numworkers=1,
								 gpu=True)
		datasets = ['cifar10','cifar100']
		models = ['quick_cifar','nin_caffe','vgg']
		for dataset in datasets:
			dataopt = DataOpts(dataset)
			for model in models:
				model_opt,optim_opt = self.get_model_opts(model,dataopt)
				opt = allOpts(model,netopts=model_opt,optimizeropts=optim_opt,epocheropts=epocheropt,dataopts=dataopt)
				opt_list.append(opt)

		return opt_list


class NIN(Experiment_):
	'''Experimenting Sigmoid vs ReLU in CNNs and FCNNs
	'''

	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
								 epochnum=150,
								 batchsz=128,
								 shuffledata=True,
								 numworkers=2,
								 gpu=True)
		datasets = ['cifar100']
		for dataset in datasets:
			dataopt = DataOpts(dataset)
			for model in [self.nin_finite]:
				for isrelu in [True]:
					model_opt, optim_opt = model(dataopt)
					opt = allOpts(model.__name__, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
								  dataopts=dataopt)
					opt_list.append(opt)
		return opt_list

	def nin_real(self, data_opts: DataOpts, isrelu=False, isnormstoch=False) -> Tuple[NetOpts, OptimOpts]:
		model_string = ''

		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
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
		model_string += 'conv|r:1,f:' + str(data_opts.classnum) + ',pad:valid,bias:1' + d + nl + d
		model_string += 'avgpool|r:7,f:32,pad:valid,stride:2,bias:1' + d + 'lnorm|s:0' + d
		model_string += finish
		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		lr_sched = nin_caffe_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=False)
		opts_net = NetOpts(model_string,
						   input_channelsize=3,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=True,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=2e-3,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=1e-4,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim

	def nin_finite(self, data_opts: DataOpts, isrelu=True, isnormstoch=False) -> Tuple[
		NetOpts, OptimOpts]:
		model_string = ''

		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'lnorm|s:{}'.format(isnormstoch)
		convparam = 'param:log,stoch:0,isrelu:{},coef:15'.format(isrelu)
		d = '->'
		finish = 'fin'
		model_string += 'klconv|r:5,f:48,icnum:4,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:80,icnum:2,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:96,icnum:1,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:5,f:48,icnum:4,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:96,icnum:2,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:192,icnum:1,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:3,f:48,icnum:4,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:96,icnum:2,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,icnum:1,f:' + str(data_opts.classnum) + ',pad:valid,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'glklavgpool|r:4,f:32,pad:valid,stride:2,bias:1' + d  # + nl + d
		model_string += finish


		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [BintoLogFSD]
		lr_sched = constant_lr(init_lr=1, step=30, exp_decay_perstep=1)

		''' Net Options'''
		netdict = dict(exact=True,divgreg=False)
		opts_net = NetOpts(model_string,
						   input_channelsize=8,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=0,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim
