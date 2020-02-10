from src.netparsers.staticnets import StaticNet
from src.optstructs import allOpts
from src.resultutils.resultstructs import *
from torch.optim.lr_scheduler import LambdaLR
from src.layers.pmaputils import *
from torch.utils.data.dataloader import DataLoader
from src.layers.klmodules import *
from src.layers.klfunctions import *
import definition
from torch.optim import *
import random


class Epocher(object):
	def __init__(self,opts:allOpts):

		super(Epocher, self).__init__()
		self.opts = opts
		self.trainloader, self.testloader = self.create_data_set(opts) # type:DataLoader
		self.model = self.create_model_module(opts) #type:MyModule
		self.optimizer = self.create_optimizer(opts,self.model)
		self.results=None
		self.path=None

	def create_data_set(self, opts):
		train_loader, test_loader = self.opts.dataopts.get_loaders(self.opts)

		return train_loader, test_loader

	def reinstantiate_model(self):
		self.model = self.create_model_module(self.opts)  # type:MyModule
		self.optimizer = self.create_optimizer(self.opts, self.model)
		self.results = None

	def create_model_module(self,opts) -> StaticNet:
		#  TODO: Static Net is replaced by Composite Net
		module = StaticNet(opts.netopts.modelstring,
		                   opts.netopts.input_channelsize,
		                   )
		return module

	def create_optimizer(self,opts, model: Module):
		optimopts = opts.optimizeropts
		if opts.epocheropts.gpu:
			device = torch.device("cpu")
			model = model.to(device=device)

		optim = globals()[optimopts.type](model.parameters(),
		                                  lr=optimopts.lr,
		                                  momentum=optimopts.momentum,
		                                  weight_decay=optimopts.weight_decay,
		                                  dampening=optimopts.dampening,
		                                  nesterov=optimopts.nestrov)
		opts.optimizeropts.lr_sched = LambdaLR(optim, opts.optimizeropts.lr_sched_lambda, last_epoch=-1)
		return optim

	def order_batch_by_label(self,batch,labels):
		_,indices = labels.sort(dim=0)
		batch = batch[indices,:,:,:]
		return batch

	def logical_index(self,batch:Tensor,booleanind):
		if booleanind.all() and booleanind.numel()==1:
			return batch
		int_index= booleanind.nonzero().squeeze()
		r = batch.index_select(dim=0,index=int_index)
		return r

	def label_to_onehot(self,output:Tensor, label):
		onehot = output.new_zeros(output.size())
		label= label.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
		label = label.transpose(0,1)
		onehot.scatter_(1,label,1)
		return onehot.float()

	def block_grad(self,paramlist:List,ind=None):
		length = paramlist.__len__()
		totalnorm = 0
		if ind is None:
			ind = random.randint(0, length-1)
		maxnorm= -1
		maxind = -1
		for i in range(length):
			if paramlist[i].grad is None: continue
			thisnorm = (paramlist[i].grad**2).sum()
			totalnorm +=thisnorm
			if thisnorm> maxnorm:
				maxnorm = thisnorm
				# if maxind!=-1 :paramlist[maxind].grad= None
				maxind = i
		return totalnorm.sqrt()

	def rnd_block_grad(self,paramlist:List,ind=None):
		length = paramlist.__len__()
		totalnorm = 0
		if ind is None:
			ind = random.randint(0, length-1)
		for i in range(length):
			if paramlist[i].grad is None:
				ind= ind-1
				continue
			totalnorm = totalnorm + paramlist[i].grad.sum()
			if i!= ind :paramlist[i].grad= None
		return

	def normalize_grad(self,paramlist:List):
		length = paramlist.__len__()

		for i in range(length):
			g = paramlist[i].grad
			paramlist[i].grad = paramlist[i].grad/ float((math.log(g.shape[1])))

	def deltac_optimization(self,inputs,labels,usemin=False,concentration=1.0):
		batchsz = inputs.shape[0]
		iternum= 0
		total_samples= 0
		total_corrects= 0
		#TODO
		avg_iter =0
		sampler = Sampler(blockidx=-1)
		while True:
			avg_iter= avg_iter+ float(inputs.shape[0])
			lp_hgm = None
			iternum += 1
			output_model, lprob,maxlprob = self.model(inputs,usemin=usemin,concentration=concentration)

			output, lprob,maxlprob = sampler(output_model,lprob,maxlprob, alpha=concentration,islast=True)
			output_model = output_model.log_softmax(dim=1)
			#maxlprob = softmax_pair(lprob*concentration, maxlprob)
			loss = self.opts.optimizeropts.loss(output.log().view(inputs.shape[0], -1), labels)
			if iternum>1000:
				print(loss.squeeze())
				print(output_model)
				print(maxlprob)
				print(lprob)
				input("Press any key")
			is_oracle = (-loss).exp() > torch.rand_like(loss)
			to_be_summed = (is_oracle).squeeze().detach()

			if iternum == 1:
				ret_output= output_model.log_softmax(dim=1).reshape(output_model.shape[0],-1).detach()
				ret_likelihood = self.opts.optimizeropts.loss(ret_output.view(-1, self.opts.dataopts.classnum), labels).mean().detach()
				ret_entropy = -lprob.mean().detach()
				total_samples += float(inputs.shape[0])
				total_corrects += to_be_summed.sum().float()

			lpmgh = lprob# - (maxlprob)
			loss = ((-lpmgh.squeeze()*(2*to_be_summed.type_as(lpmgh)-1)).sum()/ batchsz) # The issue was with size of lprob not squeezed.
			definition.hasnan(loss)
			definition.hasinf(loss)
			loss.backward()
			# self.optimizer.step()
			# self.optimizer.zero_grad()
			break
			if(to_be_summed.all()):
				break

			# break
			inputs = self.logical_index(inputs, to_be_summed ^ 1)
			labels = self.logical_index(labels, to_be_summed ^ 1)

		avg_iter = avg_iter/batchsz
		stats= dict(avg_iterations=avg_iter)
		# print(stats['jsd'],end='')
		return ret_likelihood,ret_output, total_corrects/total_samples, ret_entropy,stats


	def delta_optimization(self,inputs,labels):
		batchsz = inputs.shape[0]
		iternum= 0
		lprob_currect = 0
		mode='delta'
		while True:
			iternum += 1
			output, lp_mgh, _ = self.model(inputs, mode=mode)
			# TODO min
			outputlabelsample = output.max(dim=1, keepdim=True)[1]

			# calc accept/reject masks
			is_oracle = (outputlabelsample.squeeze() == labels).float()
			is_model =  (lp_mgh.exp() > torch.rand_like(lp_mgh)).float()
			lp_ogh = is_oracle.log()
			lp_delta_gh = ((lp_ogh.exp()+lp_mgh.exp()-2*lp_ogh.exp()*lp_mgh.exp())+definition.epsilon).log()
			lp_deltac_gh = (1-lp_ogh.exp() - lp_mgh.exp()*(1-2*lp_ogh.exp())+definition.epsilon).log()
			isdelta = lp_delta_gh.exp() > torch.rand_like(lp_deltac_gh)
			isdeltac = isdelta^1
			to_be_summed = (isdelta).squeeze().detach()

			if iternum == 1:
				ret_ldeltac = -lp_delta_gh.mean().detach()

			#lprob_currect += (((-lp_mgh.squeeze() * (to_be_summed.float())) / batchsz).sum()).detach()
			lossdc = ((lp_delta_gh[to_be_summed].float()).sum())/batchsz #+ (lp_delta_gh[to_be_summed^1].sum())/batchsz
			loss =   lossdc.sum()
			definition.hasnan(loss)
			definition.hasinf(loss)
			if to_be_summed.float().sum() >0:# 0:
				pass
				loss.sum().backward()
				break
			if to_be_summed.all():
				break
			inputs = self.logical_index(inputs, to_be_summed ^ 1)
			labels = self.logical_index(labels, to_be_summed ^ 1)
		return ret_ldeltac, ret_ldeltac, ret_ldeltac, ret_ldeltac

	def likelihood_optimization(self,inputs,labels,usemin=False,concentration=1.0):
		batchsz = inputs.shape[0]
		iternum= 0
		total_samples= 0
		total_corrects= 0
		#TODO
		avg_iter =0
		sampler = Sampler(blockidx=-1)
		while True:
			avg_iter= avg_iter+ float(inputs.shape[0])
			lp_hgm = None
			iternum += 1
			output_model, lprob,maxlprob = self.model(inputs,usemin=usemin,concentration=concentration)

			output, lprob,maxlprob = sampler(output_model,lprob,maxlprob, alpha=concentration,islast=True)
			output_model = output_model.log_softmax(dim=1)
			#maxlprob = softmax_pair(lprob*concentration, maxlprob)
			loss = self.opts.optimizeropts.loss(output.log().view(inputs.shape[0], -1), labels)
			if iternum>10000:
				print(loss.squeeze())
				print(output_model)
				print(maxlprob)
				print(lprob)
				# input("Press any key")
			is_oracle = (-loss).exp() > torch.rand_like(loss)
			to_be_summed = (is_oracle).squeeze().detach()

			if iternum == 1:
				ret_output= output_model.log_softmax(dim=1).reshape(output_model.shape[0],-1).detach()
				ret_likelihood = self.opts.optimizeropts.loss(ret_output.view(-1, self.opts.dataopts.classnum), labels).mean().detach()
				ret_entropy = -lprob.mean().detach()
				total_samples += float(inputs.shape[0])
				total_corrects += to_be_summed.sum().float()

			lpmgh = lprob #- (maxlprob)
			loss = ((-lpmgh.squeeze()[to_be_summed]).sum()/ batchsz) # The issue was with size of lprob not squeezed.
			definition.hasnan(loss)
			definition.hasinf(loss)
			loss.backward()
			# self.optimizer.step()
			# self.optimizer.zero_grad()
			# break
			if(to_be_summed.all()):
				break

			# break
			inputs = self.logical_index(inputs, to_be_summed ^ 1)
			labels = self.logical_index(labels, to_be_summed ^ 1)

		avg_iter = avg_iter/batchsz
		stats= dict(avg_iterations=avg_iter)
		# print(stats['jsd'],end='')
		return ret_likelihood,ret_output, total_corrects/total_samples, ret_entropy,stats

	def hyper_normalize(self,inputs,labels,usemin=False,concentration=2.0):
		batchsz = inputs.shape[0]
		sampler = Sampler(blockidx=-1)
		output_model,lprob,max_integral = self.model.max_integral(inputs,alpha=concentration)
		# output_model,lprob,max_integral = self.model(inputs)
		output, lprob, max_integral = sampler.max_integral(output_model,lprob, max_integral,alpha=concentration)
		# output, lprob, max_integral = sampler(output_model, lprob, max_integral)
		# max_integral = lsepair(max_integral, lprob)
		# max_integral = max_integral/concentration
		# lprob = (lprob*(concentration)).exp().detach()*lprob
		# (max_integral).mean().backward()
		lprob.mean().backward()


		return
	def test_grad(self,inputs,labels):
		for i in range(inputs.shape[0]):
			output=self.model(inputs[i:-1])
			output = output.log_softmax(dim=1)
			loss = self.opts.optimizeropts.loss(output.view(inputs.shape[0]-i, -1), labels[i:-1])

	def gpu_items(self,place):
		print(
			"________________________________________________________________________________________________________________________________________________________________")
		import gc
		num = 0
		for obj in gc.get_objects():
			try:
				if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
					num += 1
			# print(type(obj), obj.size())
			except:
				pass
		print(place + "{}".format(num))
		print(
			"________________________________________________________________________________________________________________________________________________________________")
		return

	def get_acc(self,output:Tensor,labels):
		assert output.dim()==2
		predlab = torch.argmax(output, 1, keepdim=False).squeeze()
		corrects  = (labels == predlab).sum().item()
		acc = corrects/output.shape[0]
		return acc*100

	def get_jsd(self, output):
		output = output.detach()
		meanoutput = output.logsumexp(dim=0,keepdim=True)
		meanoutput = meanoutput.log_softmax(dim=1)
		mfd = -(meanoutput - output).min(dim=1,keepdim=True)[0].mean()
		return mfd


	def run_epoch(self,prefixprint:str,epoch,path)->dict:
		run_loss = 0.0
		reg_loss = 0
		run_lprob_correct=0.0
		trytotal = 0
		corrects = 0
		entorpy = 0
		totalsamples = 0
		trials=0
		thisNorm=0
		ISNAN = False
		self.model.train()

		total_train_result = None
		# TODO: Train on batches
		for batch_n,data in enumerate(self.trainloader):

			inputs, labels = data
			inputs, labels = inputs.to(self.opts.epocheropts.device),labels.to(self.opts.epocheropts.device)
			this_samp_num = inputs.shape[0]
			alpha = self.opts.netopts.customdict["alphaPrior"]
			if batch_n ==0:
				fix_batch= inputs[0:min(inputs.shape[0],30),0:,0:,0:]
				fix_labels = labels[0:min(inputs.shape[0],30)]
				fix_batch = self.order_batch_by_label(fix_batch,fix_labels)
			log_prob_correct_temp=torch.zeros(1)
			if not self.opts.netopts.customdict["exact"]:
				with torch.autograd.set_detect_anomaly(False):
					loss,\
					output,\
					trys,\
					entropy,\
					stats= self.likelihood_optimization(inputs,labels,usemin=False,concentration=alpha)
					# self.hyper_normalize(inputs, labels, usemin=False, concentration=alpha)
					outputfull = output
					loss_conc = loss
			else:
				trys= 1
				# Exact

				output,logprob = self.model(inputs,drop=True,concentration=alpha)
				# output = alpha_lnorm(output,1,alpha)
				output = output - ((alpha*output).logsumexp(dim=1,keepdim=True))/alpha
				outputfull = output
				output = output.view(-1, self.opts.dataopts.classnum)
				loss = self.opts.optimizeropts.loss(output, labels).mean()
				stats = dict(avg_iterations=1)
				(loss).backward()
				output_conc= (output.detach()*alpha).log_softmax(dim=1)
				loss_conc = self.opts.optimizeropts.loss(output_conc, labels).mean()

			if batch_n % 4== 0:
				self.model.print(fix_batch, epoch, batch_n)
			thisNorm = self.block_grad(list(self.model.parameters()))
			self.optimizer.step()
			self.optimizer.zero_grad()
			# self.model.paint_stochastic_graph()


			''' Fill Result Dictionary'''
			temp_result = {
				'train_loss': loss.item(),
				'train_acc': self.get_acc(output,labels),
				'train_conc_loss': loss_conc.item(),
				'entropy':entropy,
				'grad_norm':thisNorm.item(),
				'jsd_output': self.get_jsd(output).item(),
				'trials':trys
			}
			if total_train_result is None:
				total_train_result = temp_result
			else:
				total_train_result = self.dict_lambda(total_train_result,temp_result,f=lambda x,y: (x*totalsamples +y*this_samp_num)/(totalsamples +this_samp_num))

			''' Print Output'''

			print("",end='\r')
			self.print_dict(total_train_result,prefix=' '+ prefixprint + '| '+ str(batch_n) +':',postfix=" ")
			totalsamples = totalsamples + inputs.shape[0]
			if hasnan(loss):
				ISNAN = True
				break

		scalar_dict = self.model.get_scalar_dict()
		stats = dict(trial= trials)
		scalar_dict.update(stats)
		print("", end='\r')

		# TODO: Evaluate Test
		self.model.eval()
		total_res_dict = None
		totalsamples = 0
		for batch_n,data in enumerate(self.testloader):
			with torch.set_grad_enabled(False):
				inputs, labels = data
				inputs, labels = inputs.to(self.opts.epocheropts.device),labels.to(self.opts.epocheropts.device)
				this_samp_num = inputs.shape[0]
				temp_result = self.val_stats(inputs,labels,iternum=self.opts.netopts.customdict['val_iters'])
				if total_res_dict is None:
					total_res_dict = temp_result
				else:
					total_res_dict = self.dict_lambda(total_res_dict,temp_result,f=lambda x,y: (x*totalsamples + y*this_samp_num)/(totalsamples + this_samp_num))

				totalsamples= totalsamples + this_samp_num

			if ISNAN:
				break

		total_res_dict.update(total_train_result)
		self.print_dict(total_res_dict, prefix=prefixprint)
		total_res_dict.update(scalar_dict)
		return total_res_dict,outputfull

	def test_intersect(self,input, iternum=None):
		self.model.eval()
		output = None
		for i in range(iternum):
			if output is None:
				output=self.model(input,drop=True)[0]
				output= output.log_softmax(dim=1)
			else:
				output = output + self.model(input,drop=True)[0].log_softmax(dim=1)

		output = (output).log_softmax(dim=1)
		return output

	def test_marginal(self, input, iternum=None):
		self.model.eval()
		output =None
		for i in range(iternum):
			if output is None:
				output = self.model(input,drop=True)[0].log_softmax(dim=1)
			else:
				output_temp = self.model(input,drop=True)[0].log_softmax(dim=1)
				output = LSE_pair(output,output_temp)
		output = output.log_softmax(dim=1)
		return output

	def test_special_case(self,input):
		self.model.eval()
		output = self.model(input,drop=False)[0].log_softmax(dim=1)
		return output

	def val_stats(self,inputs,labels,iternum=10):
		def stats(output,labels):
			output = output.log_softmax(dim=1)
			output= output.view(-1,self.opts.dataopts.classnum)
			loss =self.opts.optimizeropts.loss(output,labels).mean()
			pred_lab = output.argmax(dim=1,keepdim=False).squeeze()
			corrects= (pred_lab== labels).float()
			acc = corrects.mean()*100
			return loss,acc
		output_intersect= self.test_intersect(inputs, iternum=iternum)
		output_marginal = self.test_marginal(inputs, iternum=iternum)
		output_special = self.test_special_case(inputs)
		''' LOSSES'''
		loss_intersect, acc_intersect = stats(output_intersect,labels)
		loss_marginal, acc_marginal= stats(output_marginal,labels)
		loss_special, acc_special = stats(output_special,labels)

		alpha = self.opts.netopts.customdict["alphaPrior"]
		loss_conc_intersect,_ = stats(alpha *output_intersect, labels)
		loss_conc_marginal,_ = stats(alpha *output_marginal, labels)
		loss_conc_special,_ = stats(alpha *output_special, labels)
		res_dict= dict(test_loss_marginal=loss_marginal.item(),
		               test_loss_special=loss_special.item(),
		               test_loss_intersect=loss_intersect.item(),
		               test_acc_marginal= acc_marginal.item(),
		               test_acc_special=acc_special.item(),
		               test_acc_intersect=acc_intersect.item(),
		               test_loss_conc_marginal=loss_conc_marginal.item(),
		               test_loss_conc_special=loss_conc_special.item(),
		               test_loss_conc_intersect=loss_conc_intersect.item(),
		               )
		return res_dict

	def dict_lambda(self,*dict_list,f=lambda x:x):
		''' Applies the lambda function on the list of the dictionary, per key/val'''
		ret_dict = dict_list[0]
		for key in dict_list[0]:
			vals = []
			for this_dict in (dict_list):
				vals = vals + [this_dict[key]]
			ret_dict[key] = f(*vals)
		return ret_dict



	def run_many_epochs(self,path:str,save_result):
		self.model.to(self.opts.epocheropts.device)
		self.results = ResultStruct(path)
		self.path = path
		print(path)
		self.opts.print(printer=self.print)
		for epoch in range(self.opts.epocheropts.epochnum):
			prefixtext = 'Epoch %d' % epoch
			epochres,outputsample = self.run_epoch(prefixtext,epoch,path)

			self.results.add_epoch_res_dict(epochres,epoch,save_result)
			if save_result:
				state = {'model_string':self.opts.netopts.modelstring,
				         'model_state':self.model.state_dict(),
				         'opimizer_state': self.optimizer.state_dict(),
				         'epoch':epoch}
				torch.save(state,os.path.join(path,'final_model.model'))
				torch.save(self.results.resultdict, os.path.join(path,'result_dict.res'))
			self.opts.optimizeropts.lr_sched.step()
			#with torch.set_grad_enabled(False):
			#	generated_images = self.model.generate(sample(outputsample,1,1)[0])
			#	imagepath = './GenImages/Images'+ '_epoch_'+ str(epoch+1)+'.bmp'
			#	save_image(generated_images,imagepath,normalize=True,scale_each=True)
		self.model.to(torch.device('cpu'))
		return self.results


	def print(self,string,end='\n'):
		path = self.path
		log_file = open(os.path.join(path,'log.txt'),"a")
		print(str(string),end=end)
		log_file.write(str(string)+end)
		log_file.close()

	def print_dict(self,res_dict:Dict, prefix='', postfix='\n'):
		string = prefix
		for key, val in res_dict.items():
			string = string +' '+ key + ': ' + '%.4f'%val
		string = string
		self.print(string,end=postfix)
