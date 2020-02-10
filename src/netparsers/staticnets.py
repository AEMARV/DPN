from src.netparsers.parseutils import *
import torch.tensor
import math
import torch.nn.functional as F
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx


class stochasticGraph(object):
	def __init__(self):
		pass
	@staticmethod
	def node_id(layer,state):
		return 'H'+str(layer)+'='+str(state)
	@staticmethod
	def edge_container(matlist):
		for lnum,mat in enumerate(matlist):
			for R in range(mat.shape[0]):
				for D in range(mat.shape[1]):
					thisdict= dict(color=mat[R,D].exp().item())
					idD = stochasticGraph.node_id(lnum,D)
					idR = stochasticGraph.node_id(lnum+1, R)
					yield (idD,idR,thisdict)
	@staticmethod
	def node_container(matlist:List):
		matlist = matlist
		for lnum,mat in enumerate(matlist):
			for D in range(mat.shape[1]):
				idD = stochasticGraph.node_id(lnum,D)
				yield (idD,dict(pos=(lnum,D/mat.shape[1])))

		m = matlist[-1]
		for D in range(mat.shape[0]):
			lnum = matlist.__len__()
			idD = stochasticGraph.node_id(lnum,D)
			yield (idD,dict(color='red',pos=(lnum,D/mat.shape[0])))



class StaticNet(MyModule):
	''' A static Module constructed from a model string. The model string specs are parsed with the static functions
	in the class.
	Constructor:

	StaticNet(modelstring, opts)

	modelstring: the model specification string with delimiter '->'
	opts: opts struct.
	'''
	def __init__(self,modelstring,inputchannels,weightinit=None,biasinit=None,sample_data=None):
		super(StaticNet, self).__init__(blockidx=0)
		# 1 input image channel, 6 output channels, 5x5 square convolution
		# kernel
		self.layerlist = self.parse_model_string(modelstring,inputchannels['chansz'],inputchannels['icnum'])
		for bloacknum,layer in enumerate(self.layerlist):
			if isinstance(layer,nn.Conv2d):
				weightinit(layer.weight.data)
				biasinit(layer.bias.data)

			self.add_module('block'+str(bloacknum),layer)
		self.sampler = Sampler(blockidx=-1)

	def generate(self,y):

		for l in reversed(self.layerlist):
			if isinstance(l, torch.nn.Conv2d):
				#y = y - l.add_bias(y,)
				bias_reshaped = l.bias.view(1,y.shape[1],1,1)
				y = y - bias_reshaped.data
				wnorm = (l.weight.data**2).sum(dim=1,keepdim=True).sum(dim=2,keepdim=True).sum(dim=3,keepdim=True).sqrt()
				w = l.weight.data/wnorm

				weightmat = l.weight.view(3,3).inverse().view(3,3,1,1)
				y = F.conv_transpose2d(y,weightmat)
			elif isinstance(l, torch.nn.Sigmoid):
				y=y

			else:
				y= l.generate(y)


		return y

	def get_stochastic_mats(self):
		stochastic_list = []
		for layer in self.modules():
			if isinstance(layer,BayesFunc):
				m = layer.get_stochastic_mat()
				stochastic_list = stochastic_list + [m]
		return  stochastic_list

	def paint_stochastic_graph(self):
		''' gets a list of stochastic matrices as input or from the object and paints the graph
		The intensity of the edges of the garph represents the transition probability
		'''
		graph = nx.DiGraph()

		m = self.get_stochastic_mats()
		m_first= m[0]
		input_sz = m_first.shape[1]
		graph.add_nodes_from(stochasticGraph.node_container(m))
		graph.add_edges_from(stochasticGraph.edge_container(m))
		edges = graph.edges()
		pos= graph.nodes(data='pos')
		colors = [graph[u][v]['color'] for u, v in edges]
		plt.cla()
		nx.draw_networkx(graph,node_size=10,edge_color=colors,pos=pos,edge_cmap=plt.cm.Reds,edge_vmin=0,edge_vmax=1)
		plt.show(block=False)
		plt.pause(0.000001)


	def accumulate_lprob(self,lp,usemin=False):

		if lp is None: return None
		if usemin:
			# minlp = lp.min(dim=1, keepdim=True)[0]\
			# 	.min(dim=2, keepdim=True)[0]\
			# 	.min(dim=3, keepdim=True)[0]\
			# 	.min(dim=4,keepdim=True)[0].squeeze()
			minlp = -(-lp).logsumexp(dim=(1,2,3),keepdim=True).squeeze()
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
	def p_invert(self,state_list,y):
		state_list.reverse()
		sampler_id = 0
		logprob= None
		for i, layer in enumerate(reversed(self.layerlist)):
			if isinstance(layer, Sampler):
				y,logprob_temp =layer.p_invert(y,state_list[sampler_id])
				logprob_temp = self.accumulate_lprob(logprob_temp)
				logprob = self.accumulate_lprob_pair(logprob, logprob_temp)
				sampler_id += 1
			elif isinstance(layer, MyModule):
				y = layer.p_invert(y)
		return logprob

	def jsd(self,xl):
		xl = xl.detach()
		xl = xl- xl.logsumexp(dim=1,keepdim=True)
		log_dim= math.log(xl.shape[1])

		xp = xl.exp()
		mean_ent = -xl*xp
		mean_ent = mean_ent.sum(dim=1,keepdim=True).mean()/log_dim
		xp_mean = xp.mean(dim=(0,2,3),keepdim=True)
		xl_mean = xp_mean.log()
		ent_mean = -xp_mean*xl_mean
		ent_mean = ent_mean.sum(dim=1,keepdim=True).mean()/log_dim
		return ent_mean-mean_ent

	def get_lrob_model(self):
		prior = None
		lrob = 0
		for i, layer in enumerate(self.layerlist):
			if isinstance(layer, BayesFunc):
				temp_lrob, prior = layer.get_lrob_model(prior)
				lrob = lrob + temp_lrob.sum()

		return lrob,None
	def forward(self, x:Tensor,mode='likelihood',usemin=False,concentration=1.0,drop=True):
		# Max pooling over a (2, 2) window
		logprob= torch.zeros(1).type_as(x)
		maxlrob= None
		first_sampler = False
		for i,layer in enumerate(self.layerlist):
			if isinstance(layer,Sampler):
				if first_sampler:
					x,_,_ = layer(x,logprob,maxlrob, alpha=concentration)
					first_sampler = False
				else:

					x, logprob, maxlrob = layer(x,logprob,maxlrob, alpha=concentration)
			elif isinstance(layer,FDropOut):
				x= layer(x,force_drop=drop)
			else:
				x = layer(x)
		return x,logprob, maxlrob

	def max_integral(self,x:Tensor, alpha=1):
		logprob = torch.zeros(1).type_as(x)
		maxlrob = None
		for i, layer in enumerate(self.layerlist):
			if isinstance(layer, Sampler):
				x, logprob, maxlrob = layer.max_integral(x, logprob, maxlrob, alpha=alpha)
			else:
				x = layer(x)
		return x, logprob, maxlrob

	''' String Parsers'''
	def parse_model_string(self, modelstring:str, in_n_channel,in_icnum):
		layer_list_string = modelstring.split('->')
		layer_list = []
		out_n_channel = in_n_channel
		blockidx_dict= {}
		for blocknum, layer_string in enumerate(layer_list_string, 0):
			layer,out_n_channel,in_icnum= parse_layer_string(layer_string,out_n_channel,in_icnum,blockidx_dict)
			if layer is not None:
				layer_list += [layer]
		return layer_list

