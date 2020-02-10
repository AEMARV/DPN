import torch
import definition as definition

def BintoLogFSD(image):
	shape = image.shape
	imagel1 = image.clamp(definition.epsilon,1).log()
	imagel0 = (1-image).clamp(definition.epsilon, 1).log()
	image =[imagel0,imagel1]
	out = torch.zeros((2**shape[0],shape[1],shape[2]))
	for i in range(2**shape[0]):
		for bit in range(shape[0]):
			curbit = (i// 2**bit)%2
			out[i,0:,0:] += image[curbit][bit,0:,0:]

	return out

def BintoLogFSDFact(image):
	shape = image.shape
	image = image.unsqueeze(dim=3).permute((3,1,2,0))
	imagemin = image.min()
	image = image- imagemin
	imagemax = image.max()
	image = image /imagemax
	imagel1 = image.clamp(definition.epsilon,1).log()
	imagel0 = (1-image).clamp(definition.epsilon, 1).log()
	image = torch.cat((imagel0,imagel1),dim=0)
	return image

