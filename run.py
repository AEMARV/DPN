from src.experiment.Experiments import *
import os

def boolprompt(question):
	answer=''
	while(answer.lower!='n' or answer.lower()!='y'):
		answer = 'n'# input(question+' [y]/[n]')
		if answer[0].lower()=='y':
			return True
		elif answer[0].lower()=='n':
			return False
		else:
			print('please answer with y/n characters')



if __name__ == '__main__':
	# exp = NIN_Dropout(1)
	# exp.run()
	exp = VGG_PMAP_CIFAR10_Try(1)
	# exp = Synthetic_PMaps(1)
	exp.run()

