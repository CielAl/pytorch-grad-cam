#add batch
import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse


import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, output):
        features = []
        self.gradients = []
        #print(self.model)
        for name, module in self.model._modules.items():
            output =  module(output)
            if name in self.target_layers:
                #print(name,(self.target_layers))
                output.register_hook(self.save_gradient)
                features += [output]
        return features, output

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)
		
		#output = self.model.features.denseblock4.denselayer2.conv2(output)
		#output = self.model.features.norm5(output)
		output = F.relu(output, inplace=True)
		output = F.adaptive_avg_pool2d(output, (1, 1)).view(output.size(0), -1)
		output = self.model.classifier(output)
		return target_activations, output

def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = Variable(preprocessed_img, requires_grad = True)
	return input

def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	out = np.uint8(255 * cam)
	cv2.imwrite("cam.jpg", out)
	return out
class GradCam:
	def __init__(self, model, target_layer_names, cuda_id):
		self.model = model
		self.model.eval()
		self.device = torch.device(f'cuda:{cuda_id}' if cuda_id is not None and torch.cuda.is_available() else 'cpu')
		self.model = model.to(self.device)
		self.extractor = ModelOutputs(self.model, target_layer_names)

	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, index = None,resize = None):
		features, output = self.extractor(input.to(self.device))

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((output.shape[0], output.size()[-1]), dtype = np.float32)
		one_hot[:,index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		one_hot = torch.sum(one_hot.to(self.device) * output)
	

		self.model.features.zero_grad()
		self.model.classifier.zero_grad()
		one_hot.backward(retain_graph = True)

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
	
		target = features[-1]
		
		target = target.cpu().data.numpy()#[0, :]
	
		weights = np.mean(grads_val, axis = (2, 3),keepdims = True)#[0, :]

		

		'''
			grad (32, 128, 8, 8)
			weight (32 128,)
			target (32 128, 8, 8)
			cam (32 8, 8)		
		'''
		
		weights = torch.from_numpy(weights).to(self.device)
		target = torch.from_numpy(target).to(self.device)
		cam  =  F.relu((weights * target).mean(dim = 1), inplace=True).cpu().data.numpy()
		if resize is not None:
			cam = cv2.resize(cam, resize)
		cam = cam - np.min(cam,axis=(1,2),keepdims=True)
		cam = cam / np.max(cam,axis=(1,2),keepdims=True)
		return cam

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
	def __init__(self, model, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		# replace ReLU with GuidedBackpropReLU
		for idx, module in self.model.features._modules.items():
			if module.__class__.__name__ == 'ReLU':
				self.model.features._modules[idx] = GuidedBackpropReLU()

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			output = self.forward(input.cuda())
		else:
			output = self.forward(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.features.zero_grad()
		self.model.classifier.zero_grad()
		one_hot.backward(retain_graph=True)

		output = input.grad.cpu().data.numpy()
		#output = output[0,:,:,:]

		return output


