import numpy as np
import torch
from tqdm import tqdm
from grad_cam import GradCam
from itertools import product

class RoiPlot(object):

	def __init__(self,model,**kwargs):
		self.target_layer_names = kwargs.get('target_layer_names',["denseblock4"])
		self.cuda_id = kwargs.get('cuda_id',0)
		self.resize_shape = kwargs.get('resize_shape')
		self.model = model
		self.gc = GradCam(model = model,target_layer_names = self.target_layer_names, cuda_id = self.cuda_id)

	# Batch,Channel, H ,W
	def gradcam_maps(self,patches,target_indices):
		new_size = self.map_shape(patches = patches)
		maps = [self.gc(patches,label,resize=new_size) for label in target_indices]
		return maps
	
	
	
	def extract(self,roi_batch_iterator_fix_len,dimensions,target_indices,n_classes = 2):
		self.model.eval()
		enum = enumerate(roi_batch_iterator_fix_len)
		#softmax=(nn.Softmax(dim=1))
		
		if len(dimensions) ==3:
			dimensions = dimensions+(1,)
		elif len(dimensions)<=2 or len(dimensions)>=5:
			raise ValueError('dimensions is supposed to be a 3 or 4-sized tuple')
		num_patch,height,width,channel = dimensions
		
		output_shape = (width,height)
		#channel = 1, for heatmaps -->omitted
		map_store = [np.zeros((num_patch,)+output_shape) for x in target_indices]
		pred_store = np.zeros((num_patch,))
	
		# in case the batch is not homogeneous in size.
		arr_top = 0;
		
		#use *rest may unpack the torch itself.
		for batch_id,img in enum:
			output_map = [self.gc(img,index,resize=output_shape) for index in target_indices]
			pred_map = self.model(img.to(self.gc.device))
			output_fetch = pred_map.detach().squeeze().cpu().numpy()
			if len(output_fetch.shape)==1:
				output_fetch = np.expand_dims(output_fetch,axis=0)
			prediction = np.argmax(output_fetch,axis = 1).flatten()
			#output_map is non-empty
			batch_size = output_map[0].shape[0]
			for jj, index_val in enumerate(target_indices):
				#print(jj,output_map[jj].shape,batch_size,img.shape[0],'|',arr_top,map_store[jj].shape)		
				map_store[jj][arr_top: arr_top+ batch_size,] = output_map[jj]
			#print(prediction.shape,pred_store.shape)
			pred_store[arr_top: arr_top+ batch_size,] = prediction
			arr_top += batch_size
		return map_store,pred_store
	
	@staticmethod
	def sanitize_by_mask(self):
		...

	def map_shape(self,patches = None):
		return self.resize_shape if (self.resize_shape is not None) else patches.shape[2:]
		
		
		
	@staticmethod
	def reconstruct_strided(patches, image_size,stride = 1):
		"""Reconstruct the image from all of its patches. Strided.
		Modified from its sklearn counterpart
		"""
		i_h, i_w = image_size[:2]
		p_h, p_w = patches.shape[1:3]
		img = np.zeros(image_size)
		# compute the dimensions of the patches array
		n_h = (i_h - p_h)//stride+ 1
		n_w = (i_w - p_w)//stride + 1
		#for sliding window coords, not strided range --> 128*coords instead.
		for p, (i, j) in zip(patches, product(range(0,n_h), range(0,n_w))):
			img[i*stride:i*stride + p_h, j*stride:j*stride + p_w] += p
		

		#for every step, averaging the overlapping layers
		for i in range(0,n_h):
			for j in range(0,n_w):
				# divide by the amount of overlap
				# XXX: is this the most efficient way? memory-wise yes, cpu wise?
				# use number of steps to calculate number of overlapping. some Overlapping skipped if stride >1  
				# (patch-1)//stride +1 is to calculate how many step it needs to fully pass a whole patch --> # of overlapping
				#Note: coordiante: <i,j, stride> -> [i*stride,j*stride]
				#end of index is taken care of by numpy
				img[i*stride:i*stride+stride, j*stride:j*stride+stride] /= float(min(i + 1, (p_h-1)//stride+1, n_h - i) *
								   min(j + 1, (p_w-1)//stride+1, n_w - j))
		return img
		