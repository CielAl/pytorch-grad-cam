# add batch
"""
Modified to take batch input.

Original Author: Jacob Gildenblat; github: https://github.com/jacobgil
"""
import cv2
import numpy as np
import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from torch.autograd import Function
from matplotlib.colors import Colormap
from matplotlib import cm


class FeatureExtractor(object):
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
        # print(self.model)
        for name, module in self.model._modules.items():
            output = module(output)
            if name in self.target_layers:
                # print(name,(self.target_layers))
                output.register_hook(self.save_gradient)
                features += [output]
        return features, output


class ModelOutputs(object):
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
        target_activations, output = self.feature_extractor(x)

        # output = self.model.features.denseblock4.denselayer2.conv2(output)
        # output = self.model.features.norm5(output)
        output = F.relu(output, inplace=True)
        output = F.adaptive_avg_pool2d(output, (1, 1)).view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      color_map: Colormap = cm.coolwarm,
                      name: str = None):
    """
    Reshape Overlay the GradCam output (mask) to the input img given color map.
    Args:
        img (): Expected to be within [0, 1]. Automatically normalized if dtype is uint8
        mask (): Output of the GradCam.
        color_map (): Matplotlib colormap
        name (): Name of the output img. Default is None, which disables the imwrite.

    Returns:

    """
    if img.dtype == np.uint8:
        img = img / 255.

    # BGR order
    heatmap = color_map(mask)[:, :, 0:3]
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    print('hm_max', heatmap.max())
    # norm to [0,1]

    cam = heatmap + np.float32(img)

    cam = cam / np.max(cam)

    cam *= 255
    out = np.uint8(cam)
    if name is not None:
        # applyColorMap returns a BGR out. So it is not necessary to convert the channel order while writing.
        cv2.imwrite(name, cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    return out, heatmap

    #   grad (32, 128, 8, 8)
    #   weight (32 128,)
    #   target (32 128, 8, 8)
    #    cam (32 8, 8)


class GradCam:
    def __init__(self, model, target_layer_names, cuda_id):
        self.model = model
        self.model.eval()
        self.device = torch.device(f'cuda:{cuda_id}' if cuda_id is not None and torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input_data):
        return self.model(input_data)

    def __call__(self, input_data: torch.Tensor, index=None, resize=None):
        features, output = self.extractor(input_data.to(self.device))

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((output.shape[0], output.size()[-1]), dtype=np.float32)
        one_hot[:, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(self.device) * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        weights = grads_val.mean(axis=(2, 3), keepdims=True)  # [0, :]
        weights = torch.from_numpy(weights).to(self.device)
        cam = F.relu((weights * target).mean(dim=1), inplace=True).cpu().data.numpy()

        min_val = np.min(cam, axis=(1, 2), keepdims=True)
        max_val = np.max(cam, axis=(1, 2), keepdims=True)
        diff = max_val - min_val
        diff[diff == 0] = np.inf
        cam = (cam - min_val) / diff
        # cam = cam / (np.max(cam,axis=(1,2),keepdims=True)
        cam[np.isnan(cam)] = 0
        if resize is not None:
            cam = np.moveaxis(cam, 0, -1)  # cv2.resize only support batches if with dimension H*W*Batch
            cam = cv2.resize(cam, resize)
            cam = np.moveaxis(cam, -1, 0)
        # cam = np.uint8(255*cam)
        return cam


class GuidedBackpropReLU(Function):

    def forward(self, input_data, **kwargs):
        positive_mask = (input_data > 0).type_as(input_data)
        output = torch.addcmul(torch.zeros(input_data.size()).type_as(input_data), input_data, positive_mask)
        # noinspection PyUnresolvedReferences
        self.save_for_backward(input_data, output)
        return output

    def backward(self, grad_output):
        input_data, output = self.saved_tensors
        positive_mask_1 = (input_data > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_data.size()).type_as(input_data),
                                   torch.addcmul(torch.zeros(input_data.size()).type_as(input_data), grad_output,
                                                 positive_mask_1), positive_mask_2)

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

    def forward(self, x):
        return self.model(x)

    def __call__(self, input_data, index=None):
        if self.cuda:
            output = self.forward(input_data.cuda())
        else:
            output = self.forward(input_data)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input_data.grad.cpu().data.numpy()
        # output = output[0,:,:,:]

        return output
