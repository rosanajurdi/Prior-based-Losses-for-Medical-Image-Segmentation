#!/usr/env/bin python3.6

from typing import List

import torch
import numbers
import math
from torch import Tensor, einsum
from torch import nn
from utils import simplex, one_hot
from scipy.ndimage import distance_transform_edt, morphological_gradient, distance_transform_cdt
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from torch.nn import functional as F

def contour(x):
    '''
    Differenciable aproximation of contour extraction
    
    '''   
    min_pool_x = torch.nn.functional.max_pool2d(x*-1, (3, 3), 1, 1)*-1
    max_min_pool_x = torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1)
    contour = torch.nn.functional.relu(max_min_pool_x - min_pool_x)
    return contour


def soft_skeletonize(x, thresh_width=10):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''
    for i in range(thresh_width):
        min_pool_x = torch.nn.functional.max_pool2d(x*-1, (3, 3), 1, 1)*-1
        max_min_pool_x = torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1)
        contour = torch.nn.functional.relu(max_min_pool_x - min_pool_x)
        x = torch.nn.functional.relu(x - contour)
    return x




class contour_loss():
    '''
    inputs shape  (batch, channel, height, width).
    calculate clDice loss
    Because pred and target at moment of loss calculation will be a torch tensors
    it is preferable to calculate target_skeleton on the step of batch forming,
    when it will be in numpy array format by means of opencv
    '''
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        
        print(f"Initialized {self.__class__.__name__} with {kwargs}")
    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        b, _, w, h = pc.shape
        cl_pred = contour(pc).sum(axis=(2,3))
        target_contour = contour(tc).sum(axis=(2,3))
        big_pen: Tensor = (cl_pred - target_contour) ** 2
        contour_loss = big_pen / (w * h)
    
        return contour_loss.mean(axis=0)

def compute_morphogradient(segmentation):
    res = np.zeros(segmentation.shape)
    print(segmentation.shape)
    for i in range(segmentation.shape[0]):
        posmask = segmentation[i]
        negmask = ~posmask
        res[i] = morphological_gradient(posmask[0].astype(np.float32), size=(3,3))
    return res

    
class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = target[:, self.idc, ...].type(torch.float32)

        loss = - einsum("bcwh,bcwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss


class SurfaceLoss():
    '''
    Boundary loss implementation 
    Inputs:
    @probs: probability maps provded from the output of the network 
    @dc: distance maps computed when the dataset class is initialized
    outputs:
    @loss: boundary loss
    @description: 
    the loss finetunes the probability maps by the groundtruth distance map representations.
    '''
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss



class HDDTBinaryLoss():
    '''
    Hausdorf loss implementation for binary segmentation 
    '''
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, net_output: Tensor, target: Tensor, _: Tensor) -> Tensor:
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        
        pc = net_output[:, self.idc, ...].type(torch.float32)
        gt = target[:, self.idc, ...].type(torch.float32)
        with torch.no_grad():
            pc_dist = compute_edts_forhdloss(pc.detach().cpu().numpy()>0.5)
            gt_dist = compute_edts_forhdloss(gt.detach().cpu().cpu().numpy()>0.5)
        # print('pc_dist.shape: ', pc_dist.shape)
        
        pred_error = (gt - pc)**2
        dist = pc_dist**2 + gt_dist**2 # \alpha=2 in eq(8)

        dist = torch.from_numpy(dist)
        if dist.device != pred_error.device:
            dist = dist.to(pred_error.device).type(torch.float32)

        multipled = torch.einsum("bxyz,bxyz->bxyz", 
                                 pred_error.reshape(-1,1,pred_error.shape[1], pred_error.shape[2]), 
                                 dist.reshape(-1,1,dist.shape[1], dist.shape[2]))
        hd_loss = multipled.mean()

        return hd_loss


class soft_cldice_loss():
    '''
    inputs shape  (batch, channel, height, width).
    calculate clDice loss
    Because pred and target at moment of loss calculation will be a torch tensors
    it is preferable to calculate target_skeleton on the step of batch forming,
    when it will be in numpy array format by means of opencv
    '''
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")
    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)
        b, _, w, h = pc.shape
        cl_pred = soft_skeletonize(pc)
        target_skeleton = soft_skeletonize(tc)
        big_pen: Tensor = (cl_pred - target_skeleton) ** 2
        contour_loss = big_pen / (w * h)
    
        return contour_loss.mean()
    

import numpy as np
import cv2
import torch

def opencv_skelitonize(img):
    skel = np.zeros(img.shape, np.uint8)
    img = img.astype(np.uint8)
    size = np.size(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    return skel









def norm_intersection(center_line, vessel):
    '''
    inputs shape  (batch, channel, height, width)
    intersection formalized by first areas
    x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
    '''
    smooth = 1.
    s = center_line.shape[:2]
    clf = center_line.view(s[0], s[1], -1)
    v = vessel.shape[:2]
    vf = vessel.view(v[0],v[1], -1)
    intersection = (clf * vf).sum(-1)
    return (intersection + smooth) / (clf.sum(-1) + smooth)


def compute_edts_forhdloss(segmentation):
    res = np.zeros(segmentation.shape)
    for i in range(segmentation.shape[0]):
        posmask = segmentation[i]
        negmask = ~posmask
        res[i] = distance_transform_edt(posmask) + distance_transform_edt(negmask)
    return res



####################################################################################

class NaivePenalty():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.C = len(self.idc)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)
        assert probs.shape == target.shape

        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        k = bounds.shape[2]  # scalar or vector
        value: Tensor = self.__fn__(probs[:, self.idc, ...])
        lower_b = bounds[:, self.idc, :, 0]
        upper_b = bounds[:, self.idc, :, 1]

        assert value.shape == (b, self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape

        too_big: Tensor = (value > upper_b).type(torch.float32)
        too_small: Tensor = (value < lower_b).type(torch.float32)

        big_pen: Tensor = (value - upper_b) ** 2
        small_pen: Tensor = (value - lower_b) ** 2

        res = too_big * big_pen + too_small * small_pen

        loss: Tensor = res / (w * h)

        return loss.mean()



