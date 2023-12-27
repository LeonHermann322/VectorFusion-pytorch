import torch 
import cv2
import numpy as np
import copy

from libs.modules.edge_map.DoG import XDoG

class SparseCoordInit:

    def __init__(self, pred, gt, format='[bs x c x 2D]', quantile_interval=200, nodiff_thres=0.1):
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(gt):
            gt = gt.detach().cpu().numpy()

        if format == '[bs x c x 2D]':
            self.map = ((pred[0] - gt[0]) ** 2).sum(0)
            self.reference_gt = copy.deepcopy(np.transpose(gt[0], (1, 2, 0)))
        elif format == ['[2D x c]']:
            self.map = (np.abs(pred - gt)).sum(-1)
            self.reference_gt = copy.deepcopy(gt[0])
        else:
            raise ValueError

        # OptionA: Zero too small errors to avoid the error too small deadloop
        self.map[self.map < nodiff_thres] = 0
        quantile_interval = np.linspace(0., 1., quantile_interval)
        quantized_interval = np.quantile(self.map, quantile_interval)
        # remove redundant
        quantized_interval = np.unique(quantized_interval)
        quantized_interval = sorted(quantized_interval[1:-1])
        self.map = np.digitize(self.map, quantized_interval, right=False)
        self.map = np.clip(self.map, 0, 255).astype(np.uint8)
        self.idcnt = {}
        for idi in sorted(np.unique(self.map)):
            self.idcnt[idi] = (self.map == idi).sum()
        # remove smallest one to remove the correct region
        self.idcnt.pop(min(self.idcnt.keys()))

    def __call__(self):
        if len(self.idcnt) == 0:
            h, w = self.map.shape
            return [np.random.uniform(0, 1) * w, np.random.uniform(0, 1) * h]

        target_id = max(self.idcnt, key=self.idcnt.get)
        _, component, cstats, ccenter = cv2.connectedComponentsWithStats(
            (self.map == target_id).astype(np.uint8),
            connectivity=4
        )
        # remove cid = 0, it is the invalid area
        csize = [ci[-1] for ci in cstats[1:]]
        target_cid = csize.index(max(csize)) + 1
        center = ccenter[target_cid][::-1]
        coord = np.stack(np.where(component == target_cid)).T
        dist = np.linalg.norm(coord - center, axis=1)
        target_coord_id = np.argmin(dist)
        coord_h, coord_w = coord[target_coord_id]

        # replace_sampling
        self.idcnt[target_id] -= max(csize)
        if self.idcnt[target_id] == 0:
            self.idcnt.pop(target_id)
        self.map[component == target_cid] = 0
        return [coord_w, coord_h]


class RandomCoordInit:
    def __init__(self, canvas_width, canvas_height):
        self.canvas_width, self.canvas_height = canvas_width, canvas_height

    def __call__(self):
        w, h = self.canvas_width, self.canvas_height
        return [np.random.uniform(0, 1) * w, np.random.uniform(0, 1) * h]


class NaiveCoordInit:
    def __init__(self, pred, gt, format='[bs x c x 2D]', replace_sampling=True):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()

        if format == '[bs x c x 2D]':
            self.map = ((pred[0] - gt[0]) ** 2).sum(0)
        elif format == ['[2D x c]']:
            self.map = ((pred - gt) ** 2).sum(-1)
        else:
            raise ValueError
        self.replace_sampling = replace_sampling

    def __call__(self):
        coord = np.where(self.map == self.map.max())
        coord_h, coord_w = coord[0][0], coord[1][0]
        if self.replace_sampling:
            self.map[coord_h, coord_w] = -1
        return [coord_w, coord_h]

class AttentionCoordInit:
    def __init__(self, attention_map, softmax_temp, canvas_height, canvas_width, num_stages,num_paths,image2clip_input,xdog_intersec):
        self.attention_map = attention_map 
        self.softmax_temp = softmax_temp
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.num_stages = num_stages
        self.num_paths = num_paths
        self.image2clip_input = image2clip_input
        self.xdog_intersec = xdog_intersec
        
        self.thresh = self.set_attention_threshold_map()
        self.strokes_counter = 0 

    def __call__(self):
        if self.strokes_counter == self.num_stages * self.num_paths:
            self.thresh = self.set_attention_threshold_map()
            self.strokes_counter = 0 
            
        point = self.inds[self.strokes_counter]
        self.strokes_counter += 1
        return point 
    
    def set_attention_threshold_map(self):
        attn_map = (self.attention_map - self.attention_map.min()) / \
                   (self.attention_map.max() - self.attention_map.min())
                   
        if self.xdog_intersec:
            xdog = XDoG(k=10)
            im_xdog = xdog(self.image2clip_input[0].permute(1, 2, 0).cpu().numpy())
            intersec_map = (1 - im_xdog) * attn_map
            attn_map = intersec_map
        attn_map_soft = np.copy(attn_map)
        attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], tau=self.softmax_temp)

        # select points
        k = self.num_stages * self.num_paths
        self.inds = np.random.choice(range(attn_map.flatten().shape[0]),
                                     size=k,
                                     replace=False,
                                     p=attn_map_soft.flatten())
        self.inds = np.array(np.unravel_index(self.inds, attn_map.shape)).T
        self.inds  = self.inds.tolist()
        
        return attn_map_soft
    
    @staticmethod
    def softmax(x, tau=0.2):
            e_x = np.exp(x / tau)
            return e_x / e_x.sum()
