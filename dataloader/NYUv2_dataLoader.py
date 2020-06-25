import os, random, time, copy, sys 
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
from scipy import misc
import matplotlib.pyplot as plt
import PIL.Image

import skimage.transform 
import blosc, struct

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms

class NYUv2_dataLoader(Dataset):
    def __init__(self, root_dir, set_name='train', size=[240, 320], rgb=True, downsampleDepthFactor=1, training_depth='inpaint'):
        # training depth option: inpaint | original
        self.root_dir = root_dir  
        self.size = size
        self.set_name = set_name
        self.training_depth = training_depth
        self.rgb = rgb
        self.current_set_len = 0
        self.path2files = []
        self.downsampleDepthFactor = downsampleDepthFactor
        self.NYU_MIN_DEPTH_CLIP = 0.0
        self.NYU_MAX_DEPTH_CLIP = 10.0
        
        curfilenamelist = os.listdir(path.join(self.root_dir, self.set_name, 'rgb'))
        self.path2files += [path.join(self.root_dir, self.set_name, 'rgb')+'/'+ curfilename for curfilename in curfilenamelist]
        self.current_set_len = len(self.path2files)   
        
        self.TF2tensor = transforms.ToTensor()
        self.TF2PIL = transforms.ToPILImage()
        self.TFNormalize = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        self.funcResizeTensor = nn.Upsample(size=self.size, mode='nearest', align_corners=None)
        self.funcResizeDepth = nn.Upsample(size=[int(self.size[0]*self.downsampleDepthFactor),
                                                 int(self.size[1]*self.downsampleDepthFactor)], 
                                           mode='nearest', align_corners=None)
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):
        filename = self.path2files[idx]
        image = PIL.Image.open(filename)
        image = np.array(image).astype(np.float32) / 255.
        
        if self.set_name == 'train':
            if self.training_depth == 'original':
                depthname = filename.replace('rgb','depth').replace('png','bin')
            else:
                depthname = filename.replace('rgb','depth_inpainted').replace('png','bin')
        else:
            # use real depth for validation and testing
            depthname = filename.replace('rgb','depth').replace('png','bin')

        depth = read_array_compressed(depthname)
        
        if self.set_name =='train' and np.random.random(1)>0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
        
        # rescale depth samples in training phase
        if self.set_name == 'train':
            depth = np.clip(depth, self.NYU_MIN_DEPTH_CLIP, self.NYU_MAX_DEPTH_CLIP) # [0, 10]
            depth = ((depth/self.NYU_MAX_DEPTH_CLIP) - 0.5) * 2.0 # [-1, 1]

        image = self.TF2tensor(image)
        image = self.TFNormalize(image)
        image = image.unsqueeze(0)
            
        depth = np.expand_dims(depth, 2)
        depth = self.TF2tensor(depth)
        depth = depth.unsqueeze(0)
        
        image = processNYU_tensor(image)
        depth = processNYU_tensor(depth)

        image = self.funcResizeTensor(image)
        depth = self.funcResizeTensor(depth)
        
        if self.downsampleDepthFactor != 1:
            depth = self.funcResizeDepth(depth)
            
        if self.rgb:
            image = image.squeeze(0)
        else:
            image = image.mean(1)
            image = image.squeeze(0).unsqueeze(0)

        depth = depth.squeeze(0)           
        return image, depth
     
def ensure_dir_exists(dirname, log_mkdir=True):
    """
    Creates a directory if it does not already exist.
    :param dirname: Path to a directory.
    :param log_mkdir: If true, a debug message is logged when creating a new directory.
    :return: Same as `dirname`.
    """
    dirname = path.realpath(path.expanduser(dirname))
    if not path.isdir(dirname):
        # `exist_ok` in case of race condition.
        os.makedirs(dirname, exist_ok=True)
        if log_mkdir:
            log.debug('mkdir -p {}'.format(dirname))
    return dirname


def read_array(filename, dtype=np.float32):
    """
    Reads a multi-dimensional array file with the following format:
    [int32_t number of dimensions n]
    [int32_t dimension 0], [int32_t dimension 1], ..., [int32_t dimension n]
    [float or int data]

    :param filename: Path to the array file.
    :param dtype: This must be consistent with the saved data type.
    :return: A numpy array.
    """
    with open(filename, mode='rb') as f:
        content = f.read()
    return bytes_to_array(content, dtype=dtype)


def read_array_compressed(filename, dtype=np.float32):
    """
    Reads a multi-dimensional array file compressed with Blosc.
    Otherwise the same as `read_float32_array`.
    """
    with open(filename, mode='rb') as f:
        compressed = f.read()
    decompressed = blosc.decompress(compressed)
    return bytes_to_array(decompressed, dtype=dtype)


def save_array_compressed(filename, arr: np.ndarray):
    """
    See `read_array`.
    """
    encoded = array_to_bytes(arr)
    compressed = blosc.compress(encoded, arr.dtype.itemsize, clevel=7, shuffle=True, cname='lz4hc')
    with open(filename, mode='wb') as f:
        f.write(compressed)
    log.info('Saved {}'.format(filename))


def array_to_bytes(arr: np.ndarray):
    """
    Dumps a numpy array into a raw byte string.
    :param arr: A numpy array.
    :return: A `bytes` string.
    """
    shape = arr.shape
    ndim = arr.ndim
    ret = struct.pack('i', ndim) + struct.pack('i' * ndim, *shape) + arr.tobytes(order='C')
    return ret


def bytes_to_array(s: bytes, dtype=np.float32):
    """
    Unpacks a byte string into a numpy array.
    :param s: A byte string containing raw array data.
    :param dtype: Data type.
    :return: A numpy array.
    """
    dims = struct.unpack('i', s[:4])[0]
    assert 0 <= dims < 1000  # Sanity check.
    shape = struct.unpack('i' * dims, s[4:4 * dims + 4])
    for dim in shape:
        assert dim > 0
    ret = np.frombuffer(s[4 * dims + 4:], dtype=dtype)
    assert ret.size == np.prod(shape), (ret.size, shape)
    ret.shape = shape
    return ret.copy()       


def processNYU_tensor(X):    
    X = X[:,:,45:471,41:601]
    return X
    

def cropPBRS(X):    
    if len(X.shape)==3: return X[45:471,41:601,:]
    else: return X[45:471,41:601]    
    