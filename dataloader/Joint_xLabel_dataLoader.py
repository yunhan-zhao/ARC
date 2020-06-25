import os, sys, random, time, copy
from skimage import io, transform
import numpy as np
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

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.bin'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class Joint_xLabel_train_dataLoader(Dataset):
    def __init__(self, real_root_dir, syn_root_dir, size=[240, 320], rgb=True, downsampleDepthFactor=1, paired_data=False):
        self.real_root_dir = real_root_dir
        self.syn_root_dir = syn_root_dir
        self.size = size
        self.rgb = rgb
        self.current_set_len = 0
        self.real_path2files = []
        self.syn_path2files = []
        self.downsampleDepthFactor = downsampleDepthFactor
        self.NYU_MIN_DEPTH_CLIP = 0.0
        self.NYU_MAX_DEPTH_CLIP = 10.0
        self.paired_data = paired_data # whether 1 to 1 matching
        self.augment = None # whether to augment each batch data
        self.x_labels = False # whether to collect extra labels in synthetic data, such as segmentation or instance boundaries

        self.set_name = 'train' # Joint_xLabel_train_dataLoader is only used in training phase
        
        real_curfilenamelist = os.listdir(os.path.join(self.real_root_dir, self.set_name, 'rgb'))
        for fname in sorted(real_curfilenamelist):
            if is_image_file(fname):
                path = os.path.join(self.real_root_dir, self.set_name, 'rgb', fname)
                self.real_path2files.append(path)

        self.real_set_len = len(self.real_path2files)

        syn_curfilenamelist = os.listdir(os.path.join(self.syn_root_dir, self.set_name, 'rgb'))
        for fname in sorted(syn_curfilenamelist):
            if is_image_file(fname):
                path = os.path.join(self.syn_root_dir, self.set_name, 'rgb', fname)
                self.syn_path2files.append(path)

        self.syn_set_len = len(self.syn_path2files)

        self.TF2tensor = transforms.ToTensor()
        self.TF2PIL = transforms.ToPILImage()
        self.TFNormalize = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        self.funcResizeTensor = nn.Upsample(size=self.size, mode='nearest', align_corners=None)
        self.funcResizeDepth = nn.Upsample(size=[int(self.size[0]*self.downsampleDepthFactor),
                                                 int(self.size[1]*self.downsampleDepthFactor)], 
                                                 mode='nearest', align_corners=None)
        
    def __len__(self):
        # looping over real dataset
        return self.real_set_len
    
    def __getitem__(self, idx):
        real_filename = self.real_path2files[idx % self.real_set_len]
        rand_idx = random.randint(0, self.syn_set_len - 1)
        if self.paired_data:
            assert self.real_set_len == self.syn_set_len
            syn_filename = self.syn_path2files[idx]

        else:
            syn_filename = self.syn_path2files[rand_idx]

        if np.random.random(1) > 0.5:
            self.augment = True
        else:
            self.augment = False

        real_img, real_depth = self.fetch_img_depth(real_filename)
        syn_img, syn_depth = self.fetch_img_depth(syn_filename)
        return_dict = {'real': [real_img, real_depth], 'syn': [syn_img, syn_depth]}

        if self.x_labels:
            # not really used in this project
            extra_label_list = self.fetch_syn_extra_labels(syn_filename)
            return_dict = {'real': [real_img, real_depth], 'syn': [syn_img, syn_depth], 'syn_extra_labels': extra_label_list}
        return return_dict

    def fetch_img_depth(self, filename):
        image = PIL.Image.open(filename)
        image = np.array(image, dtype=np.float32) / 255.
        
        if self.set_name == 'train':
            depthname = filename.replace('rgb','depth_inpainted').replace('png','bin')
        else:
            # use real depth for validation and testing
            depthname = filename.replace('rgb','depth').replace('png','bin')

        depth = read_array_compressed(depthname)
        
        if self.set_name=='train' and self.augment:
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
        
        if "nyu" in filename:
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

    def fetch_syn_extra_labels(self, filename):
        # currently only fetch segmentation labels and instance boundaries
        seg_name = filename.replace('rgb','semantic_seg')
        ib_name = filename.replace('rgb','instance_boundary')

        seg_np = np.array(PIL.Image.open(seg_name), dtype=np.float32)
        ib_np = np.array(PIL.Image.open(ib_name), dtype=np.float32)

        if self.set_name=='train' and self.augment:
            seg_np = np.fliplr(seg_np).copy()
            ib_np = np.fliplr(ib_np).copy()

        seg_np = np.expand_dims(seg_np, 2)
        seg_tensor = self.TF2tensor(seg_np)

        ib_np = np.expand_dims(ib_np, 2)
        ib_tensor = self.TF2tensor(ib_np) # size [1, 240, 320]

        return [seg_tensor, ib_tensor]
        
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
    