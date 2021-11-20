"""
dataset related classes and methods

NA: Changes made by Nikolas Alge are tagged with "NA:" 
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import sys
import time

import cv2
import numpy as np


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dataset")

class Item():
    def __init__(self, label, img, idx):
        self.label = label
        self.img = img
        self.idx = idx
        self.start = time.time()


def usleep(sec):
    if sys.platform == 'win32':
        # on windows time.sleep() doesn't work to well
        import ctypes
        kernel32 = ctypes.windll.kernel32
        timer = kernel32.CreateWaitableTimerA(ctypes.c_void_p(), True, ctypes.c_void_p())
        delay = ctypes.c_longlong(int(-1 * (10 * 1000000 * sec)))
        kernel32.SetWaitableTimer(timer, ctypes.byref(delay), 0, ctypes.c_void_p(), ctypes.c_void_p(), False)
        kernel32.WaitForSingleObject(timer, 0xffffffff)
    else:
        time.sleep(sec)


class Dataset():
    def __init__(self):
        self.arrival = None
        self.image_list = []
        self.label_list = []
        self.image_list_inmemory = {}
        self.last_loaded = -1

    def preprocess(self, use_cache=True):
        raise NotImplementedError("Dataset:preprocess")

    def get_item_count(self):
        return len(self.image_list)

    def get_list(self):
        raise NotImplementedError("Dataset:get_list")

    def load_query_samples(self, sample_list):
        self.image_list_inmemory = {}
        for sample in sample_list:
            self.image_list_inmemory[sample], _ = self.get_item(sample)
        self.last_loaded = time.time()

    def unload_query_samples(self, sample_list):
        if sample_list:
            for sample in sample_list:
                if sample in self.image_list_inmemory :
                    del self.image_list_inmemory[sample]
        else:
            self.image_list_inmemory = {}

    def get_samples(self, id_list):
        data = np.array([self.image_list_inmemory[id] for id in id_list])
        return data, self.label_list[id_list]

    def get_item_loc(self, id):
        raise NotImplementedError("Dataset:get_item_loc")


#
# Post processing
#
class PostProcessCommon:
    def __init__(self, offset=0):
        self.offset = offset
        self.good = 0
        self.total = 0

    def __call__(self, results, ids, expected=None, result_dict=None):
        processed_results = []
        n = len(results[0])
        for idx in range(0, n):
            result = results[0][idx] + self.offset
            processed_results.append([result])
            if result == expected[idx]:
                self.good += 1
        self.total += n
        return processed_results

    def add_results(self, results):
        pass

    def start(self):
        self.good = 0
        self.total = 0

    def finalize(self, results, ds=False,  output_dir=None):
        results["good"] = self.good
        results["total"] = self.total


class PostProcessArgMax:
    """NA: This class is used to postprocess the results for models, where all 
    probabilities are in the output tensor.
    """
    def __init__(self, offset=0):
        self.offset = offset
        self.good = 0
        self.total = 0

    def __call__(self, results, ids, expected=None, result_dict=None):
        """NA: Return Post-Processed output Tensors and check if the prediction
        is correct.         
        
        Takes the output Tensor, which is of shape [N,1001]. The argmax 
        function then looks for the biggest probability and returns the index,
        and creates a Tensor with shape [N,1] and format [N,I]. 
        N is the batch size, I the index, where the probability is the highest.
        I is of range [0,1000].    
        
        :param results: 
        :type results: 
        
        :return: 
        :rtype: 
        """
        processed_results = []
        results = np.argmax(results[0], axis=1)
        n = results.shape[0]
        for idx in range(0, n):
            result = results[idx] + self.offset
            #print("NA: OUTPUT with max prob: " + str(result)) #debug
            #print("NA: EXPECTED categorie: " + str(expected[idx])) #debug
            processed_results.append([result])
            if result == expected[idx]:
                self.good += 1
        self.total += n
        return processed_results

    def add_results(self, results):
        pass

    def start(self):
        self.good = 0
        self.total = 0

    def finalize(self, results, ds=False, output_dir=None):
        results["good"] = self.good
        results["total"] = self.total


#
# pre-processing
#

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img


def pre_process_vgg(img, dims=None, need_transpose=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output_height, output_width, _ = dims
    cv2_interpol = cv2.INTER_AREA
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2_interpol)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')

    # normalize image
    means = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    img -= means

    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return img


def pre_process_mobilenet(img, dims=None, need_transpose=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')

    img /= 255.0
    img -= 0.5
    img *= 2

    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return img


#NA: add quantized preprocessing for coral
def pre_process_mobilenet_int8(img, dims=None, need_transpose=False):
    """Preprocesses Image for a quantized Mobilenet
    
    :param img: image that is loaded with cv2
    :type img: Mat, return of cv.imread()
    
    :param dims: format you want the img to be converted to in HWC, 
        defaults to None
    :type dims: list, e.g. [224, 224, 3], optional
    
    :param need_transpose: specifies, if the image needs to be transposed,
        defaults to False
    :type need_transpose: bool, optional
    
    :return: processed image
    :rtype: ndarray
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output_height, output_width, _ = dims #[244, 244, 3]
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='uint8')

    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return img

#NA add preproc for yolo quantized 416
def pre_process_yolov3_int8(img, dims=None, need_transpose=False):
    """
    TODO: CHANGE FOR YOLO
    Preprocesses Image for a quantized Mobilenet
    
    :param img: image that is loaded with cv2
    :type img: Mat, return of cv.imread()
    
    :param dims: format you want the img to be converted to in HWC, 
        defaults to None
    :type dims: list, e.g. [224, 224, 3], optional
    
    :param need_transpose: specifies, if the image needs to be transposed,
        defaults to False
    :type need_transpose: bool, optional
    
    :return: processed image
    :rtype: ndarray
    """
    img = maybe_resize(img, dims)
    img = np.asarray(img, dtype=np.uint8)
    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return img
    
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#    output_height, output_width, _ = dims #[416, 416, 3]
#    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
#    img = center_crop(img, output_height, output_width)
#    img = np.asarray(img, dtype='uint8')
#
#    # transpose if needed
#    if need_transpose:
#        img = img.transpose([2, 0, 1])
#    return img

#NA: add float16 preprocessing for ncs2
def pre_process_mobilenet_fp16(img, dims=None, need_transpose=False):
    """Preprocessing of Image for a FP16 Mobilenet
    
    :param img: image that is loaded with cv2
    :type img: Mat, return of cv.imread()
    
    :param dims: format you want the img to be converted to in HWC, 
        defaults to None
    :type dims: list, e.g. [224, 224, 3], optional
    
    :param need_transpose: specifies, if the image needs to be transposed,
        defaults to False
    :type need_transpose: bool, optional
    
    :return: processed image
    :rtype: ndarray
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    #return img as ndarray
    img = np.asarray(img, dtype='float16')
        
    # convert form [0,255] to [-1,1]
    img /= 255.0
    img -= 0.5
    img *= 2

    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return img


def pre_process_mobilenet_fp16_ov(img, dims=None, need_transpose=False):
    """Preprocessing of Image for a FP16 Mobilenet
    Works with Converted Models of OpenVINO.
    
    :param img: image that is loaded with cv2
    :type img: Mat, return of cv.imread()
    
    :param dims: format you want the img to be converted to in HWC, 
        defaults to None
    :type dims: list, e.g. [224, 224, 3], optional
    
    :param need_transpose: specifies, if the image needs to be transposed,
        defaults to False
    :type need_transpose: bool, optional
    
    :return: processed image
    :rtype: ndarray
    """
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output_height, output_width, _ = dims
#    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
#    img = center_crop(img, output_height, output_width)
    img = cv2.resize(img, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
    #return img as ndarray
    img = np.asarray(img, dtype='float16')
        
    # convert form [0,255] to [-1,1]
#    img /= 255.0
#    img -= 0.5
#    img *= 2

    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return img


def pre_process_imagenet_pytorch(img, dims=None, need_transpose=False):
    from PIL import Image
    import torchvision.transforms.functional as F

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = F.resize(img, 256, Image.BILINEAR)
    img = F.center_crop(img, 224)
    img = F.to_tensor(img)
    img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    if not need_transpose:
        img = img.permute(1, 2, 0) # NHWC
    img = np.asarray(img, dtype='float32')
    return img


def maybe_resize(img, dims):
    img = np.array(img, dtype=np.float32)
    if len(img.shape) < 3 or img.shape[2] != 3:
        # some images might be grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if dims != None:
        im_height, im_width, _ = dims
        img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
    return img


def pre_process_coco_mobilenet(img, dims=None, need_transpose=False):
    img = maybe_resize(img, dims)
    img = np.asarray(img, dtype=np.uint8)
    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return img


def pre_process_coco_pt_mobilenet(img, dims=None, need_transpose=False):
    img = maybe_resize(img, dims)
    img -= 127.5
    img /= 127.5
    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return img


def pre_process_coco_resnet34(img, dims=None, need_transpose=False):
    img = maybe_resize(img, dims)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = img / 255. - mean
    img = img / std

    if need_transpose:
        img = img.transpose([2, 0, 1])

    return img


def pre_process_coco_resnet34_tf(img, dims=None, need_transpose=False):
    img = maybe_resize(img, dims)
    mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    img = img - mean
    if need_transpose:
        img = img.transpose([2, 0, 1])

    return img
