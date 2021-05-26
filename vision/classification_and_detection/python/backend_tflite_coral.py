"""
tflite backend for coral

NA:
backend_tflite.py was used as a template

NOTES:
* to change mode of coral (max and std frequency):
    * to change between std and max frequency, just install the other runtime:
    * "sudo apt-get install libedgetpu1-std" or
    * "sudo apt-get install libedgetpu1-max"
"""

from threading import Lock

import tflite_runtime
import tflite_runtime.interpreter as tflite

import backend

__author__ = 'Nikolas Alge'

class BackendTfliteCoral(backend.Backend):
    """Backend Class to run predictions on Google Coral with TensorFlow Lite.    
    """
    
    def __init__(self):
        """Constructor method
        """
        super(BackendTfliteCoral, self).__init__()
        self.sess = None
        self.lock = Lock()
        
    def version(self):
        """Returns the version of the backend

        :return: version of the backend
        :rtype: string
        """
        return tflite_runtime.__version__ + "/" + tflite_runtime.__git_version__

    def name(self):
        """Returns the name of the backend

        :return: name of the backend
        :rtype: string
        """
        return "tflite_coral"

    def image_format(self):
        """Returns the image format of the backend

        :return: image format of the backend
        :rtype: string
        """
        # tflite is always NHWC
        return "NHWC"

    def load(self, model_path, inputs=None, outputs=None):
        """Loads the Model to Backend and sets in- and outputs
        
        :param model_path: path to model compiled for edgetpu
        :type model_path: string
            
        :param inputs: inputs of the model, defaults to None
        :type inputs: list, optional
            
        :param outputs: outputs of the model, defaults to None
        :type outputs: list, optional       
        
        :return: self
        :rtype: instance of BackendTfliteCoral
        
        .. todo::
            use inputs/outputs if given
        """
        self.sess = tflite.Interpreter(
                model_path=model_path,
                experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
        self.sess.allocate_tensors()
        
        # keep input/output name to index mapping
        self.input2index = {i["name"]: i["index"] for i in self.sess.get_input_details()}
        self.output2index = {i["name"]: i["index"] for i in self.sess.get_output_details()}
       
        # keep input/output names
        self.inputs = list(self.input2index.keys())
        self.outputs = list(self.output2index.keys())
        return self

    def predict(self, feed):
        """Inference with TensorFlow Lite on Google Coral.
        Runs the input image through the model to get the ouput tensor.
        
        :param feed: inputs with respective image data
        :type feed: dict of {inputs: imagedata in cache as ndarray}
        
        :return: results of prediction
        :rtype: list with ouput ndarray and type
        """
        self.lock.acquire()
        
        # set inputs
        for k, v in self.input2index.items():
            self.sess.set_tensor(v, feed[k])
        self.sess.invoke()
        
        # get results
        res = [self.sess.get_tensor(v) for _, v in self.output2index.items()]
        self.lock.release()
        return res
