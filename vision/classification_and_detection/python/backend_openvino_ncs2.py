"""
openvino backend for ncs2
"""

from threading import Lock
import time

import backend
from openvino.inference_engine import IECore

__author__ = 'Nikolas Alge'

class BackendOpenVinoNCS2(backend.Backend):
    """Backend Class to run predictions on Intel NCS2 with OpenVINO.    
    """
    
    def __init__(self):
        """Constructor method
        """
        super(BackendOpenVinoNCS2, self).__init__()
        self.sess = None
        self.lock = Lock()
        #NA:
        self.infer_count = -5
        self.timing = []

    def version(self):
        """Returns the version of the backend

        :return: version of the backend
        :rtype: string
        """
        ie = IECore()
        version = ie.get_versions("MYRIAD")["MYRIAD"]
        return (str(version.major) + "." + str(version.minor) + "/" + 
              str(version.build_number) + " [" + 
              str(version.description) + "]")

    def name(self):
        """Returns the name of the backend

        :return: name of the backend
        :rtype: string
        """
        return "openvino"

    def image_format(self):
        """Returns the image format of the backend

        :return: image format of the backend
        :rtype: string
        """
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        """Loads the Model to Backend and sets in- and outputs
        
        :param model_path: path to model .xml file
        :type model_path: string
            
        :param inputs: inputs of the model, defaults to None
        :type inputs: list, optional
            
        :param outputs: outputs of the model, defaults to None
        :type outputs: list, optional       
        
        :return: self
        :rtype: instance of BackendOpenVinoNCS2
        
        .. todo::
            * use inputs/outputs if given
            * use model optimizer directly
        """
        # load model (now from ir, later add model optimizer before and load tf model?)
        ie = IECore()
        self.model = ie.read_network(model=model_path) 
        
        self.inputs = [next(iter(self.model.input_info))] #['input']
        self.outputs = next(iter(self.model.outputs)) #MobilenetV1/Predictions/Reshape_1
               
        self.sess = ie.load_network(network=self.model, device_name="MYRIAD")  
        return self

    def predict(self, feed):
        """Inference with OpenVino on NCS2.
        Runs the input image through the model to get the ouput tensor.
        
        :param feed: inputs with respective image data
        :type feed: dict of {inputs: imagedata in cache as ndarray}
        
        :return: results of prediction
        :rtype: list with ouput ndarray and type
        """
        self.infer_count += 1
        self.lock.acquire()
        # inputs: dict of input to np array
        if self.infer_count > 0:
            self.timing.append(str(self.infer_count) + ";start;" + str(time.time()))
        res = self.sess.infer(inputs=feed)
        if self.infer_count > 0:
            self.timing.append(str(self.infer_count) + ";end;" + str(time.time())) #timing
        res = [res[self.outputs]]
#        print(type(res[0][0][0]))
#               
#        exit()
        self.lock.release()
        return res
