MLPerf on Coral and NCS2
========================

What you need
-------------
* this repository
* install loadgen
* get model:
    * mobilnetv1
    * .xml and .bin for ncs2
    * .tflite (compiled for edgetpu) for coral
* get dataset:
    * imagenet validation
    * val_map.txt

How to run
----------
run :code:`vision/classification_and_detection/python/main.py` with the following arguments:

* :code:`--model [path to model file]`
* :code:`--dataset-path [path to dataset folder containing pictures and val_map]`
* :code:`--profile [mobilenet_coral, mobilenet_ncs2]`
* :code:`--accuracy` use loadgen accuracy mode instead of performance mode
* :code:`--count [number of images to use]` not mlperf compliant, for testing
* further arguments are specified in vision/classification_and_detection/REAMDME.md

**example for coral:**

.. code-block:: console

    python vision/classification_and_detection/python/main.py \
    --model ~/models/mobilenet_v1_1.0_224_quant_edgetpu.tflite \
    --dataset-path ~/dataset/ILSVRC2012_img_val \
    --profile mobilenet_coral \
    --accuracy --count 100

Important Files
---------------

::

   inference in develop/nikolas branch
   └── vision/classification_and_detection/python
      ├── main.py
      ├── backend_openvino_ncs2.py
      ├── backend_tflite_coral.py
      └── dataset.py
        
* **main.py**: run the benchmarkt with this file like the example shown above
* **backend_openvino_ncs2.py**: new backend for NCS2 compatibility
* **backend_tflite_coral.py**: new backend for Coral compatibility
* **dataset.py**: added preprocessing methods for int8 and float16
    
