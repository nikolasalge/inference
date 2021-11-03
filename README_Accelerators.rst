MLPerf on Goolge Coral and Intel NCS2
=====================================

Model and Dataset
-----------------
* Model:
    * MobilnetV1
    * The model has to be compiled for the specific accelerator (for more details see Prerequisisites below)
    * Model format for Coral: .tflite (compiled for Edge TPU)
    * Model format for NCS2: .xml and .bin
    * Get model from `here`__, e.g. `MobilenetV1 quantized`__
* Dataset:
    * Imagenet validation: you have to register on `image-net.org`__ and download ILSVRC2012 Validation Images (6.3GB)
    * You need the val_map: get val_map.txt via `CK`__

__ https://github.com/nikolasalge/inference/tree/develop/nikolas/vision/classification_and_detection#supported-models
__ https://zenodo.org/record/2269307/files/mobilenet_v1_1.0_224_quant.tgz
__ https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
__ https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection#using-collective-knowledge-ck

Prerequisites Coral
-------------------
* Install Edge TPU runtime: `Coral Doc`__

Note: Frequency mode is chosen by installing the respective runtime, to change the mode simply install the other runtime (you can't have both installed at the same time).

* Instead of compiling models yourself, Coral-ready models can be downloaded from `coral.ai`__, e.g. `MobileNetV1 compiled for Edge TPU`__
* To use Tensorflow models, they first have to be converted to 8-bit tflite models and then have to be compiled with the `Edge TPU Compiler`__
* Details on the model conversion process: `Tensorflow models on the Edge TPU`__

__ https://coral.ai/docs/accelerator/get-started
__ https://coral.ai/models/image-classification/
__ https://github.com/google-coral/test_data/raw/master/mobilenet_v1_1.0_224_quant_edgetpu.tflite
__ https://coral.ai/docs/edgetpu/compiler/
__ https://coral.ai/docs/edgetpu/models-intro/

Prerequisites NCS 2
-------------------
* Install OpenVINO
* Prepare model with Model Optimizer

Prepare MLPerf
--------------
* Repository: `MLPerf Inference`__
* Clone the repository with with recurse submodules argument
* :code:`git clone --recurse-submodules https://github.com/nikolasalge/inference/tree/develop/nikolas`
* Install loadgen: `README_BUILD`__

__ https://github.com/nikolasalge/inference/tree/develop/nikolas
__ https://github.com/nikolasalge/inference/blob/develop/nikolas/loadgen/README_BUILD.md#git-submodules-approach

How to run MLPerf
-----------------
Change to :code:`vision/classification_and_detection`, then run :code:`python/main.py` with the following arguments:

* General Arguments
    * :code:`--model [path to model file]`
    * :code:`--dataset-path [path to dataset folder containing images and val_map.txt]`
    * :code:`--profile [mobilenet_coral, mobilenet_ncs2]`
        sets default settings
    * :code:`--max-batchsize [N]` 
        set this parameter to the batch size of the model. Coral only supports a batch size of 1, the NCS2 up to 128 (refers to N in NHWC, you have to hand this parameter to the Model Optimizer when compiling the model)

* LoadGen Arguments    
    * :code:`--scenario [SingleStream, MultiStream, Server, Offline]`
        sets loadgen scenario, for more info see below (the Coral and NCS2 profiles use MultitSream mode by default)
    * :code:`--samples-per-query [no. of samples]`
        only used in MultiStream Mode, sets number of samples that are sent each query, set to model-batchsize (max-batchsize*n, where n is a positive integer, works too)
    * :code:`--accuracy` 
        use LoadGen AccuracyOnly mode instead of PerformanceOnly mode
    * :code:`--count [number of images to use]` 
        not MLPerf compliant for AccuracyOnly mode, use for P´erformance mode or for testing

* `further arguments`__

**example:**

.. code-block:: console

    python python/main.py \
    --model ~/models/mobilenet_v1_1.0_224_quant_edgetpu.tflite \
    --dataset-path ~/dataset/ILSVRC2012_img_val \
    --profile mobilenet_coral \
    --max-batchsize 1 \
    --scenario MultiStream --samples-per-query 4 \
    --accuracy --count 100 
    

__ https://github.com/nikolasalge/inference/tree/develop/nikolas/vision/classification_and_detection#usage

More information on LoadGen Scenarios
-------------------------------------
There are 4 Scenarios:

* **SingleStream**: Queries with 1 Image each are sent sequentially
* **MultiStream**: Each Query has N samples (see :code:`--samples-per-query` parameter); Queries are sent every 50ms (for Object Detection)
* **Server**: Query arrival is random
* **Offline**: A single Query with all Samples is sent

for more information see Chapter III - Section C in `MLPerf Inference Benchmark`__

__ https://arxiv.org/pdf/1911.02549.pdf

Changes made to ensure compatibility of Accelerators
----------------------------------------------------

::

   inference in branch: develop/nikolas
   └── vision/classification_and_detection/python
      ├── main.py
      ├── backend_openvino_ncs2.py
      ├── backend_tflite_coral.py
      └── dataset.py

* **main.py**: run the benchmark with this file like the example shown above
* **backend_openvino_ncs2.py**: new backend for NCS2 compatibility
* **backend_tflite_coral.py**: new backend for Coral compatibility
* **dataset.py**: added preprocessing methods for int8 (Coral) and float16 (NCS2)
    
