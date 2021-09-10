MLPerf on Coral and NCS2
========================

Prerequisites Coral
-------------------
* install Edge TPU runtime: `Coral Doc`__

Note: Frequency mode is chosen by installing the respective runtime, to change the mode simply install the other runtime (you can't have both installed at the same time).

* Models for Coral can either be downloaded from `Coral.ai`__ or compiled with the `Edge TPU compiler`__
* To use Tensorflow models, they first have to be converted to 8-bit Tensorflow lite models: `Tensorflow models on the Edge TPU`__

__ https://coral.ai/docs/accelerator/get-started
__ https://coral.ai/models/
__ https://coral.ai/docs/edgetpu/compiler/
__ https://coral.ai/docs/edgetpu/models-intro/

Prerequisites NCS 2
-------------------
* Openvino

Prepare MLPerf
--------------
* repository: `MLPerf Inference`__
* clone the repository with with recurse submodules argument
* :code:`git clone --recurse-submodules https://github.com/nikolasalge/inference/tree/develop/nikolas`
* install loadgen: `README_BUILD`__

__ https://github.com/nikolasalge/inference/tree/develop/nikolas
__ https://github.com/nikolasalge/inference/blob/develop/nikolas/loadgen/README_BUILD.md#git-submodules-approach

Model and Dataset
---------------------
* model:
    * mobilnetv1
    * .xml and .bin for ncs2
    * .tflite (compiled for edgetpu) for coral
* dataset:
    * imagenet validation: you have to login on `image-net.org`__ and download ILSVRC2012 Validation Images (6.3GB)
    * val_map.txt: get the val_map via `CK`__

__ https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
__ https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection#using-collective-knowledge-ck

How to run MLPerf
-----------------
change to :code:`vision/classification_and_detection`, then run :code:`python/main.py` with the following arguments:

* :code:`--model [path to model file]`
* :code:`--dataset-path [path to dataset folder containing images and val_map]`
* :code:`--profile [mobilenet_coral, mobilenet_ncs2]`
* :code:`--accuracy` use loadgen accuracy mode instead of performance mode
* :code:`--count [number of images to use]` not mlperf compliant for accuracy mode, use for performance mode or for testing
* :code:`--samples-per-query [No. of samples]` mlperf multi-stream sample per query (the coral and ncs2 profiles use multistream mode by default)
* `further arguments`__

__ https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection#usage

**example:**

.. code-block:: console

    python python/main.py \
    --model ~/models/mobilenet_v1_1.0_224_quant_edgetpu.tflite \
    --dataset-path ~/dataset/ILSVRC2012_img_val \
    --profile mobilenet_coral \
    --accuracy --count 100 --samples-per-query 4

Changes made to ensure compatibility of Accelerators
----------------------------------------------------

::

   inference in branch: develop/nikolas
   └── vision/classification_and_detection/python
      ├── main.py
      ├── backend_openvino_ncs2.py
      ├── backend_tflite_coral.py
      └── dataset.py

* **main.py**: run the benchmarkt with this file like the example shown above
* **backend_openvino_ncs2.py**: new backend for NCS2 compatibility
* **backend_tflite_coral.py**: new backend for Coral compatibility
* **dataset.py**: added preprocessing methods for int8 (coral) and float16 (ncs2)
    
