MLPerf on Goolge Coral and Intel NCS2
=====================================

Prerequisites Coral
-------------------
* Install Edge TPU runtime: `Coral Doc`__

Note: Frequency mode is chosen by installing the respective runtime, to change the mode simply install the other runtime (you can't have both installed at the same time).

* Models for Coral can either be downloaded from `coral.ai`__ or you can use your own
* To use Tensorflow models, they first have to be converted to 8-bit tflite models and then have to be compiled with the `Edge TPU Compiler`__
* Details on the model conversion process: `Tensorflow models on the Edge TPU`__

__ https://coral.ai/docs/accelerator/get-started
__ https://coral.ai/models/
__ https://coral.ai/docs/edgetpu/compiler/
__ https://coral.ai/docs/edgetpu/models-intro/

Prerequisites NCS 2
-------------------
* Openvino
* Prepare model with Model Optimizer

Prepare MLPerf
--------------
* Repository: `MLPerf Inference`__
* Clone the repository with with recurse submodules argument
* :code:`git clone --recurse-submodules https://github.com/nikolasalge/inference/tree/develop/nikolas`
* Install loadgen: `README_BUILD`__

__ https://github.com/nikolasalge/inference/tree/develop/nikolas
__ https://github.com/nikolasalge/inference/blob/develop/nikolas/loadgen/README_BUILD.md#git-submodules-approach

Model and Dataset
-----------------
* Model:
    * Mobilnetv1 (others may work too)
    * .xml and .bin for NCS2
    * .tflite (compiled for edgetpu) for Coral
    * The number of samples N (as in NHWC) you compile the model with has to be the same as the :code:`--samples-per-query` parameter given to MLPerf (see parameter description below)
* Dataset:
    * Imagenet validation: you have to register on `image-net.org`__ and download ILSVRC2012 Validation Images (6.3GB)
    * You need the val_map: get val_map.txt via `CK`__

__ https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
__ https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection#using-collective-knowledge-ck

How to run MLPerf
-----------------
Change to :code:`vision/classification_and_detection`, then run :code:`python/main.py` with the following arguments:

* :code:`--profile [mobilenet_coral, mobilenet_ncs2]`
* :code:`--model [path to model file]`
* :code:`--dataset-path [path to dataset folder containing images and val_map]`
* :code:`--accuracy` use loadgen accuracy mode instead of performance mode
* :code:`--count [number of images to use]` not mlperf compliant for accuracy mode, use for performance mode or for testing
* :code:`--samples-per-query [no. of samples]` mlperf multi-stream sample per query (the coral and ncs2 profiles use multistream mode by default), refers to N in NHWC, the number of samples which the model was compiled with
* `further arguments`__

__ https://github.com/nikolasalge/inference/tree/develop/nikolas/vision/classification_and_detection#usage

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
    
