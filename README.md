# Neural Network Compression Framework for TensorFlow (NNCF TF)

This repository contains a TensorFlow*-based framework and samples for neural networks compression.

The framework is the implementaiton of the [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf_pytorch) for TensorFlow\*.

The framework is organized as a Python\* package that can be built and used in a standalone mode.
The framework architecture is unified to make it easy to add different compression methods.
 
The samples demonstrate the usage of compression algorithms for two different use cases on public models and datasets: Image Classification, Object Detection.
[Compression results](#nncf-tf-compression-results) achievable with the NNCF-powered samples can be found in a table at the end of this document.

## Key Features

- Support of various compression algorithms, applied during a model fine-tuning process to achieve best compression parameters and accuracy:
    - Quantization
    - Sparsity
- Automatic, configurable model graph transformation to obtain the compressed model. The model is wrapped by the custom class and additional compression-specific layers are inserted in the graph.
  > **NOTE**: Only Keras models created using Sequential or Keras Functional API are supported.
- Common interface for compression methods.
- Distributed training support.
- Configuration file examples for each supported compression algorithm.
- Exporting compressed models to Frozen Graph or TensorFlow\* SavedModel ready for usage with [OpenVINO&trade; toolkit](https://github.com/openvinotoolkit/).

## Usage
The NNCF TF is organized as a regular Python package that can be imported in an arbitrary training script.
The basic workflow is loading a JSON configuration script containing NNCF-specific parameters determining the compression to be applied to your model, and then passing your model along with the configuration script to the `nncf.create_compressed_model` function.
This function returns a transformed model ready for compression fine-tuning, and handle to the object allowing you to control the compression during the training process:

```python
import nncf
from nncf import create_compressed_model
from nncf import Config as NNCFConfig

# Instantiate your uncompressed model
from tensorflow.keras.applications import ResNet50
model = ResNet50()

# Apply compression according to a loaded NNCF config
nncf_config = NNCFConfig.from_json("resnet50_imagenet_int8.json")
compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)

# Now use compressed_model as a usual Keras model

# ... the rest of the usual TensorFlow-powered training pipeline

# Export to Frozen Graph, TensorFlow SavedModel or .h5  when done fine-tuning 
compression_ctrl.export_model("compressed_model.pb", save_format='frozen_graph')
```

## Model Compression Samples

For a quick start with NNCF-powered compression, you can also try the sample scripts, each of them provides a basic training pipeline for Image Classification and Object Detection correspondingly.

To run the samples please refer to the corresponding tutorials:
- [Image Classification sample](examples/classification/README.md)
- [Object Detection sample](examples/object_detection/README.md)

## System requirements
- Ubuntu\* 16.04 or later (64-bit)
- Python\* 3.6 or later
- NVidia CUDA\* Toolkit 10.1 or later
- TensorFlow\* 2.2.0 or later

## Installation
We suggest to install or use the package in the [Python virtual environment](https://docs.python.org/3/tutorial/venv.html).


#### As a package built from a checked-out repository:
1) Install the following system dependencies:
    ```
    sudo apt-get install python3-dev
    ```

2) Install the package and its dependencies by running the following in the repository root directory:
    ```
    python setup.py install
    ```

_NB_: For launching example scripts in this repository, we recommend replacing the `install` option above with `develop` and setting the `PYTHONPATH` variable to the root of the checked-out repository.

## Contributing
Refer to the [Contribution Guidelines](./CONTRIBUTING.md) for more information on contributions to the NNCF TF repository.

## NNCF TF compression results

Achieved using sample scripts and NNCF TF configuration files provided with this repository. See README files for [sample scripts](#model-compression-samples) for links to exact configuration files and pre-trained models.

Quick jump to the samples:
- [Classification](#classification)
- [Object Detection](#object-detection)

#### Classification

|**Model**|**Compression algorithm**|**Dataset**|**TensorFlow FP32 baseline**|**TensorFlow compressed accuracy**|
| :---: | :---: | :---: | :---: | :---: |
|Inception V3|INT8 w:sym,per-tensor a:sym,per-tensor |ImageNet|77.9|78.27|
|Inception V3|Sparsity 54% (Magnitude)|ImageNet|77.9|77.87|
|MobileNet V2|INT8 w:sym,per-tensor a:sym,per-tensor |ImageNet|71.85|71.81|
|MobileNet V2|Sparsity 35% (Magnitude)|ImageNet|71.85|72.36|

#### Object detection

|**Model**|**Compression algorithm**|**Dataset**|**TensorFlow FP32 baseline**|**TensorFlow compressed accuracy**|
| :---: | :---: | :---: | :---: | :---: |
|RetinaNet|INT8 w:sym,per-tensor a:sym,per-tensor |COCO2017|-|-|
|RetinaNet|Sparsity 50% (Magnitude)|COCO2017|-|-|

## Legal Information
[*] Other names and brands may be claimed as the property of others.
