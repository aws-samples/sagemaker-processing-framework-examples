# sagemaker-processing-framework-examples

With commit [b3c8bb1c](https://bit.ly/3kf3PC1), SM Python SDK introduced the FrameworkProcessor to use natively framework DLC with a SageMaker Processing job. Here is a repo with some examples.

## Available examples

- [`huggingface-sample`](huggingface-sample): uses a HF Transformer model to enrich a dataset with sentiment analysis
- [`mxnet-sample`](mxnet-sample): uses MXNet GluonTS library to perform pre-processing of a time-series dataset
- [`pytorch-sample`](pytorch-sample): uses PyTorch Torch Vision library to extract features from images
- [`tensorflow2-sample`](tensorflow2-sample): uses a custom built Keras model that returns a prediction on the dataset