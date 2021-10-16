# Inference Optimization Techniques for Neural Networks on Embedded Platforms: An Experimental Evaluation
This is a benchmark for inference optimization methods for CNNs. On this repo 
you will find scripts for the benchmark itself and for the implementation on
TensorFlow of 3 optimization methods:
-  Tai, Xiao, Zhang, et al., *“Convolutional neural networks with low-rank regularization”* [link](https://arxiv.org/pdf/1511.06067)
-  Astrid & Lee, *“CP-decomposition with tensor power method for convolutional neural networks compression”* [link](https://arxiv.org/pdf/1701.07148)
-  Liu, Li, Shen, et al., *“Learning efficient convolutional networks through network slimming”* [link](https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)

Requirements:
- Python 3.5.2+
- Tensorflow 2.2.0+
- Numpy 1.16.4+
- Jupyter 1.0.0+
- Torchvision(CPU) 0.9.2+ (*only needed for building VGG16-BN*)

Script usage:
```
> python imagenet_benchmark.py --help
usage: imagenet_benchmark.py [-h] [--cpu] [--acc] [--mem] [--pow_serv POW_SERV] [--input INPUT] [-n N] [-s S]
                             mod_file b_sz

positional arguments:
  mod_file             model file
  b_sz                 batch size

optional arguments:
  -h, --help           show this help message and exit
  --cpu                force CPU usage
  --acc                measure accuracy
  --mem                measure RAM usage
  --pow_serv POW_SERV  power measuring server IP
  --input INPUT        input size
  -n N                 number of trials
  -s S                 measurement sample size
```

For training and accuracy measurements, make sure to have the ImageNet (ILSVRC) dataset on your PC. You can check [Academic torrents](https://academictorrents.com/) to find it.

*Note: For measuring speed or power, try using a nº of trials (-n) of 2 to 8, as several GPUs need some warm up passes to fully activate their full-performance mode.*

