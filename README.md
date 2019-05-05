# ESync

This is a MXNet implementation of the ESync algorithm described in the paper "ESync: An Efficient Synchronous Parallel Algorithm for Distributed ML in Heterogeneous Clusters". 

ESync is an efficient synchronous parallel algorithm designed for distributed machine learning tasks in heterogeneous clusters (the clusters may consist of computing devices with different computing capabilities, e.g. CPU, GPU, TPU, FPGA), which:

* takes both the accuracy of [SSGD](https://arxiv.org/pdf/1604.00981.pdf) and training speed of [ASGD](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf);
* takes full advantage of the computing capabilities of the heterogeneous clusters with lowest traffic load. 
* allows the aggregation operations to be performed in a synchronous manner in heterogeneous clusters, and provides users with flexibility in selecting different collective communication algorithms according to the characteristics of tasks and network (e.g. [Parameter Server](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf), [Ring Allreduce](http://research.baidu.com/bringing-hpc-techniques-deep-learning/), [Butterfly](https://link.springer.com/content/pdf/10.1007%2F978-3-540-24685-5_1.pdf), [Binary Blocks](https://link.springer.com/content/pdf/10.1007%2F978-3-540-24685-5_1.pdf)).

# Usage

## Prerequisites

* python == 3.6
* mxnet == 1.4.0 
* numpy == 1.16.2
* argparse == 1.4.0
* matplotlib == 3.0.3

> Note: MXNet should be compiled with the build flag **USE\_DIST\_KVSTORE=1** to support distributed training. See [Distributed Training in MXNet](https://mxnet.incubator.apache.org/versions/master/faq/distributed_training.html) for more details.

## Hyper Parameters

Parameter Name | flag | type | Default Value |  Description 
:-:|:-:|:-:|:-:|:--
**learning\_rate** | -l | float | 0.001 |  Set **learning\_rate** when **mode** is *sync*, *async* or *local*. This parameter is used in the optimizer (e.g. SSGD, ASGD) to scale the gradient.
**local\_lr** | -ll | float | 0.001 | Set **local\_lr** when **mode** is *esync*. This parameter is used in the local optimizer (e.g. SGD, Momentum, Adam) to scale the gradient.
**global\_lr** | -gl | float | 1.0 | Set **global\_lr** when **mode** is *esync*. This parameter is used in the global aggregation operation to scale the aggregated model updates and can be simply set to 1.0.
**batch\_size** | -b | int | 64 | The number of samples processed in an iteration on each device.
**data\_dir**| -dd | string | /home/lizh/ESync/data | Path to the data files. Include a folder named *fashion-mnist*, which contains *t10k-images-idx3-ubyte.gz*, *t10k-labels-idx1-ubyte.gz*, *train-images-idx3-ubyte.gz*, *train-labels-idx1-ubyte.gz*. The Fashion-MNIST dataset is available on [Github](https://github.com/zalandoresearch/fashion-mnist).
**gpu** | -g | int | 0 | The ID of GPU used for training. We default to using only one GPU for each process in the current version, i.e. only one integer is allowed.
**cpu** | -c | bool | *False* | Default to training on GPU 0 (set by the option **gpu**), set **cpu** to *True* to support training on CPUs.
**network** | -n | string | *resnet18-v1* | The network used to evaluate the performance of *esync*, *sync* and *async*. We support [*alexnet*, *resnet18-v1*, *resnet50-v1*, *resnet50-v2*, *mobilenet-v1*, *mobilenet-v2*, *inception-v3*] in the current version.
**log\_dir** | -ld | string | /home/lizh/ESync/logs | Path to save the logs. The folder named *logs* will be created automatically at the specified path, and it will be emptied during initialization. The Measure module will create subfolders "{device\_name}{device\_id}" and save log files "iter-{iter\_num}.txt" in these subfolders.
**eval\_duration** | -e | int | 1 | Interval for model evaluation, default to evaluating the model in each communication round. We recommend evaluating the model on devices with strong computing capability.
**mode** | -m | string | *esync* | Support [*esync*, *sync*, *async*, *local*]. Set **mode** to *local* to train the model on single device.
**split\_by\_class** | -s | bool | *False* | Default to allocating datasets using the uniform random sampling. Set **split\_by\_class** to *True* to allocate specific classes of samples to each device, for example, allocate samples with labels 0\~4 to device 0 and samples with labels 5\~9 to device 1.
**state\_server\_ip** | -ip | string | 10.1.1.34 | The IP of State Server.
**state\_server\_port** | -port | string | 10010 | The port of State Server.

> Note: The default values can be modified through [config.py](https://github.com/Lizonghang/ESync/blob/master/config.py).

> Note: DO NOT run multiple processes that use the same device on each server, otherwise the log files will be overwritten.

# References

[1] [Chen, Jianmin, et al. "Revisiting distributed synchronous SGD." arXiv preprint arXiv:1604.00981 (2016).](https://arxiv.org/pdf/1604.00981.pdf)

[2] [Dean, Jeffrey, et al. "Large scale distributed deep networks." Advances in neural information processing systems. 2012.](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf)

[3] [Li, Mu, et al. "Scaling distributed machine learning with the parameter server." 11th {USENIX} Symposium on Operating Systems Design and Implementation ({OSDI} 14). 2014.](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf)

[4] [Gibiansky, Andrew. "Bringing HPC techniques to deep learning". http://research.baidu.com/bringing-hpc-techniques-deep-learning/, 2017.2.21.](http://research.baidu.com/bringing-hpc-techniques-deep-learning/)

[5] [Rabenseifner, Rolf. "Optimization of collective reduction operations." International Conference on Compu- tational Science, Krakow, Poland, 2004.6.6-6.9.](https://link.springer.com/content/pdf/10.1007%2F978-3-540-24685-5_1.pdf)
