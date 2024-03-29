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
* django == 2.1.7

> Note: MXNet should be compiled with the build flag **USE\_DIST\_KVSTORE=1** to support distributed training. See [Distributed Training in MXNet](https://mxnet.incubator.apache.org/versions/master/faq/distributed_training.html) for more details.

## Hyper Parameters

Parameter Name | Flag | Type | Default Value |  Description 
:-:|:-:|:-:|:-:|:--
**learning\_rate** | -l | float | 0.001 |  Set **learning\_rate** when **mode** is *sync*, *async* or *local*. This parameter is used in SSGD, ASGD, and DC-ASGD to scale the gradient.
**local\_lr** | -ll | float | 0.001 | Set **local\_lr** when **mode** is *esync*. This parameter is only used in ESync to scale the gradient.
**global\_lr** | -gl | float | 1.0 | Set **global\_lr** when **mode** is *esync*. This parameter is only used in ESync to scale the global learning rate, which can be simply set to 1.0.
**batch\_size** | -b | int | 64 | The number of samples processed in an iteration on each device.
**data\_dir**| -dd | string | /home/lizh/ESync/data | Path to the data files. Include a folder named *fashion-mnist*, which contains *t10k-images-idx3-ubyte.gz*, *t10k-labels-idx1-ubyte.gz*, *train-images-idx3-ubyte.gz*, *train-labels-idx1-ubyte.gz*. The Fashion-MNIST dataset is available on [Github](https://github.com/zalandoresearch/fashion-mnist).
**gpu** | -g | int | 0 | The ID of GPU used for training. We default to using only one GPU for each process in the current version, i.e. only one integer is allowed.
**cpu** | -c | int | 0 | Default to training on GPU 0 (set by the option **gpu**), set **cpu** to 1 to support training on CPU.
**network** | -n | string | *resnet18-v1* | The network used to evaluate the performance of ESync, SSGD, ASGD and DC-ASGD. We support [*alexnet*, *resnet18-v1*, *resnet50-v1*, *resnet50-v2*, *mobilenet-v1*, *mobilenet-v2*, *inception-v3*] in the current version.
**log\_dir** | -ld | string | /home/lizh/ESync/logs | Path to save the logs. The folder named *logs* will be created automatically at the specified path, and it will be emptied during initialization. The Measure module will create subfolders "{device\_name}{device\_id}" and save log files "iter-{iter\_num}.txt" in these subfolders.
**eval\_duration** | -e | int | 1 | Interval for model evaluation, default to evaluating the model in each communication round. We recommend evaluating the model on devices with strong computing capability.
**mode** | -m | string | *esync* | Support [*esync*, *sync*, *async*, *local*]. Set **mode** to *local* to train the model on single device.
**use\_dcasgd** | -dcasgd | int | 0 | Set **use\_dcasgd** to 1 to enable [DC-ASGD](https://www.researchgate.net/publication/308692359_Asynchronous_Stochastic_Gradient_Descent_with_Delay_Compensation_for_Distributed_Deep_Learning) optimizer when **mode** is *async*.
**split\_by\_class** | -s | int | 0 | Default to allocating datasets using the uniform random sampling. Set **split\_by\_class** to 1 to allocate specific classes of samples to each device, for example, allocate samples with labels 0\~4 to device 0 and samples with labels 5\~9 to device 1.
**state\_server\_ip** | -ip | string | 10.1.1.34 | The IP of State Server.
**state\_server\_port** | -port | string | 10010 | The port of State Server.

> Note: The default values can be modified through <code>config.py</code>.

> Note: DO NOT run multiple processes that use the same device on each server, otherwise the log files will be overwritten.

## Start Training

### Training on a single device

```
python main.py -m local -g 0 -n resnet18-v1 -l 0.001 -b 64 -e 1000
```

In the above command, we train the *resnet18-v1* model on a single device (GPU 0) using learning rate 0.001 and batch size 64 and evaluate the model every 1000 iterations. This mode only prints test accuracy on the terminal without recording log files.

### Distributed Training

Suppose we have two servers, cloud1 (IP: 10.1.1.29) and cloud3 (IP: 10.1.1.33), and each with two GPUs. We take GPU 0, GPU 1 and CPU of these servers as separate workers, and obtain a small-scale heterogeneous cluster with 6 workers. In addition, we take an extra server cloud2 (IP: 10.1.1.34) to take the role of Parameter Server, Scheduler and State Server.

**Step 1: Start the State Server** 

> Note: This step can be ignored if **mode** is not *esync*.

The state server is used in ESync algorithm to assign the number of local iterations for each device automatically. The main idea is that, when the slowest device completes computations, other devices have completed local iterations as many times as possible.

Run the following commands on cloud2 to start the state server:

```
> cd /path/to/ESync/SimpleStateServer
> nohup python manage.py runserver 0.0.0.0:10010 > /dev/null &
```

The state server will listen on port 10010 in the background to wait for the queries from workers.

**Step 2: Start the Schduler**

Run the following commands on cloud2 (IP: 10.1.1.34) to start the scheduler:

```
> DMLC_ROLE=scheduler DMLC_PS_ROOT_URI=10.1.1.34 DMLC_PS_ROOT_PORT=9091 DMLC_NUM_SERVER=1 \
  DMLC_NUM_WORKER=6 PS_VERBOSE=1 DMLC_INTERFACE=eno2 \
  nohup python main.py > scheduler.log &
```

We start the scheduler on cloud2 and listen on port 9091 to wait for the messages (e.g. register, heartbeat) from workers. We specify the scheduler to use CPU to avoid errors that no GPU resources available, and specify **mode** to *esync* to run ESync algorithm.

> Note: Set **mode** to *sync* or *async* to run SSGD or ASGD. If **mode** is *async*, you can set **use\_dcasgd** to 1 to enable DC-ASGD.

> Note: Use **-ll** to specify the local learning rate in ESync, and use **-l** to specify the learning rate in SSGD, ASGD and DC-ASGD.

> Note: Specify the network interface manually through **DMLC_INTERFACE** if multiple network interfaces exist on the server, otherwise, we may fail to access other servers.

> Note: Set **PS\_VERBOSE** to 1, and check the log of PS-LITE to troubleshoot errors.

**Step 3: Start the Server**

Run the following commands on cloud2 (IP: 10.1.1.34) to start the parameter server:

```
> DMLC_ROLE=server DMLC_PS_ROOT_URI=10.1.1.34 DMLC_PS_ROOT_PORT=9091 DMLC_NUM_SERVER=1 \
  DMLC_NUM_WORKER=6 PS_VERBOSE=1 DMLC_INTERFACE=eno2 \
  nohup python main.py > server.log &
```

The parameter server will create a TCP connection to the scheduler and complete registration automatically by specifying **DMLC\_PS\_ROOT\_URI** and **DMLC\_PS\_ROOT\_PORT** (the same as workers). The aggregation operations will be performed on GPU if **cpu** is set to *False*.

**Step 4: Start the Workers**

Run the following commands on cloud1 (IP: 10.1.1.29) and cloud3 (IP: 10.1.1.33) to start the workers:

```
> DMLC_ROLE=worker DMLC_PS_ROOT_URI=10.1.1.34 DMLC_PS_ROOT_PORT=9091 DMLC_NUM_SERVER=1 \
  DMLC_NUM_WORKER=6 PS_VERBOSE=1 DMLC_INTERFACE=eno2 \
  nohup python main.py -m esync -g 0 -n resnet18-v1 -ll 0.001 -b 64 -dcasgd 0 -s 0 > worker_gpu_0.log &
  
> DMLC_ROLE=worker DMLC_PS_ROOT_URI=10.1.1.34 DMLC_PS_ROOT_PORT=9091 DMLC_NUM_SERVER=1 \
  DMLC_NUM_WORKER=6 PS_VERBOSE=1 DMLC_INTERFACE=eno2 \
  nohup python main.py -m esync -g 1 -n resnet18-v1 -ll 0.001 -b 64 -dcasgd 0 -s 0 > worker_gpu_1.log &
  
> DMLC_ROLE=worker DMLC_PS_ROOT_URI=10.1.1.34 DMLC_PS_ROOT_PORT=9091 DMLC_NUM_SERVER=1 \
  DMLC_NUM_WORKER=6 PS_VERBOSE=1 DMLC_INTERFACE=eno2 \
  nohup python main.py -m esync -c 1 -n resnet18-v1 -ll 0.001 -b 64 -dcasgd 0 -s 0 > worker_cpu.log &
```

We start workers on GPU 0, GPU 1 and CPU respectively on both cloud1 and cloud3, and obtain a small-scale heterogeneous cluster with 6 workers, 1 parameter server, 1 scheduler and 1 state server (optional). In the example above, we run a distributed machine learning task to train the ResNet18-v1 model based on ESync algorithm, and the evaluation operations will be performed on device with rank 0.

## Log Visualization

To visualize the log data, we should collect the log files on cloud1 and cloud3 manually. Take the ResNet18-v1 model and ESync algorithm as an example, we create a new folder named *resnet18-v1* with a subfolder named *esync*, and rename the log folders as their host names, the structure is as follows:

```
resnet18-v1
    |── esync
          |── cloud1
                |── cpu
                |── gpu0
                |── gpu1
          |── cloud3
                |── cpu
                |── gpu0
                |── gpu1
```

Pre-trained log files are available at <code>lfs/logs.zip</code>, which contains the logs generated by several classic models (AlexNet, Inception-v3, MobileNet-v1, ResNet18-v1, ResNet50-v1, ResNet50-v2) and five training modes (ESync, SSGD, ASGD, DC-ASGD, Sequential SGD). In addition, we also provide the test accuracy curve of ESync and Sync in the *results* folders.

### Log Summary

In this section, we summarize the large number of *iter-x.txt* files into a JSON file, which includes statistical data on each device and is used for visualizing the logs.

Run the following commands to summarize the logs:

```
> python summary.py -b /path/to/logs -n resnet18-v1 -m esync
```

The script <code>summary.py</code> will read files from /path/to/logs/resnet18-v1/esync and generate a JSON file named *ESync.json* there.

### Visualization

#### ResNet18-v1

Run the following commands to visualize the logs of *resnet18-v1*:

```
> python drawer.py -b /path/to/logs -n resnet18-v1
```

The script <code>drawer.py</code> will read data from:

* /path/to/logs/resnet18-v1/esync/ESync.json
* /path/to/logs/resnet18-v1/sync/Sync.json
* /path/to/logs/resnet18-v1/async/Async.json (optional)
* /path/to/logs/resnet18-v1/dcasgd/DCASGD.json (optional)
* /path/to/logs/resnet18-v1/esync-niid/ESync-Non-IID.json (optional)
* /path/to/logs/resnet18-v1/sync-niid/Sync-Non-IID.json (optional)
* /path/to/logs/resnet18-v1/async-niid/Async-Non-IID.json (optional)
* /path/to/logs/resnet18-v1/dcasgd-niid/DCASGD-Non-IID.json (optional)

and draw the following figures:

1\. Test Accuracy Curve of ESync, SSGD, ASGD, DC-ASGD on i.i.d. Fashion-MNIST dataset;

2\. Test Accuracy Curve of ESync, SSGD, ASGD, DC-ASGD on non-i.i.d. Fashion-MNIST dataset;

<img src="images/resnet18-v1-accuracy-iid.png" width="400px" /><img src="images/resnet18-v1-accuracy-non-iid.png" width="400px" />

3\. Data Throughput of ESync, SSGD, ASGD, DC-ASGD;

4\. Traffic Load of ESync, SSGD, ASGD, DC-ASGD;

<img src="images/resnet18-v1-data-throughput.png" width="400px" /><img src="images/resnet18-v1-traffic-load.png" width="400px" />

5\. Computing Time Ratio of ESync, SSGD, ASGD, DC-ASGD.

<img src="images/resnet18-v1-computing-time-ratio.png" width="800px" />

> Note: The dotted lines in Fig. 1 and Fig. 2 draw the test accuracy when training on a single device. The test accuracy of standalone training is available in the file *standalone.txt* and should be specified at lines 202 and 215 in the file <code>drawer.py</code>.

> Note: Since only device with rank 0 records the values of test accuracy, we need to ensure that the variable *config* in the file <code>drawer.py</code> at lines 201 and 214 specifies the correct devices. We list the *config* for each model in our experiments as follows:

* AlexNet: <code>("cloud3", "gpu1"), ("cloud3", "gpu1"), (null, null), (null, null)</code>;
* Inception-v3: <code>("cloud3", "gpu0"), ("cloud3", "gpu0"), (null, null), (null, null)</code>;
* MobileNet-v1: <code>("cloud3", "gpu1"), ("cloud3", "gpu0"), (null, null), (null, null)</code>;
* ResNet18-v1: <code>("cloud3", "gpu0"), ("cloud3", "gpu1"), ("cloud3", "gpu0"), ("cloud3", "gpu1")</code>;
* ResNet50-v1: <code>("cloud3", "gpu0"), ("cloud3", "gpu1"), (null, null), (null, null)</code>;
* ResNet50-v2: <code>("cloud3", "gpu0"), ("cloud3", "gpu1"), (null, null), (null, null)</code>;

#### Other Models

<img src="images/alexnet-accuracy-iid.png" width="400px" /><img src="images/inception-v3-accuracy-iid.png" width="400px" />

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Test Accuracy on AlexNet
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Test Accuracy on Inception-v3

<img src="images/resnet50-v1-accuracy-iid.png" width="400px" /><img src="images/resnet50-v2-accuracy-iid.png" width="400px" />

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Test Accuracy on ResNet50-v1
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Test Accuracy on ResNet50-v2

<img src="images/mobilenet-v1-accuracy-iid.png" width="400px" />

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Test Accuracy on MobileNet-v1

# Speed-up and Convergence

We define the speedup as the time ratio to achieve test accuracy 0.8, and compare the speedup of ESync to Sync and the test accuracy with the standalone training on several classic models. The results show that ESync can greatly accelerate the training process and finally achieve the same (sometimes higher) accuracy as sequential SGD. 

<div align="center"><img src="images/speedup-convergence.png" width="500px" /></div>

# References

[[1] Chen, Jianmin, et al. "Revisiting distributed synchronous SGD." arXiv preprint arXiv:1604.00981 (2016).](https://arxiv.org/pdf/1604.00981.pdf)

[[2] Dean, Jeffrey, et al. "Large scale distributed deep networks." Advances in neural information processing systems. 2012.](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf)

[[3] Li, Mu, et al. "Scaling distributed machine learning with the parameter server." 11th {USENIX} Symposium on Operating Systems Design and Implementation ({OSDI} 14). 2014.](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf)

[[4] Gibiansky, Andrew. "Bringing HPC techniques to deep learning". http://research.baidu.com/bringing-hpc-techniques-deep-learning/, 2017.2.21.](http://research.baidu.com/bringing-hpc-techniques-deep-learning/)

[[5] Rabenseifner, Rolf. "Optimization of collective reduction operations." International Conference on Compu- tational Science, Krakow, Poland, 2004.6.6-6.9.](https://link.springer.com/content/pdf/10.1007%2F978-3-540-24685-5_1.pdf)

[[6] Zheng, Shuxin, et al. "Asynchronous stochastic gradient descent with delay compensation." Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.](https://www.researchgate.net/publication/308692359_Asynchronous_Stochastic_Gradient_Descent_with_Delay_Compensation_for_Distributed_Deep_Learning)
