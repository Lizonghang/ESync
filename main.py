import argparse
import mxnet as mx
from config import *
from mxnet import init, nd
from mxnet.gluon import loss as gloss


if __name__ == "__main__":
    """COMMAND
    [Standalone]
    python ~/ESync/main.py -g 0 -m local -n resnet18-v1 -e 1000

    [Distributed]
    DMLC_ROLE=scheduler DMLC_PS_ROOT_URI=10.1.1.34 DMLC_PS_ROOT_PORT=9091 DMLC_NUM_SERVER=1 DMLC_NUM_WORKER=6 \
        PS_VERBOSE=1 DMLC_INTERFACE=eno2 \
        nohup python ~/ESync/main.py -c 1 -m esync -dcasgd 0 -l 0.0005 > scheduler.log &

    DMLC_ROLE=server DMLC_PS_ROOT_URI=10.1.1.34 DMLC_PS_ROOT_PORT=9091 DMLC_NUM_SERVER=1 DMLC_NUM_WORKER=6 \
        PS_VERBOSE=1 DMLC_INTERFACE=eno2 \
        nohup python ~/ESync/main.py -c 1 -m esync -dcasgd 0 -l 0.0005 > server.log &

    DMLC_ROLE=worker DMLC_PS_ROOT_URI=10.1.1.34 DMLC_PS_ROOT_PORT=9091 DMLC_NUM_SERVER=1 DMLC_NUM_WORKER=6 \
        PS_VERBOSE=1 DMLC_INTERFACE=eno2 \
        nohup python ~/ESync/main.py -g 0 -m esync -dcasgd 0 -n resnet18-v1 -s 0 -l 0.0005 > worker_gpu_0.log &
    DMLC_ROLE=worker DMLC_PS_ROOT_URI=10.1.1.34 DMLC_PS_ROOT_PORT=9091 DMLC_NUM_SERVER=1 DMLC_NUM_WORKER=6 \
        PS_VERBOSE=1 DMLC_INTERFACE=eno2 \
        nohup python ~/ESync/main.py -g 1 -m esync -dcasgd 0 -n resnet18-v1 -s 0 -l 0.0005 > worker_gpu_1.log &
    DMLC_ROLE=worker DMLC_PS_ROOT_URI=10.1.1.34 DMLC_PS_ROOT_PORT=9091 DMLC_NUM_SERVER=1 DMLC_NUM_WORKER=6 \
        PS_VERBOSE=1 DMLC_INTERFACE=eno2 \
        nohup python ~/ESync/main.py -c 1 -m esync -dcasgd 0 -n resnet18-v1 -s 0 -l 0.0005 > worker_cpu.log &
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("-ll", "--local-lr", type=float, default=LEARNING_RATE_LOCAL)
    parser.add_argument("-gl", "--global-lr", type=float, default=LEARNING_RATE_GLOBAL)
    parser.add_argument("-b", "--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("-dd", "--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("-g", "--gpu", type=int, default=DEFAULT_GPU_ID)
    parser.add_argument("-c", "--cpu", type=int, default=USE_CPU)
    parser.add_argument("-n", "--network", type=str, default=NETWORK)
    parser.add_argument("-ld", "--log-dir", type=str, default=LOG_DIR)
    parser.add_argument("-e", "--eval-duration", type=int, default=EVAL_DURATION)
    parser.add_argument("-m", "--mode", type=str, default=MODE)
    parser.add_argument("-dcasgd", "--use-dcasgd", type=int, default=USE_DCASGD)
    parser.add_argument("-s", "--split-by-class", type=int, default=SPLIT_BY_CLASS)
    parser.add_argument("-ip", "--state-server-ip", type=str, default=STATE_SERVER_IP)
    parser.add_argument("-port", "--state-server-port", type=str, default=STATE_SERVER_PORT)
    args, unknown = parser.parse_known_args()

    lr = args.learning_rate
    local_lr = args.local_lr
    global_lr = args.global_lr
    batch_size = args.batch_size
    data_dir = args.data_dir
    network = args.network
    eval_duration = args.eval_duration
    log_dir = args.log_dir
    ctx = mx.cpu() if args.cpu else mx.gpu(args.gpu)
    mode = args.mode
    use_dcasgd = args.use_dcasgd
    shape = (batch_size, 1, 28, 28)
    split_by_class = args.split_by_class
    state_server_ip = args.state_server_ip
    state_server_port = args.state_server_port
    common_url = "http://{ip}:{port}/%s/".format(ip=state_server_ip, port=state_server_port)

    net = None
    if network == "resnet18-v1":
        from symbols.resnet import resnet18_v1
        net = resnet18_v1(classes=10)
    elif network == "resnet50-v1":
        from symbols.resnet import resnet50_v1
        net = resnet50_v1(classes=10)
    elif network == "resnet50-v2":
        from symbols.resnet import resnet50_v2
        net = resnet50_v2(classes=10)
    elif network == "alexnet":
        from symbols.alexnet import alexnet
        net = alexnet(classes=10)
        shape = (batch_size, 1, 224, 224)
    elif network == "mobilenet-v1":
        from symbols.mobilenet import mobilenet1_0
        net = mobilenet1_0(classes=10)
    elif network == "mobilenet-v2":
        from symbols.mobilenet import mobilenet_v2_1_0
        net = mobilenet_v2_1_0(classes=10)
    elif network == "inception-v3":
        from symbols.inception import inception_v3
        net = inception_v3(classes=10)
        shape = (batch_size, 1, 299, 299)
    net.initialize(init=init.Xavier(), ctx=ctx)
    net(nd.random.uniform(shape=shape, ctx=ctx))

    loss = gloss.SoftmaxCrossEntropyLoss()

    kwargs = {
        "lr": lr,
        "batch_size": batch_size,
        "data_dir": data_dir,
        "log_dir": log_dir,
        "eval_duration": eval_duration,
        "ctx": ctx,
        "shape": shape,
        "net": net,
        "loss": loss,
        "split_by_class": split_by_class
    }

    if mode == "esync":
        from trainer.esync_trainer import trainer
        kwargs.update({
            "local_lr": local_lr,
            "global_lr": global_lr,
            "common_url": common_url
        })
        trainer(kwargs)
    elif mode == "sync":
        from trainer.sync_trainer import trainer
        trainer(kwargs)
    elif mode == "async":
        from trainer.async_trainer import trainer
        kwargs.update({
            "use_dcasgd": use_dcasgd
        })
        trainer(kwargs)
    elif mode == "local":
        from trainer.local_trainer import trainer
        trainer(kwargs)
    else:
        raise NotImplementedError("Not implemented.")
