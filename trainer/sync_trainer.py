import os
import mxnet as mx
from mxnet import kv, nd, autograd
from utils import load_data, get_batch, eval_acc, Measure


def trainer(kwargs):
    lr = kwargs["lr"]
    batch_size = kwargs["batch_size"]
    log_dir = kwargs["log_dir"]
    eval_duration = kwargs["eval_duration"]
    ctx = kwargs["ctx"]
    shape = kwargs["shape"]
    net = kwargs["net"]
    loss = kwargs["loss"]
    split_by_class = kwargs["split_by_class"]

    kvstore_dist = kv.create("dist_sync")
    rank = kvstore_dist.rank
    num_workers = kvstore_dist.num_workers
    print("My rank is", rank)

    params = list(net.collect_params().values())
    param2idx = {}
    for i, param in enumerate(params):
        param2idx[param.name] = i

    for param in params:
        idx = param2idx[param.name]
        kvstore_dist.init(idx, param.data())
        kvstore_dist.pull(idx, param.data(), priority=-idx)

    train_iter, test_iter = load_data(batch_size, num_workers, rank, split_by_class=split_by_class, resize=shape[-2:])

    if ctx == mx.cpu():
        subdir = "cpu"
    elif ctx == mx.gpu(0):
        subdir = "gpu0"
    elif ctx == mx.gpu(1):
        subdir = "gpu1"
    else:
        print("[ERROR] This gpu is not supported.")
        return

    global_iters = 1
    measure = Measure(log_dir, subdir)
    measure.set_num_iters(global_iters)

    print("Training on", ctx)
    while True:
        for _, batch in enumerate(train_iter):
            measure.next_iter()

            measure.start("get_batch")
            Xs, ys, batch_size = get_batch(batch, ctx)
            nd.waitall()
            measure.stop("get_batch")

            measure.start("forward and backward")
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            nd.waitall()
            measure.stop("forward and backward")
            measure.add_samples(batch_size)

            measure.start("global aggregate")
            for i, param in enumerate(params):
                if param.grad_req == "null":
                    if "gamma" in param.name or "beta" in param.name \
                            or "running_mean" in param.name or "running_var" in param.name:
                        idx = param2idx[param.name]
                        kvstore_dist.push(idx, param.data(), priority=-idx)
                        agg_data = nd.zeros(param.shape, ctx=ctx)
                        kvstore_dist.pull(idx, agg_data, priority=-idx)
                        agg_data.wait_to_read()
                        param.set_data(agg_data / num_workers)
                else:
                    idx = param2idx[param.name]
                    kvstore_dist.push(idx, param.grad(), priority=-idx)
                    agg_grads = nd.zeros(param.shape, ctx=ctx)
                    kvstore_dist.pull(idx, agg_grads, priority=-idx)
                    agg_grads.wait_to_read()
                    param_ = param.data() - lr * (agg_grads / num_workers)
                    param.set_data(param_)
            nd.waitall()
            measure.stop("global aggregate")

            if rank == 0 and global_iters % eval_duration == 0:
                measure.start("evaluation")
                test_acc = eval_acc(test_iter, net, ctx)
                print('[Iteration %d] Test Acc %.3f' % (global_iters, test_acc))
                measure.set_accuracy(test_acc)
                measure.stop("evaluation")

            measure.save_report()

            global_iters += 1

            measure.reset(global_iters)
