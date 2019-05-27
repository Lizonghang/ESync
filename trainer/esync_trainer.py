import time
import requests
import mxnet as mx
from mxnet import nd, kv, autograd
from utils import load_data, get_batch, eval_acc, Measure


def trainer(kwargs):
    local_lr = kwargs["local_lr"]
    global_lr = kwargs["global_lr"]
    batch_size = kwargs["batch_size"]
    data_dir = kwargs["data_dir"]
    log_dir = kwargs["log_dir"]
    eval_duration = kwargs["eval_duration"]
    ctx = kwargs["ctx"]
    shape = kwargs["shape"]
    common_url = kwargs["common_url"]
    net = kwargs["net"]
    loss = kwargs["loss"]
    split_by_class = kwargs["split_by_class"]
    factor = kwargs["factor"]

    nd.waitall()
    ts = time.time()
    net(nd.random.uniform(shape=shape, ctx=ctx))
    nd.waitall()
    te = time.time()
    c = te - ts

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

    if rank == 0:
        requests.post(common_url % "init", data={"num_workers": num_workers, "epsilon": 0.5})

    train_iter, test_iter = load_data(batch_size, num_workers, rank,
                                      split_by_class=split_by_class, resize=shape[-2:], root=data_dir)

    trainer = mx.gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": local_lr})

    pre_params = {}
    for i, param in enumerate(params):
        if param.grad_req == "null":
            continue
        idx = param2idx[param.name]
        pre_params[idx] = param.data().copy()
    nd.waitall()

    self_iters = 0
    global_iters = 1
    te = time.time()

    if ctx == mx.cpu():
        subdir = "cpu"
    elif ctx == mx.gpu(0):
        subdir = "gpu0"
    elif ctx == mx.gpu(1):
        subdir = "gpu1"
    else:
        print("[ERROR] This gpu is not supported.")
        return

    measure = Measure(log_dir, subdir)
    measure.set_num_iters(global_iters)

    print("Training on", ctx)
    while True:
        for _, batch in enumerate(train_iter):
            measure.next_iter()
            measure.start("check ready")
            ready = requests.post(common_url % "apply",
                                  data={"r": rank, "k": self_iters, "c": c, "te": te}
                                  ).json().get("ready")
            measure.stop("check ready")

            if not ready:
                nd.waitall()
                ts = time.time()

                measure.start("get_batch")
                Xs, ys, batch_size = get_batch(batch, ctx)
                nd.waitall()
                measure.stop("get_batch")

                measure.start("forward and backward")
                calc_start_time = time.time()
                ls = []
                with autograd.record():
                    y_hats = [net(X) for X in Xs]
                    ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
                for l in ls:
                    l.backward()
                nd.waitall()
                calc_time = time.time() - calc_start_time
                time.sleep((factor - 1) * calc_time)
                measure.stop("forward and backward")
                measure.add_samples(batch_size)

                measure.start("local update")
                self_iters += 1
                trainer.step(1)
                # for i, param in enumerate(params):
                #     if param.grad_req == "null":
                #         continue
                #     param_ = param.data() - lr * (param.grad() + wd * param.data())
                #     param.set_data(param_)
                nd.waitall()
                measure.stop("local update")

                te = time.time()
                c = te - ts

            else:
                measure.start("global update")
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
                        model_update = (param.data() - pre_params[idx])
                        kvstore_dist.push(idx, model_update, priority=-idx)
                        agg_grads = nd.zeros(param.shape, ctx=ctx)
                        kvstore_dist.pull(idx, agg_grads, priority=-idx)
                        agg_grads.wait_to_read()
                        param_ = pre_params[idx] + global_lr * (agg_grads / num_workers)
                        param.set_data(param_)
                        pre_params[idx] = param_.copy()
                nd.waitall()
                measure.stop("global update")

                if rank == 0 and global_iters % eval_duration == 0:
                    measure.start("evaluation")
                    test_acc = eval_acc(test_iter, net, ctx)
                    print('[Iteration %d] Test Acc %.3f' % (global_iters, test_acc))
                    measure.set_accuracy(test_acc)
                    measure.stop("evaluation")

                measure.save_report()

                self_iters = 0
                global_iters += 1
                te = time.time()

                if global_iters == 200:
                    return 0

                measure.start("reset state server")
                requests.post(common_url % "reset", data={"r": rank, "t": global_iters, "te": te})
                measure.stop("reset state server")

                measure.reset(global_iters)
