import mxnet as mx
from mxnet import autograd
from utils import load_data, get_batch, eval_acc


def trainer(kwargs):
    lr = kwargs["lr"]
    batch_size = kwargs["batch_size"]
    eval_duration = kwargs["eval_duration"]
    ctx = kwargs["ctx"]
    shape = kwargs["shape"]
    net = kwargs["net"]
    loss = kwargs["loss"]
    train_iter, test_iter = load_data(batch_size, resize=shape[-2:])
    trainer = mx.gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": lr})
    iters = 0
    print("Training on", ctx)
    while True:
        for _, batch in enumerate(train_iter):
            Xs, ys, batch_size = get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            trainer.step(1)
            iters += 1
            if iters % eval_duration == 0:
                test_acc = eval_acc(test_iter, net, ctx)
                print('[Iteration %d] Test Acc %.3f' % (iters, test_acc))
