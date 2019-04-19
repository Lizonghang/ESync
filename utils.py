import os
import time
import json
import shutil
import random
import mxnet as mx
from mxnet.gluon import data as gdata
from mxnet.gluon import utils as gutils


class SplitSampler(gdata.sampler.Sampler):
    def __init__(self, length, num_parts=1, part_index=0):
        self.part_len = length // num_parts
        self.start = self.part_len * part_index
        self.end = self.start + self.part_len

    def __iter__(self):
        indices = list(range(self.start, self.end))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.part_len


class ClassSplitSampler(gdata.sampler.Sampler):
    def __init__(self, class_list, length, num_parts=1, part_index=0):
        self.class_list = class_list
        self.part_len = length // num_parts
        self.start = self.part_len * part_index
        self.end = self.start + self.part_len

    def __iter__(self):
        indices = self.class_list[self.start:self.end]
        return iter(indices)

    def __len__(self):
        return self.part_len


def load_data(batch_size, num_workers=1, rank=0, split_by_class=False, resize=None,
              root=os.path.join("/", "home", "lizh", "ESync", "data", "fashion-mnist")):
    root = os.path.expanduser(root)
    train = gdata.vision.FashionMNIST(root=root, train=True)
    test = gdata.vision.FashionMNIST(root=root, train=False)

    if num_workers > 1:
        if split_by_class:
            num_classes = 10
            class_list = [[] for _ in range(num_classes)]
            for idx, sample in enumerate(train):
                class_list[sample[1]].append(idx)
            flat_class_list = []
            for class_ in class_list:
                flat_class_list += class_
            sampler = ClassSplitSampler(flat_class_list, len(train), num_workers, rank)
        else:
            sampler = SplitSampler(len(train), num_workers, rank)
    else:
        sampler = None

    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)

    train_iter = gdata.DataLoader(train.transform_first(transformer), batch_size, sampler=sampler, num_workers=4)
    test_iter = gdata.DataLoader(test.transform_first(transformer), batch_size, num_workers=4)
    return train_iter, test_iter


def get_batch(batch, ctx):
    features, labels = batch
    if type(ctx) == mx.Context:
        ctx = [ctx]
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])


def eval_acc(test_iter, net, ctx):
    test_acc = 0.0
    for X, y in test_iter:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        pred_class = net(X).argmax(axis=1)
        batch_acc = (pred_class == y.astype("float32")).mean().asscalar()
        test_acc += batch_acc
    return test_acc / len(test_iter)


class Measure:
    def __init__(self, log_path):
        self.num_iters = -1
        self.self_iter = -1
        self.begin = time.time()
        self.total_time = -1
        self.start_time = 0.
        self.time_map = {}
        self.accuracy = 0.
        self.num_sampels = 0
        self.log_path = log_path
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        os.mkdir(log_path)

    def set_num_iters(self, num_iters):
        assert num_iters >= 0
        self.num_iters = num_iters

    def next_iter(self):
        self.self_iter += 1
        self.time_map[self.self_iter] = {}

    def start(self, name):
        self.time_map[self.self_iter][name] = 0
        self.start_time = time.time()

    def stop(self, name):
        if self.time_map.get(self.self_iter, -1) != -1 and \
           self.time_map[self.self_iter].get(name, -1) == 0:
            self.time_map[self.self_iter][name] = round(time.time() - self.start_time, 4)

    def set_accuracy(self, accuracy):
        self.accuracy = accuracy

    def add_samples(self, num_samples):
        self.num_sampels += num_samples

    def reset(self, num_iters=-1):
        self.start_time = 0.
        self.time_map = {}
        self.self_iter = -1
        self.accuracy = 0.
        self.num_sampels = 0
        if num_iters != -1:
            self.num_iters = num_iters

    def save_report(self):
        if self.num_iters == -1:
            print("[Error] Incorrect iteration number %d." % self.num_iters)
            return -1

        log = {
            "num_iters": self.num_iters,
            "time": self.time_map,
            "num_samples": self.num_sampels,
            "total_time": time.time() - self.begin
        }

        if self.accuracy:
            log.update({"accuracy": self.accuracy})

        log_file = os.path.join(self.log_path, "iter-%d.txt" % self.num_iters)
        with open(log_file, "w") as fp:
            json.dump(log, fp)
