import os
import argparse
import matplotlib.pyplot as plt
from summary import load_summary


def down_sampling(accuracy_list, num_iters, interval, start=0):
    return accuracy_list[start::interval], num_iters[start::interval]


def interval_averaging(accuracy_list, interval):
    num_iters = len(accuracy_list)
    padded_list = [accuracy_list[0]] * (interval - 1) + accuracy_list
    smoothed = accuracy_list.copy()
    for iter_idx in range(num_iters):
        idx = iter_idx + interval - 1
        avg = sum(padded_list[idx-(interval-1):idx+1]) / interval
        smoothed[iter_idx] = avg
    return smoothed


def draw_accuracy(esync_summary, sync_summary, async_summary, config, vline=0.9,
                  fignum=0, timespan=120, down_sample_interval=10, smooth_interval=10, shift=0):
    if not (esync_summary or sync_summary or async_summary):
        return

    plt.figure(fignum)
    iid = config[3]
    title = "Test Accuracy Curve of ESync, Sync, Async on %si.i.d. dataset" % ("" if iid else "non-")
    plt.title(title, fontsize=fontsize)
    plt.xlabel("Time (minutes)", fontsize=fontsize)
    plt.ylabel("Test Accuracy", fontsize=fontsize)
    plt.xlim((0, timespan))
    plt.ylim((0, 1))

    if esync_summary:
        node, worker = config[0]
        accuracy_list = esync_summary[node][worker]["accuracy_list"]
        elapsed_time_list = [s / 60. for s in esync_summary[node][worker]["elapsed_time_list"]]
        smoothed, elapsed_time_list = down_sampling(accuracy_list, elapsed_time_list, down_sample_interval, shift)
        smoothed = [0.1] + smoothed
        elapsed_time_list = [0] + elapsed_time_list
        smoothed = interval_averaging(smoothed, smooth_interval)
        plt.plot(elapsed_time_list, smoothed, c="m")

    if sync_summary:
        node, worker = config[1]
        accuracy_list = sync_summary[node][worker]["accuracy_list"]
        elapsed_time_list = [s / 60. for s in sync_summary[node][worker]["elapsed_time_list"]]
        smoothed, elapsed_time_list = down_sampling(accuracy_list, elapsed_time_list, down_sample_interval, shift)
        smoothed = [0.1] + smoothed
        elapsed_time_list = [0] + elapsed_time_list
        smoothed = interval_averaging(smoothed, smooth_interval)
        plt.plot(elapsed_time_list, smoothed, c="orange")

    if async_summary:
        node, worker = config[2]
        accuracy_list = async_summary[node][worker]["accuracy_list"]
        elapsed_time_list = [s / 60. for s in async_summary[node][worker]["elapsed_time_list"]]
        smoothed, elapsed_time_list = down_sampling(accuracy_list, elapsed_time_list, down_sample_interval, 0)
        smoothed = [0.1] + smoothed
        elapsed_time_list = [0] + elapsed_time_list
        smoothed = interval_averaging(smoothed, smooth_interval)
        plt.plot(elapsed_time_list, smoothed, c="b")

    plt.legend(["ESync", "Sync", "Async"], fontsize=fontsize)
    plt.plot((0, timespan), (vline, vline), c="k", linestyle=":", linewidth="1")


def draw_data_throughput(esync_summary, sync_summary, async_summary, fignum=1):
    if not (esync_summary or sync_summary or async_summary):
        return

    esync_throughput = 0
    if esync_summary:
        total_samples = 0
        total_time = 0
        for node in esync_summary.keys():
            for worker in esync_summary[node].keys():
                total_samples += sum(esync_summary[node][worker]["num_samples_list"])
                if not total_time:
                    total_time = esync_summary[node][worker]["elapsed_time_list"][-1]
        esync_throughput = int(total_samples / total_time)
        print("[ESync] Data Throughput (samples per second):", esync_throughput)

    sync_throughput = 0
    if sync_summary:
        total_samples = 0
        total_time = 0
        for node in sync_summary.keys():
            for worker in sync_summary[node].keys():
                total_samples += sum(sync_summary[node][worker]["num_samples_list"])
                if not total_time:
                    total_time = sync_summary[node][worker]["elapsed_time_list"][-1]
        sync_throughput = int(total_samples / total_time)
        print("[Sync] Data Throughput (samples per second):", sync_throughput)

    async_throughput = 0
    if async_summary:
        total_samples = 0
        total_time = 0
        for node in async_summary.keys():
            for worker in async_summary[node].keys():
                total_samples += sum(async_summary[node][worker]["num_samples_list"])
                if not total_time:
                    total_time = async_summary[node][worker]["elapsed_time_list"][-1]
        async_throughput = int(total_samples / total_time)
        print("[Async] Data Throughput (samples per second):", async_throughput)

    plt.figure(fignum, figsize=(6, 4))
    plt.bar(x=(0, 1, 2),
            height=(esync_throughput, sync_throughput, async_throughput),
            color=("m", "orange", "b"),
            width=0.5,
            align="center")
    plt.title("Data Throughput of ESync, Sync, Async", fontsize=fontsize)
    plt.xlabel("Synchronous Mode", fontsize=fontsize)
    plt.ylabel("Data Throughput (samples per second)", fontsize=fontsize)
    plt.xticks((0, 1, 2), ("ESync", "Sync", "Async"), fontsize=fontsize)


def draw_traffic_load(esync_summary, sync_summary, async_summary, config, fignum=2):
    if not (esync_summary or sync_summary or async_summary):
        return

    # the number of parameters of ResNet18-v1: 10.65 Million
    model_size = 42.6

    esync_load = 0
    if esync_summary:
        node, worker = config[0]
        num_workers = 6
        communication_round = len(esync_summary[node][worker]["accuracy_list"])
        data_size = communication_round * num_workers * model_size
        total_time = esync_summary[node][worker]["elapsed_time_list"][-1]
        esync_load = data_size / total_time
        print("[ESync] Traffic Load (MBytes per second):", esync_load)

    sync_load = 0
    if sync_summary:
        node, worker = config[1]
        num_workers = 6
        communication_round = len(sync_summary[node][worker]["accuracy_list"])
        data_size = communication_round * num_workers * model_size
        total_time = sync_summary[node][worker]["elapsed_time_list"][-1]
        sync_load = data_size / total_time
        print("[Sync] Traffic Load (MBytes per second):", sync_load)

    async_load = 0
    if async_summary:
        communication_round = 0
        total_time = 0
        for node in async_summary.keys():
            for worker in async_summary[node].keys():
                communication_round += len(async_summary[node][worker]["self_iters_list"])
                if not total_time:
                    total_time = async_summary[node][worker]["elapsed_time_list"][-1]
        data_size = communication_round * model_size
        async_load = int(data_size / total_time)
        print("[Async] Traffic Load (MBytes per second):", async_load)

    plt.figure(fignum, figsize=(6, 4))
    plt.bar(x=(0, 1, 2),
            height=(esync_load, sync_load, async_load),
            color=("m", "orange", "b"),
            width=0.5,
            align="center")
    plt.title("Traffic Load of ESync, Sync, Async", fontsize=fontsize)
    plt.xlabel("Synchronous Mode", fontsize=fontsize)
    plt.ylabel("Traffic Load (MBytes per second)", fontsize=fontsize)
    plt.xticks((0, 1, 2), ("ESync", "Sync", "Async"), fontsize=fontsize)


def draw_computation_communication_ratio(esync_summary, sync_summary, async_summary, fignum=3):
    if not (esync_summary or sync_summary or async_summary):
        return

    esync_load = {}
    if esync_summary:
        for node in esync_summary.keys():
            esync_load[node] = {}
            for worker in esync_summary[node].keys():
                total_communication_time = sum(esync_summary[node][worker]["aggregate_time_list"])
                total_time = esync_summary[node][worker]["elapsed_time_list"][-1]
                esync_load[node][worker] = total_communication_time / total_time

    sync_load = {}
    if sync_summary:
        for node in sync_summary.keys():
            sync_load[node] = {}
            for worker in sync_summary[node].keys():
                total_communication_time = sum(sync_summary[node][worker]["aggregate_time_list"])
                total_time = sync_summary[node][worker]["elapsed_time_list"][-1]
                sync_load[node][worker] = total_communication_time / total_time

    async_load = {}
    if async_summary:
        for node in async_summary.keys():
            async_load[node] = {}
            for worker in async_summary[node].keys():
                total_communication_time = sum(async_summary[node][worker]["aggregate_time_list"])
                total_time = async_summary[node][worker]["elapsed_time_list"][-1]
                async_load[node][worker] = total_communication_time / total_time

    plt.figure(fignum)
    width = 0.35
    interval = 0.45
    color = ("m", "orange", "b")

    stack_bar21 = [1-esync_load["cloud3"]["gpu0"], 1-sync_load["cloud3"]["gpu0"], 1-async_load["cloud3"]["gpu0"]]
    stack_bar22 = [1-esync_load["cloud3"]["gpu1"], 1-sync_load["cloud3"]["gpu1"], 1-async_load["cloud3"]["gpu1"]]
    stack_bar23 = [1-esync_load["cloud3"]["cpu"], 1-sync_load["cloud3"]["cpu"], 1-async_load["cloud3"]["cpu"]]
    stack_bar24 = [1-esync_load["cloud1"]["gpu0"], 1-sync_load["cloud1"]["gpu0"], 1-async_load["cloud1"]["gpu0"]]
    stack_bar25 = [1-esync_load["cloud1"]["gpu1"], 1-sync_load["cloud1"]["gpu1"], 1-async_load["cloud1"]["gpu1"]]
    stack_bar26 = [1-esync_load["cloud1"]["cpu"], 1-sync_load["cloud1"]["cpu"], 1-async_load["cloud1"]["cpu"]]

    p1 = plt.bar((0*interval, 3+0*interval, 6+0*interval), stack_bar21, width=width, color=color)
    p2 = plt.bar((1*interval, 3+1*interval, 6+1*interval), stack_bar22, width=width, color=color)
    p3 = plt.bar((2*interval, 3+2*interval, 6+2*interval), stack_bar23, width=width, color=color)
    p4 = plt.bar((3*interval, 3+3*interval, 6+3*interval), stack_bar24, width=width, color=color)
    p5 = plt.bar((4*interval, 3+4*interval, 6+4*interval), stack_bar25, width=width, color=color)
    p6 = plt.bar((5*interval, 3+5*interval, 6+5*interval), stack_bar26, width=width, color=color)
    plt.title("Computing Time Ratio of ESync, Sync, Async", fontsize=fontsize)
    plt.xlabel("Devices", fontsize=fontsize)
    plt.ylabel("Computing Time Ratio", fontsize=fontsize)
    plt.xticks((0*interval, 1*interval, 2*interval, 3*interval, 4*interval, 5*interval,
                3+0*interval, 3+1*interval, 3+2*interval, 3+3*interval, 3+4*interval, 3+5*interval,
                6+0*interval, 6+1*interval, 6+2*interval, 6+3*interval, 6+4*interval, 6+5*interval),
               ("cloud3\ngpu0", "cloud3\ngpu1", "cloud3\ncpu", "cloud1\ngpu0", "cloud1\ngpu1", "cloud1\ncpu",
                "cloud3\ngpu0", "cloud3\ngpu1", "cloud3\ncpu", "cloud1\ngpu0", "cloud1\ngpu1", "cloud1\ncpu",
                "cloud3\ngpu0", "cloud3\ngpu1", "cloud3\ncpu", "cloud1\ngpu0", "cloud1\ngpu1", "cloud1\ncpu"))
    plt.legend((p1[0], p1[1], p1[2]), ("ESync", "Sync", "Async"), fontsize=fontsize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base-dir", type=str, default="/Users/mac/Desktop/ESync/logs/")
    parser.add_argument("-n", "--network", type=str, default="resnet18-v1")
    args, unknown = parser.parse_known_args()

    base_dir = os.path.join(args.base_dir, args.network)
    summary_name_dict = {
        "esync": "ESync.json",
        "esync-niid": "ESync-Non-IID.json",
        "sync": "Sync.json",
        "sync-niid": "Sync-Non-IID.json",
        "async": "Async.json",
        "async-niid": "Async-Non-IID.json"
    }

    fontsize = 12

    # I.I.D.
    esync_summary = load_summary(os.path.join(base_dir, "esync"), summary_name_dict["esync"])
    sync_summary = load_summary(os.path.join(base_dir, "sync"), summary_name_dict["sync"])
    async_summary = load_summary(os.path.join(base_dir, "async"), summary_name_dict["async"])

    config = [("cloud3", "gpu0"), ("cloud3", "gpu1"), ("cloud3", "gpu0"), True]
    draw_accuracy(esync_summary, sync_summary, async_summary, config,
                  vline=0.926, fignum=0, down_sample_interval=5, smooth_interval=10, shift=0, timespan=120)
    draw_data_throughput(esync_summary, sync_summary, async_summary, fignum=1)
    draw_traffic_load(esync_summary, sync_summary, async_summary, config, fignum=2)
    draw_computation_communication_ratio(esync_summary, sync_summary, async_summary, fignum=3)

    # Non I.I.D.
    esync_niid_summary = load_summary(os.path.join(base_dir, "esync-niid"), summary_name_dict["esync-niid"])
    sync_niid_summary = load_summary(os.path.join(base_dir, "sync-niid"), summary_name_dict["sync-niid"])
    async_niid_summary = load_summary(os.path.join(base_dir, "async-niid"), summary_name_dict["async-niid"])

    config = [("cloud3", "gpu1"), ("cloud3", "gpu0"), ("cloud3", "gpu1"), False]
    draw_accuracy(esync_niid_summary, sync_niid_summary, async_niid_summary, config,
                  vline=0.926, fignum=4, down_sample_interval=10, smooth_interval=30, shift=7)

    plt.show()
