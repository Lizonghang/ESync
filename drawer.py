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


def draw_accuracy(summaries, config, vline=0.9,
                  fignum=0, timespan=120, down_sample_interval=10, smooth_interval=10, shift=0):
    plt.figure(fignum)
    iid = config[4]
    title = "Test Accuracy Curve of ESync, SSGD, ASGD, DC-ASGD on %si.i.d. dataset" % ("" if iid else "non-")
    plt.title(title, fontsize=fontsize)
    plt.xlabel("Time (minutes)", fontsize=fontsize)
    plt.ylabel("Test Accuracy", fontsize=fontsize)
    plt.xlim((0, timespan))
    plt.ylim((0, 1))

    colors = ("m", "orange", "b", "g")
    markers = ("p", "^", "s", "o")
    linewidth = 2
    linestyle = ("-", ":", "--", "-.")
    marker_sizes = (8, 7, 6, 7)
    if iid:
        marker_intervals = (32, 16, 32, 45)
        marker_prefix = [[5, 7, 9], [5], [7, 10], [13]]
        marker_starts = (15, 15, 40, 30)
    else:
        marker_intervals = (12, 14, 23, 24)
        marker_prefix = [[], [], [], []]
        marker_starts = (9, 12, 18, 40)

    for idx, summary in enumerate(summaries):
        node, worker = config[idx]
        if node == None or summary == None:
            continue
        accuracy_list = summary[node][worker]["accuracy_list"]
        elapsed_time_list = [s / 60. for s in summary[node][worker]["elapsed_time_list"]]
        smoothed, elapsed_time_list = down_sampling(accuracy_list, elapsed_time_list, down_sample_interval, shift)
        smoothed = [0.1] + smoothed
        elapsed_time_list = [0] + elapsed_time_list
        smoothed = interval_averaging(smoothed, smooth_interval)
        plt.plot(elapsed_time_list, smoothed,
                 marker=markers[idx], markersize=marker_sizes[idx],
                 markevery=marker_prefix[idx]+\
                           list(range(marker_starts[idx], len(elapsed_time_list), marker_intervals[idx])),
                 c=colors[idx], linewidth=linewidth, linestyle=linestyle[idx])

    plt.plot((0, timespan), (vline, vline), c="k", linestyle="--", linewidth=1)
    plt.legend(["ESync", "SSGD", "ASGD", "DC-ASGD", "Standalone"], fontsize=fontsize)


def draw_data_throughput(summaries, fignum=1):
    assert len(summaries) == 4

    plt.figure(fignum)
    colors = ("m", "orange", "b", "g")
    hatches = ("x", "/", "\\", "-")
    plt.title("Data Throughput of ESync, SSGD, ASGD, DC-ASGD", fontsize=fontsize)
    plt.xlabel("Algorithm", fontsize=fontsize)
    plt.ylabel("Data Throughput (samples per second)", fontsize=fontsize)
    plt.xticks((0, 1, 2, 3), ("ESync", "SSGD", "ASGD", "DC-ASGD"), fontsize=fontsize)

    throughputs = [0, 0, 0, 0]
    for idx, summary in enumerate(summaries):
        total_samples = 0
        total_time = 0
        for node in summary.keys():
            for worker in summary[node].keys():
                total_samples += sum(summary[node][worker]["num_samples_list"])
                if not total_time:
                    total_time = summary[node][worker]["elapsed_time_list"][-1]
        throughputs[idx] = int(total_samples / total_time)
        print("Data Throughput (samples per second):", throughputs[idx])

    for idx, tp in enumerate(throughputs):
        plt.bar(x=idx, height=tp, color="w", edgecolor=colors[idx], width=0.6, align="center", hatch=hatches[idx])


def draw_traffic_load(summaries, config, fignum=2):
    assert len(summaries) == 4

    plt.figure(fignum)
    colors = ("m", "orange", "b", "g")
    hatches = ("x", "/", "\\", "-")
    plt.title("Traffic Load of ESync, SSGD, ASGD, DC-ASGD", fontsize=fontsize)
    plt.xlabel("Algorithm", fontsize=fontsize)
    plt.ylabel("Traffic Load (MBytes per second)", fontsize=fontsize)
    plt.xticks((0, 1, 2, 3), ("ESync", "SSGD", "ASGD", "DC-ASGD"), fontsize=fontsize)

    # the number of parameters of ResNet18-v1: 10.65 Million
    model_size = 42.6
    num_workers = 6

    traffic_loads = [0, 0, 0, 0]
    for idx, summary in enumerate(summaries):
        if idx <= 1:
            node, worker = config[idx]
            communication_round = len(summary[node][worker]["accuracy_list"])
            data_size = communication_round * num_workers * model_size
            total_time = summary[node][worker]["elapsed_time_list"][-1]
        else:
            communication_round = 0
            total_time = 0
            for node in summary.keys():
                for worker in summary[node].keys():
                    communication_round += len(summary[node][worker]["self_iters_list"])
                    if not total_time:
                        total_time = summary[node][worker]["elapsed_time_list"][-1]
            data_size = communication_round * model_size
        traffic_loads[idx] = int(data_size / total_time)
        print("Traffic Load (MBytes per second):", traffic_loads[idx])

    for idx, tl in enumerate(traffic_loads):
        plt.bar(x=idx, height=tl, color="w", edgecolor=colors[idx], width=0.6, align="center", hatch=hatches[idx])


def draw_computing_time_ratio(summaries, fignum=3):
    assert len(summaries) == 4

    num_workers = 6
    num_modes = len(summaries)
    width = 0.5
    interval = 0.6
    colors = ("m", "orange", "b", "g")
    hatches = ("x", "/", "\\", "-")
    plt.figure(fignum)
    plt.title("Computing Time Ratio of ESync, SSGD, ASGD, DC-ASGD", fontsize=fontsize)
    plt.xlabel("Devices", fontsize=fontsize)
    plt.ylabel("Computing Time Ratio", fontsize=fontsize)
    plt.xticks([])

    comm_time = [{}, {}, {}, {}]
    for idx, summary in enumerate(summaries):
        for node in summary.keys():
            comm_time[idx][node] = {}
            for worker in summary[node].keys():
                total_comm_time = sum(summary[node][worker]["aggregate_time_list"])
                total_time = summary[node][worker]["elapsed_time_list"][-1]
                comm_time[idx][node][worker] = total_comm_time / total_time

    stack_bars = []
    for device in [("cloud3", "gpu0"), ("cloud3", "gpu1"), ("cloud3", "cpu"),
                   ("cloud1", "gpu0"), ("cloud1", "gpu1"), ("cloud1", "cpu")]:
        bars = []
        for i in range(num_modes):
            bars.append([1-comm_time[i][device[0]][device[1]]])
        stack_bars.append(bars)

    ps = []
    for i in range(num_workers):
        for j in range(len(summaries)):
            p = plt.bar(num_modes*j+i*interval, stack_bars[i][j],
                        color="w", edgecolor=colors[j], width=width, align="center", hatch=hatches[j])
            ps.append(p)

    plt.legend(ps[:num_modes], ("ESync", "SSGD", "ASGD", "DC-ASGD"), fontsize=fontsize)


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
        "async-niid": "Async-Non-IID.json",
        "dcasgd": "DCASGD.json",
        "dcasgd-niid": "DCASGD-Non-IID.json"
    }

    fontsize = 12

    # I.I.D.
    esync_summary = load_summary(os.path.join(base_dir, "esync"), summary_name_dict["esync"])
    sync_summary = load_summary(os.path.join(base_dir, "sync"), summary_name_dict["sync"])
    async_summary = load_summary(os.path.join(base_dir, "async"), summary_name_dict["async"])
    dcasgd_summary = load_summary(os.path.join(base_dir, "dcasgd"), summary_name_dict["dcasgd"])
    summaries = [esync_summary, sync_summary, async_summary, dcasgd_summary]

    config = [("cloud3", "gpu0"), ("cloud3", "gpu1"), ("cloud3", "gpu0"), ("cloud3", "gpu1"), True]
    draw_accuracy(summaries, config, vline=0.926, fignum=0, down_sample_interval=5, smooth_interval=10)
    draw_data_throughput(summaries, fignum=1)
    draw_traffic_load(summaries, config, fignum=2)
    draw_computing_time_ratio(summaries, fignum=3)

    # Non I.I.D.
    esync_niid_summary = load_summary(os.path.join(base_dir, "esync-niid"), summary_name_dict["esync-niid"])
    sync_niid_summary = load_summary(os.path.join(base_dir, "sync-niid"), summary_name_dict["sync-niid"])
    async_niid_summary = load_summary(os.path.join(base_dir, "async-niid"), summary_name_dict["async-niid"])
    dcasgd_niid_summary = load_summary(os.path.join(base_dir, "dcasgd-niid"), summary_name_dict["dcasgd-niid"])
    summaries = [esync_niid_summary, sync_niid_summary, async_niid_summary, dcasgd_niid_summary]

    config = [("cloud3", "gpu0"), ("cloud3", "gpu0"), ("cloud3", "gpu1"), ("cloud3", "gpu1"), False]
    draw_accuracy(summaries, config, vline=0.926, fignum=4, down_sample_interval=10, smooth_interval=30)

    plt.show()
