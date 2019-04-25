import os
import json
import argparse


def collect_report(path):
    iter_idx = 1
    merged = {}
    while True:
        filename = os.path.join(path, "iter-%d.txt" % iter_idx)
        if not os.path.exists(filename):
            iter_idx -= 1
            break
        with open(filename, "r") as fp:
            merged[iter_idx] = json.load(fp)
        iter_idx += 1
    print("%d reports collected." % iter_idx)
    return merged


def summary_report(data):
    summary = {}
    for node in data.keys():
        summary[node] = {}
        for worker in data[node].keys():
            summary[node][worker] = {
                "accuracy_list": [],
                "num_samples_list": [],
                "self_iters_list": [],
                "elapsed_time_list": []
            }

    for node in data.keys():
        for worker in data[node].keys():
            num_samples_list = []
            self_iters_list = []
            accuracy_list = []
            elapsed_time_list = []
            aggregate_time_list = []
            for iter_idx, val in data[node][worker].items():
                num_samples_list.append(val["num_samples"])
                self_iters_list.append(len(val["time"]))
                if "accuracy" in val:
                    acc = round(val["accuracy"], 3)
                    accuracy_list.append(acc)
                if "total_time" in val:
                    total_time = int(round(val["total_time"]))
                    elapsed_time_list.append(total_time)
                max_iter = str(max(map(int, val["time"].keys())))
                aggregate_key = "global update" if "global update" in val["time"][max_iter] else "global aggregate"
                aggregate_time_list.append(val["time"][max_iter][aggregate_key])
            summary[node][worker]["accuracy_list"] = accuracy_list
            summary[node][worker]["num_samples_list"] = num_samples_list
            summary[node][worker]["self_iters_list"] = self_iters_list
            summary[node][worker]["elapsed_time_list"] = elapsed_time_list
            summary[node][worker]["aggregate_time_list"] = aggregate_time_list

    return summary


def save_summary(summary, summary_path, summary_name):
    summary_file = os.path.join(summary_path, summary_name)
    with open(summary_file, "w") as fp:
        json.dump(summary, fp)


def load_summary(summary_path, summary_name):
    summary_file = os.path.join(summary_path, summary_name)
    if not os.path.exists(summary_file):
        return None
    with open(summary_file, "r") as fp:
        summary = json.load(fp)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base-dir", type=str, default="/Users/mac/Desktop/ESync/logs/")
    parser.add_argument("-n", "--network", type=str, default="resnet18-v1")
    parser.add_argument("-m", "--mode", type=str, default="esync")
    args, unknown = parser.parse_known_args()

    summary_name_dict = {
        "esync": "ESync.json",
        "esync-niid": "ESync-Non-IID.json",
        "sync": "Sync.json",
        "sync-niid": "Sync-Non-IID.json",
        "async": "Async.json",
        "async-niid": "Async-Non-IID.json"
    }

    base_dir = os.path.join(args.base_dir, args.network, args.mode)
    summary_name = summary_name_dict[args.mode]

    merged = {}
    workers = {"cloud1": ["cpu", "gpu0", "gpu1"], "cloud3": ["cpu", "gpu0", "gpu1"]}
    for node in workers.keys():
        merged[node] = {}
        for worker in workers[node]:
            path = os.path.join(base_dir, node, worker)
            merged[node][worker] = collect_report(path)
    summary = summary_report(merged)
    save_summary(summary, base_dir, summary_name)
