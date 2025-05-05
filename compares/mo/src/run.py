import sys
import os
import datetime
import pickle
import argparse
import pandas as pd
import time
import numpy as np
import math

from mo import PrunedSuffixTree, mo_estimate


def get_model_args():
    parser = argparse.ArgumentParser(description="parse")
    parser.add_argument('--dname', type=str, help='dataset name', required=True)
    parser.add_argument('--top_k_percent', type=int, help='prune top-k percent fo suffix tree', required=True)
    parser.add_argument('--only_query', type=str, default="False", help='Already built, test for query.')
    parser.add_argument('--add_info', type=str, default=None, help='any additional info')

    args = parser.parse_args()
    args.only_query = (args.only_query in ("True", "true"))
    
    args.dataset_path = "../../../datasets/" + args.dname + "/"
    # if args.add_info is not None:
    #     args.exp_path = "../exp/" + args.dname + "_" + args.add_info + "/"
    # else:
    #     args.exp_path = "../exp/" + args.dname + "/"
    if args.add_info is not None:
        args.exp_path = f"../exp/{args.dname}/top{args.top_k_percent}_{args.add_info}/"
    else:
        args.exp_path = f"../exp/{args.dname}/top{args.top_k_percent}/"

    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)
    else:
        flg = input("Experiment already exists, rewrite? (press y) ")
        if flg != 'y':
            print("skipped.")
            exit()

    with open(os.path.join(args.exp_path, "args_build.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")
    
    return args


def save(filename, idx):
    parent_directory = os.path.dirname(filename)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    f = open(filename, 'wb')
    pickle.dump(idx,f)

def load(filename):
    f = open(filename, "rb")
    ret = pickle.load(f)
    return ret
    
def get_size(filepath):
    size = os.path.getsize(filepath)
    return size / 1024 / 1024

def save_exp_results(path, result_name, results):
    with open(os.path.join(path, result_name + ".txt"), "w") as f:
        for key, value in results.items():
            f.write(str(key) + ": " + str(value) + "\n")

def write_results_to_file(file, strings, actuals, estimates, latencies):
    with open(file, "w") as f:
        f.write("s_q, ground_truth, estimate, latency\n")
        for i, (s_q, ground_truth, estimate, latency) in \
            enumerate(zip(strings, actuals, estimates, latencies)):
                
            f.write(s_q + "," + str(ground_truth) + ',')
            f.write(str(estimate) + ',' + str(latency) + '\n')


def build(args):
    
    data = []
    sum_len = 0
    filename = args.dataset_path + args.dname + ".csv"
    with open(filename, 'r') as f:
        
        for line in f.readlines():
            s = line.strip()
            data.append(s)
            sum_len += len(s)
    
    print(f"Total data len:{len(data)}")
    print(f"Total strings len:{sum_len}")
    
    
    # exit()

    building_results = {}
    
    # data = data[:10000]
    
    sta = datetime.datetime.now()
    pst = PrunedSuffixTree(len(data))
    for s in data:
        pst.insert(s)
    building_results["Full suffix tree node count"] = pst.total_node
    print(f"Full suffix tree node count: {pst.total_node}")
    pst.prune_by_top_k_percent(top_k_percent=args.top_k_percent)
    end = datetime.datetime.now()
    print(f"Building time: {end - sta}")
    print(f"PST node count: {pst.total_node}")

    building_results["PST node count"] = pst.total_node
    building_results["Building time"] = end - sta
    
    save(args.exp_path + "model.pkl", pst)
    model_size = get_size(args.exp_path + "model.pkl")
    building_results["Model size"] = model_size
    print(f"Mo size: {model_size}")
    save_exp_results(args.exp_path, "building_results", building_results)  
    
    del pst


def query(args):
    pattern = []
    num = []
    query_filename = args.dataset_path + args.dname + "_test_new.csv"
    
    df = pd.read_csv(query_filename, na_values=[], keep_default_na=False)
    # df = pd.read_csv(query_filename, dtype={'string': str, 'selectivity': float})
    pattern = df['string'].astype(str).tolist()
    num = df['selectivity'].astype(float).tolist()
    
    pst = load(args.exp_path + "model.pkl")

    t_count1 = datetime.datetime.now()

    latencies = []
    counts = np.zeros_like(num)
    for i, (p, n) in enumerate(zip(pattern, num)):
        sta = time.time()
        c = mo_estimate(pst, p)
        if args.dname != 'WIKI':
            c = max(1, c)
        counts[i] = c
        end = time.time()
        latencies.append(end - sta)
        
        tp = max(c / n, n / c)
        if tp > 5000:
            print(p, n, c, tp)

    t_count2 = datetime.datetime.now()
    write_results_to_file(args.exp_path + "analyses.csv", pattern, num, counts, latencies)

    if args.dname == 'WIKI': return
    
    q_error = []
    for i, (c, n) in enumerate(zip(counts, num)):
        q_error.append(max(c / n, n / c))
    
    MAE = np.mean([math.fabs(y1 - y2) for (y1, y2) in zip(counts,num)])
    
    print("MAE:", MAE)
    print(f"Avg: {np.mean(q_error)} Percentile: [0.5, 0.9, 0.99]: \
        {[np.quantile(q_error, q) for q in [0.5, 0.9, 0.99]]} Max:{np.max(q_error)}")
    print(f"Count time(substring):{t_count2 - t_count1}")  
    print("Avg. query time(s)", np.mean(latencies))
    
    query_results = {}
    query_results["Query time"] = t_count2 - t_count1
    query_results["MAE"] = MAE
    query_results["Avg q-error"] = np.mean(q_error)
    query_results["Percentile: [0.5, 0.7, 0.9, 0.99]"] = [np.quantile(q_error, q) for q in [0.5, 0.7, 0.9, 0.99]]
    query_results["Max q-error"] = np.max(q_error)
    query_results["Avg. query time(s)"] = np.mean(latencies)
    save_exp_results(args.exp_path, "query_results", query_results) 


def divide_dataset(args, k = 5):
    data = []
    sum_len = 0
    filename = args.dataset_path + args.dname + ".csv"
    with open(filename, 'r') as f:
        
        for line in f.readlines():
            s = line.strip()
            data.append(s)
            sum_len += len(s)
    
    # data = data[:100]
    
    l, m = divmod(len(data), k)
    divided_data = [data[i*l + min(i, m):(i+1)*l + min(i+1, m)] for i in range(k)]
    
    for i in range(k):
        filename = args.dataset_path + args.dname + f"_{i}.csv"
        with open(filename, 'w') as f:
            for s in divided_data[i]:
                f.write(s + '\n')


def load_estimation_results(filename):
    strings = []
    sels = []
    counts = []
    latencies = []
    with open(filename, "r") as f:
        first_line = True
        for line in f.readlines():
            if first_line:
                first_line = False
                continue
            tmp_list = line.strip().rsplit(',', 3)
            sels.append(float(tmp_list[1]))
            counts.append(float(tmp_list[2]))
            latencies.append(float(tmp_list[3]))
            strings.append(tmp_list[0])
    return strings, sels, counts, latencies


def summarize_results(args, old_dname, K=5):
    pattern = []
    num = []
    query_filename = args.dataset_path + old_dname + "_test_new.csv"
    
    df = pd.read_csv(query_filename, na_values=[], keep_default_na=False)
    # df = pd.read_csv(query_filename, dtype={'string': str, 'selectivity': float})
    pattern = df['string'].astype(str).tolist()
    num = df['selectivity'].astype(float).tolist()
    
    counts = np.zeros(len(num), dtype=float)
    latencies = np.zeros(len(num), dtype=float)
    
    print(args.exp_path)
    for i in range(K):
        if args.add_info is not None:
            result_file = f"{args.exp_path[:12]}{i}_top{args.top_k_percent}_{args.add_info}/analyses.csv"
        else:
            result_file = f"{args.exp_path[:12]}{i}_top{args.top_k_percent}/analyses.csv"
        _, _, cs, lats = load_estimation_results(result_file)
        counts = [x + y for x, y in zip(counts, cs)]
        latencies = [x + y for x, y in zip(latencies, lats)]
    
    counts = [max(x, 1) for x in counts]
    
    if args.add_info is not None:
        args.exp_path = f"{args.exp_path[:12]}all_top{args.top_k_percent}/"
    else:
        args.exp_path = f"{args.exp_path[:12]}all_top{args.top_k_percent}_{args.add_info}/"
    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)
    
    write_results_to_file(args.exp_path + "analyses.csv", pattern, num, counts, latencies)
    
    q_error = []
    for i, (c, n) in enumerate(zip(counts, num)):
        q_error.append(max(c / n, n / c))
    
    MAE = np.mean([math.fabs(y1 - y2) for (y1, y2) in zip(counts,num)])
    
    print("==========Total result==========")
    print("MAE:", MAE)
    print(f"Avg: {np.mean(q_error)} Percentile: [0.5, 0.9, 0.99]: \
        {[np.quantile(q_error, q) for q in [0.5, 0.9, 0.99]]} Max:{np.max(q_error)}")
    print("Avg. query time(s)", np.mean(latencies))
    
    query_results = {}
    query_results["MAE"] = MAE
    query_results["Avg q-error"] = np.mean(q_error)
    query_results["Percentile: [0.5, 0.7, 0.9, 0.99]"] = [np.quantile(q_error, q) for q in [0.5, 0.7, 0.9, 0.99]]
    query_results["Max q-error"] = np.max(q_error)
    query_results["Avg. query time(s)"] = np.mean(latencies)
    save_exp_results(args.exp_path, "query_results", query_results) 


if __name__ == '__main__':
    args = get_model_args()
    
    if args.dname == 'WIKI':
        K = 10
        # divide_dataset(args, k=K)
    # exit()
    
    if args.dname == 'WIKI':
        old_dname = args.dname
        for i in range(K):
            print(f"=========={i+1}/{K}==========")
            args.dname = old_dname + f"_{i}"
            if args.add_info is not None:
                args.exp_path = f"../exp/{old_dname}/{i}_top{args.top_k_percent}_{args.add_info}/"
            else:
                args.exp_path = f"../exp/{old_dname}/{i}_top{args.top_k_percent}/"
            # print(args.exp_path)
            if args.only_query == False:
                build(args)
            args.dname = old_dname
            query(args)
            if (i + 1) % 5 == 0:
                summarize_results(args, old_dname, i + 1)
        summarize_results(args, old_dname, K)
        
    else:    
        if args.only_query == False:
            build(args)
        
        query(args)