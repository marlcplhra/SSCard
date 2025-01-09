import os
import sys
import datetime

import random
import numpy as np
import argparse
import math
import time
import pandas as pd

import sscard
    

def get_model_args():
    parser = argparse.ArgumentParser(description="parse")
    parser.add_argument('--dname', type=str, help='dataset name', required=True)
    parser.add_argument('--h', type=int, default=5, help='prune tree height', required=True)
    parser.add_argument('--buc', type=int, help='bucket size, only used for linear fitting', default=1, required=False)
    parser.add_argument('--e', type=int, help='error bound of spline', default=16, required=False)
    parser.add_argument('--l', type=int, default=None, help='prune node len')
    parser.add_argument('--fitting', type=str, default="linear", help='fitting method, linear(MSE) or spline(MAE)')
    parser.add_argument('--note', type=str, default=None)
    parser.add_argument('--pushup', type=str, default="True", help='If false, building fitting functions all on leaves.')
    
    args = parser.parse_args()
    args.pushup = (args.pushup in ("True", "true"))
    
    args.data_path = "../datasets/" + args.dname + "/"
    args.exp_name = "SSCard"
    args.exp_name += "_buc" + str(args.buc)
    args.exp_name += "_h" + str(args.h)
    if args.l is None:
        args.l = args.buc * 10
        args.exp_name += "_lx10buc"
        # args.l = args.buc
        # args.exp_name += 'lx1buc'
    else:
        args.exp_name += "_l" + str(args.l)
    if args.e is not None:
        args.exp_name += "_e" + str(args.e)
    args.exp_name += "_" + args.fitting
    
    if args.note is not None:
        args.exp_name += "_" + args.note

    if args.pushup == False:
        args.exp_name += "_no_pushup"
    
    args.exp_path = "../exp/" + args.dname + "/" + args.exp_name + "/"
        
    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)
    else:
        flg = input("Experiment already exists, rewrite? (press y) ")
        if flg != 'y':
            print("skipped.")
            exit()
    print("Experiment name:", args.exp_name)
    with open(os.path.join(args.exp_path, "args_build.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")
    return args
        

def save_exp_results(args, result_name, results):
    with open(os.path.join(args.exp_path, result_name + ".txt"), "w") as f:
        for key, value in results.items():
            f.write(str(key) + ": " + str(value) + "\n")


def get_size(filepath):
    size = os.path.getsize(filepath)
    return size / 1024 / 1024


def write_results_to_file(file, strings, actuals, estimates, latencies):
    with open(file, "w") as f:
        f.write("s_q, ground_truth, estimate, latency\n")
        for i, (s_q, ground_truth, estimate, latency) in \
            enumerate(zip(strings, actuals, estimates, latencies)):
                
            f.write(s_q + "," + str(ground_truth) + ',')
            f.write(str(estimate) + ',' + str(latency) + '\n')


def build_sscard(args):
    data = []
    
    sta = datetime.datetime.now()
    
    cnt = 0
    building_results = {}
    
    filename = args.data_path + args.dname + ".csv"
    with open(filename, 'r') as f:
        sum_len = 0
        for line in f.readlines():
            s = line.strip()
            cnt += 1
            data.append(s)
            sum_len += len(s)
    
    # data = data[:2]
    # print(data)
    idx_num = 0
    print("string num:", len(data))
    print("data len:", sum_len + len(data))
    building_results["data len" + str(idx_num)] = sum_len
    filename = args.exp_name + "_" + str(idx_num)
    
    
    estimator = sscard.build_estimator(data, args)
    sscard.save(os.path.join(args.exp_path, filename), estimator)
    building_results["Constructing L array time" + str(idx_num)] = estimator.L_time
    if hasattr(estimator, "myTree_multi"):
        building_results["Tree node" + str(idx_num)] = estimator.myTree_multi.total_node  
    building_results["Constructing spline time"] = estimator.spline_time
    model_size = get_size(os.path.join(args.exp_path, filename))
    building_results["Model size" + str(idx_num)] = model_size
    idx_num += 1

    end = datetime.datetime.now()
    print(f"Building time: {end - sta}")
    
    building_results["Building time"] = end - sta
    save_exp_results(args, "building_results", building_results)  
    

def query_sscard(args):
    idx_num = 1
    # query
    pattern = []
    num = []
    query_filename = args.data_path + args.dname + "_test.csv"
    
    df = pd.read_csv(query_filename, na_values=[], keep_default_na=False)
    # df = pd.read_csv(query_filename, dtype={'string': str, 'selectivity': float})
    pattern = df['string'].astype(str).tolist()
    num = df['selectivity'].astype(float).tolist()
    
    t_count1 = datetime.datetime.now()
    counts = np.zeros_like(num)
    estimators = []
    for i in range(idx_num):
        print(f"Loading {i}th model:")
        filename = args.exp_name + "_" + str(i)
        print(os.path.join(args.exp_path, filename))
        estimator = sscard.load(os.path.join(args.exp_path, filename))
            
        estimators.append(estimator)
    
    latencies = []
    t_count1 = datetime.datetime.now()
    for i, (p, n) in enumerate(zip(pattern, num)):
        sta = time.time()
        for j, estimator in enumerate(estimators):
            c, flg = estimator.estimate_for_single_pattern_with_suffix_tree(p, "likecard_tree")
                    
            counts[i] += c
        end = time.time()
        latencies.append(end - sta)
        assert n > 0
        
        if i % 10000 == 0:
            print(f"{i}/{len(num)}")
        tp = max(counts[i] / n, n / counts[i])
            
    t_count2 = datetime.datetime.now()
    
    write_results_to_file(args.exp_path + "analyses.csv", pattern, num, counts, latencies)

    results = [(p, n, c) for (p, n, c) in zip(pattern, num, counts)]

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
    save_exp_results(args, "query_results", query_results) 



def main():
    random.seed(1234)
    
    args = get_model_args()

    build_sscard(args)
    
    query_sscard(args)
        

if __name__ == '__main__':
    main()