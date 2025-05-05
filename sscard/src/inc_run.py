
import datetime

import gc
import sys
import os
import argparse
import logging
import time

import pandas as pd
import numpy as np
from pympler import asizeof

import sscard

from inc_suffix_tree import PatternTrie
from inc_suffix_tree import SuffixTree


def set_up_logging(filepath):
    logging.basicConfig(
        filename= filepath + 'output.log',  
        level=logging.INFO,
        format='%(message)s',
        filemode='w'
    )

def get_model_args():
    parser = argparse.ArgumentParser(description="parse")
    parser.add_argument('--dname', type=str, help='dataset name', required=True)
    parser.add_argument('--h', type=int, default=3, help='prune tree height', required=False)
    parser.add_argument('--buc', type=int, help='bucket size', default=1, required=False)
    parser.add_argument('--e', type=int, help='error bound of spline', default=32, required=False)
    parser.add_argument('--l', type=int, default=5000, help='prune node len')
    parser.add_argument('--fitting', type=str, default="spline", help='fitting method, linear(MSE) or spline(ME)')
    parser.add_argument('--load_L', type=str, default="False", help='No suffix tree, only fitting L.')
    parser.add_argument('--pushup', type=str, default="True", help='If false, building fitting functions all on leaves.')
    parser.add_argument('--add_info', type=str, default=None, help='any additional info')
    # parser.add_argument('--details', type=str, default="False", help='Test with every inserted data.')
    parser.add_argument('--cache_space', type=float, required=True, help='The space for incremental suffix tree (MB)')
    parser.add_argument('--inc_h', type=int, default=5, help='Incremental Suffix tree height')
    parser.add_argument('--only_query', type=str, default="False", help='Already built, test for query.')
    
    args = parser.parse_args()
    args.load_L = (args.load_L in ("True", "true"))
    args.pushup = (args.pushup in ("True", "true"))
    args.only_query = (args.only_query in ("True", "true"))
    
    args.data_path = "../../datasets/" + args.dname + "/"
    args.exp_name = f"SSCard_cache{args.cache_space}"

    args.exp_name += f"_{args.inc_h}"

    if args.add_info is not None:
        args.exp_name += "_" + args.add_info
    
    args.exp_path = "../inc_exp/" + args.dname + "/" + args.exp_name + "/"
    args.load_path = "../exp/" + args.dname + "/"
    
    # if args.dname == "test":
    #     args.dname = "data_for_demo"
        
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
    
    set_up_logging(args.exp_path)
    
    return args
        

def save_exp_results(args, result_name, results):
    with open(os.path.join(args.exp_path, result_name + ".txt"), "a") as f:
        for key, value in results.items():
            f.write(str(key) + ": " + str(value) + "\n")


def get_size(filepath):
    size = os.path.getsize(filepath)
    return size / 1024 / 1024


def inc_main(args):
    # load data strings
    data = []
    filename = args.data_path + args.dname + ".csv"
    with open(filename, 'r') as f:
        sum_len = 0
        for line in f.readlines():
            s = line.strip()
            # cnt += 1
            data.append(s)
            sum_len += len(s)
    
    # load pattern strings
    query_filename = args.data_path + args.dname + "_test_new.csv"
    df = pd.read_csv(query_filename, na_values=[], keep_default_na=False)
    pattern = df['string'].astype(str).tolist()
    # num = df['selectivity'].astype(float).tolist()
    
    num = np.zeros(len(pattern), dtype=float)
    
    Larger = 0
    Smaller = 0
    
    pattern_trie = PatternTrie(pattern)
    
    tree = SuffixTree(args.inc_h)
    rebuilt_times = 0
    
    cached_sta = 0
    cached_end = 0
    
    inc_q_error = []
    estimator1 = None
    estimator2s = []
    for i, s in enumerate(data):
        tree.insert_string(s)
        cached_end = i + 1
        
        idx_list = pattern_trie.query_datastring(s)
        # print(sorted(idx_list))
        for idx in idx_list:
            num[idx] += 1
        
        # tree_size =  asizeof.asizeof(tree) / 1024 / 1024
        if tree.total_nodes > args.cache_space:
        # if tree_size > args.cache_space:
            print(f"========={i}/{len(data)} Rebuild=========")
            logging.info(f"========={i}/{len(data)} Rebuild=========")
            if args.only_query == False:
                sta1 = time.time()
                e1 = sscard.SSCard(data[:cached_end], args)
                end1 = time.time()
                sta2 = time.time()
                e2 = sscard.SSCard(data[cached_sta:cached_end], args)
                end2 = time.time()
                cached_sta = cached_end
                print(f"Method 1 rebuild time:{end1 - sta1}")
                logging.info(f"Method 1 rebuild time:{end1 - sta1}")
                print(f"Method 2 rebuild time:{end2 - sta2}")
                logging.info(f"Method 2 rebuild time:{end2 - sta2}")

                sscard.save(os.path.join(args.exp_path, f"Total_rebuild{rebuilt_times}.pkl"), e1)
                sscard.save(os.path.join(args.exp_path, f"Incremental_rebuild{rebuilt_times}.pkl"), e2)
        
            tree.del_tree(tree.root)
            tree = SuffixTree(args.inc_h)
            rebuilt_times += 1

        # if i % 30000 == 0:
        if (i + 1) % 30000 == 0:        # test every 30000 data strings
            # tree_size =  asizeof.asizeof(tree) / 1024 / 1024
            # print(f"Tree nodes {tree.total_nodes}:\tSize {tree_size}")
            # logging.info(f"Tree nodes {tree.total_nodes}:\tSize {tree_size}")
            print(f"Tree nodes {tree.total_nodes}")
            logging.info(f"Tree nodes {tree.total_nodes}")
            positive_idxs = np.where(num > 0)[0]
            positive_pattern = [pattern[x] for x in positive_idxs]
            positive_num = [num[x] for x in positive_idxs]
            
            # counts_tree = np.zeros(len(positive_num), dtype=float)
            counts1 = np.zeros(len(positive_num), dtype=float)
            counts2 = np.zeros(len(positive_num), dtype=float)
            latencies1 = np.zeros(len(positive_num), dtype=float)
            latencies2 = np.zeros(len(positive_num), dtype=float)
            
            
            if rebuilt_times > 0:
                
                gc.disable()
                sta1 = time.time()
                estimator1 = sscard.load(os.path.join(args.exp_path, f"Total_rebuild{rebuilt_times - 1}.pkl"))
                end1 = time.time()
                
                
                duration2 = 0
                sta2 = time.time()
                estimator2s = []
                for j in range(rebuilt_times):
                    # sta2 = time.time()
                    e = sscard.load(os.path.join(args.exp_path, f"Incremental_rebuild{j}.pkl"))
                    estimator2s.append(e)
                    # end2 = time.time()
                    # duration2 += end2 - sta2
                end2 = time.time()
                duration2 = end2 - sta2
                
                tp = f"{i + 1}/{len(data)}:"
                print(f"{tp:<18}{(end1 - sta1):<20}:{(end2 - sta2):<20}")
                logging.info(f"{tp:<18}{(end1 - sta1):<20}:{(duration2):<20}")
                gc.enable()
            
            for j, (p, n) in enumerate(zip(positive_pattern, positive_num)):
                sta = time.time()
                c = tree.query(p)
                # counts_tree[j] = c
                end = time.time()
                
                counts1[j] = c
                counts2[j] = c

                latencies1[j] = end - sta
                latencies2[j] = end - sta
                if rebuilt_times > 0:
                    sta1 = time.time()
                    c, _ = estimator1.estimate_for_single_pattern_with_suffix_tree(p, "sscard_tree")
                    counts1[j] += c
                    end1 = time.time()
                    latencies1[j] += end1 - sta1
                    sta2 = time.time()
                    # counts2[j] = counts_tree[j]
                    for k in range(rebuilt_times):
                        c, _ = estimator2s[k].estimate_for_single_pattern_with_suffix_tree(p, "sscard_tree")
                        counts2[j] += c
                    end2 = time.time()
                    latencies2[j] += end2 - sta2
                
            counts1 = [max(1, c) for c in counts1]
            counts2 = [max(1, c) for c in counts2]
            
            q_error1 = [max(c / n, n / c) for c, n in zip(counts1, positive_num)]
            q_error2 = [max(c / n, n / c) for c, n in zip(counts2, positive_num)]
            q1 = np.mean(q_error1)
            q2 = np.mean(q_error2)
            l1 = np.mean(latencies1)
            l2 = np.mean(latencies2)
            inc_q_error.append((q1, q2))
            tp = f"{i + 1}/{len(data)}:"
            print(f"{tp:<18}{q1:<20}:{q2:<20}:{l1:<24}:{l2:<24}")
            logging.info(f"{tp:<18}{q1:<20}:{q2:<20}:{l1:<24}:{l2:<24}")
            # print(f"{tp:<18}{l1:<20}:  {l2:<20}")
            # logging.info(f"{tp:<18}{l1:<20}:  {l2:<20}")


    
if __name__ == '__main__':
    args = get_model_args()
    inc_main(args)