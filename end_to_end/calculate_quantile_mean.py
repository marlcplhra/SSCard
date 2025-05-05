import sys
import logging
import os
import numpy as np


def setup_logging(log_fn):
    dir_path = os.path.dirname(log_fn)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # log_fn = "./logs/db_api_results_serial_new2.txt"
    # log_fn = "./results_test.txt"
    logging.basicConfig(filename=log_fn, filemode='w', format='%(message)s',level=logging.INFO)
    # also log into console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    print(f"setup logging to {log_fn}.")


def read_results(query_set, mname, f, t):
    filename = f"./pg_exp/{query_set}/{mname}/exe_time_{f}_{t}.txt"
    results = dict()
    with open(filename, "r") as f:
        for line in f.readlines():
            tp = line.strip().split(":")
            if tp[0] == 'Total time': break
            # if tp[0] == '4a': continue
            results[tp[0]] = float(tp[1].strip())
    return results


def add(d, key, val):
    if key not in d:
        d[key] = [val]
    else:
        d[key].append(val)


def calculate_quantile_mean(query_set, fname, times):
    print(fname)
    r = dict()
    sum_time = 0
    for t in range(times):
        if fname == "pg":
            results = read_results(query_set, "true", fname + "_pg_single",t)
        else:
            results = read_results(query_set, fname, fname + "_pg_single",t)
        for key in results:
            add(r, key, results[key])
    etimes = []
    for key in r:
        tp = np.mean(r[key])
        # if tp > 10000: print(key)
        etimes.append(tp)
        sum_time += tp
    etimes = sorted(etimes)
    # print(etimes)
    quantiles = np.quantile(etimes, [0.5, 0.9, 0.99, 1.0]) / 1000
    print("Mean: {:.2f}".format(np.mean(etimes) / 1000), end = " ")
    print("")
    print("Quantile: [0.5, 0.9, 0.99, 1.0]: ", end="")
    for x in quantiles:
        print("{:.2f}".format(x), end=" ")
    print("")
    print("Sum: {:.2f}".format(sum_time / 1000))
    

if __name__ == "__main__":
    
    
    # mnames = ["pg", "mo", "lbs", "astrid", "dream", "sscard", "true"]
    mnames = ["sscard"]
    for mname in mnames:
        calculate_quantile_mean("like_queries_single", mname, 3)
    # calculate_quantile_mean(sys.argv[1], sys.argv[2], int(sys.argv[3]))