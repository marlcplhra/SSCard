## Run SSCard

---

Run the following command to build SSCard model and test its performance on a dataset:

```bash
cd sscard/src
python run.py --dname=<dataset_name> --h=<tree_height> --buc=<buc_size> --l=<segment_size> --e=<error_bound> --fitting=<fitting_method> --load_L=<Load_L> --only_query=<only_query>
```

- `<dataset_name>`: name of a dataset (`DBLP_AN`, `IMDB_AN`, `IMDB_MT`, `TPCH_PN`, or `WIKI`).
- `<tree_height>`: the maximum height of the suffix tree.
- `<buc_size>`: the bucket size for a linear fitting function, only used for linear fitting.
- `<segment_size>`: segment size of the pruned suffix tree.
- `<error_bound>`: the error bound for greedy spline interpolation.
- `<fitting>`: fitting method (`spline` or `linear`).
- `<load_L>`: default is False, if True, SSCard will load the L-array from file instead of sorting all cyclic shifts. This can be set for testing other components in SSCard, because sorting all cyclic shifts takes relatively a long time. 
- `<only_query>`: default is False, if True, we will load the existing model and only evaluate it onthe test set.



For example, run SSCard on DBLP-AN dataset with the same hyperparameter as the paper:

```bash
cd sscard/src
python run.py --dname=DBLP_AN --h=3 --buc=1 --l=5000 --e=32 --fitting=spine --load_L=False
```



Also, you can replot the figures in the paper:

```bash
cd sscard/src
python plot_SSCard.py
```





## Incremental Update

---

Run the following command to do the experiment for incremental update in SSCard:

```bash
cd sscard/src
python inc_run.py --dname=<dataset_name> --cache_space=<max_tree_nodes> --inc_h=<h> --only_query=<only_query>
```

- `<dataset_name>`: name of the dataset, we take `DBLP_AN` for incremental update experiments.
- `<max_tree_nodes>`: the maximum number of tree nodes for the incremental suffix tree.
- `<h>`: the height for the incremental suffix tree,
- `<only_query>`: default is False, if True, we will load the existing model and only evaluate it on the test set.

For example, run incremental update experiment with the same hyperparameter as the paper:

```bash
cd sscard/src
python inc_run.py --dname=DBLP_AN --cache_space=250000 --inc_h=5  --only_query=False
```





## Run other competitors

---

As described in the paper, we compared SSCard with the state-of-the-art methods, including:

- MO [PODS1999]: a suffix tree based method.
- LBS [EDBT2009]: a $k$-gram based method.
- Astrid [VLDB2020]: a neural model that learns selectivity-aware embeddings of substrings from the data strings.
- DREAM [VLD2022]: the SOTA estimator for approximate string queries.
- LPLM [SIGMOD2024]: the SOTA estimator for LIKE predicates.

Except MO, the code are all from their corresponding Github repositories. We implement MO based on the original paper.

Run MO:
```
cd compares/mo
python run.py --dname=<dataset_name> --top_k_percent=2 --add_info=final --only_query=True
```

- `<dataset_name>`: name of a dataset (`DBLP_AN`, `IMDB_AN`, `IMDB_MT`, `TPCH_PN`, or `WIKI`). (The same applies below)

Run LBS:

```
cd compares/dream
python run.py --model LBS --dname <dataset_name> --p-test 0.1 --seed 0 --Ntbl 5 --PT 20 --max-d 0 --L 10
```

Run Astrid: 

```bash
cd compares/astrid
# prepare the counts.csv and triplets.csv for training
python prepare_datasets.py
python AstridEmbed.py <dataset_name>
```

Run DREAM:

```bash
cd compares/dream
run.py --model DREAM --dname <dataset_name> --seed 1234 --l2 0.00000001 --lr 0.001 --layer 1 --pred-layer 3 --cs 512 --max-epoch 100 --patience 10 --max-d 0 --max-char 200 --bs 128 --h-dim 512 --es 100 --clip-gr 10.0
```

Run LPLM:

```bash
cd compares/lplm
# compute the ground truth labels for the training data
python compute_ground_truth.py
python main.py <dataset_name>
```





## Compare SSCard with FM-Index

---

We also compare SSCard with the FM-Index. Since the [state-of-the-art implementation](https://github.com/simongog/sdsl-lite) of the FM-Index is written in C++, we additionally implement a C++ version of SSCard for a fair comparison.

Use  the following command to run SSCard in C++:
```bash
cd sscard_cpp/src
# compile
g++ -O3 -I ../cereal-1.3.2/include main.cpp sscard.cpp suffix_tree.cpp -o main
# run
./main <dataset_name> 128 3 5000 10 32
```

- Note that `<dataset_name>` are `DBLP_AN`, `IMDB_AN`, `IMDB_MT`, `TPCH_PN`, for WIKI dataset, we implement a version based on utf-8 encoding, use the following command to run SSCard (C++) on WIKI:

```
cd sscard_cpp_wiki/src
# compile
g++ -O3 -I ../cereal-1.3.2/include -I ../utfcpp/source main.cpp sscard.cpp suffix_tree.cpp -o main
# run
./main WIKI 3736 3 5000 10 32
```



For FM-Index, use the following command to run on various datasets:

```bash
cd fm_index
# compile
g++ -std=c++11 -O3 -DNDEBUG -I ~/include -L ~/lib test.cpp -o test -lsdsl -ldivsufsort -ldivsufsort64
# run
./test <dataset_name>
```





## Compare SSCard with gzip

---

We also compare the space consumption between SSCard and standard compression technique `gzip` in python.

Use the following command to run `gzip` in SSCard.

```
cd gzip/src
python run.py --dname=<dataset_name> --k=500 --only_query=False
```





## Injecting Estimated Cardinalities into PostgreSQL

---

We evaluate the end-to-end query execution time by injecting estimated cardinalities into PostgreSQL 14.5. We take IMDB as dataset and select 79 out of 116 queries of the JOB workload [34] that contains LIKE statement with the form $\%word\%$.

Please follow the instructions of [LPLM](https://github.com/dbis-ukon/lplm?tab=readme-ov-file) and [End-to-End-CardEst-Benchmark](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark) to modify the PostgreSQL codebase to accept estimated cardinalities.

(The data for the `cast_info.note` exceeds 100MB, so we have hosted it on OneDrive. Please download the file and place it in the `end_to_end/columns/cast_info.note` folder.)

After that first build SSCard on all the 11 string columns:

```
cd end_to_end/sscard/src
./run
```

And then run SSCard on the LIKE predicates to get the estimated cardinalities:

```
python cal_sel.py
```

This will generate an estimation result file at `end_to_end/cards/like_queries_single/sscard_pg_single.txt`. Please move `sscard_pg_single.txt` into the *data directory* of your PostgreSQL instance (e.g., `/var/lib/pgsql/14.5/data`). This ensures that PostgreSQL can access the SSCard estimation results. the following commands are:
```bash
cd ../..
sudo cp sscard_pg_single.txt /var/lib/pgsql/14.5/data
sudo -i
chown postgres:postgres mo_pg_single.txt
```

Finally, test the end-to-end query time on PostgreSQL:

```
./run_and_analyse
python calculate_quantile_mean.py
```

