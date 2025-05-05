### Run SSCard

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

```
cd sscard/src
python plot_SSCard.py
```





### Incremental Update

---

Run the following command to do the experiment for incremental update in SSCard:

```
cd sscard/src
python inc_run.py --dname=<dataset_name> --cache_space=<max_tree_nodes> --inc_h=<h> --only_query=<only_query>
```

- `<dataset_name>`: name of the dataset, we take `DBLP_AN` for incremental update experiments.
- `<max_tree_nodes>`: the maximum number of tree nodes for the incremental suffix tree.
- `<h>`: the height for the incremental suffix tree,
- `<only_query>`: default is False, if True, we will load the existing model and only evaluate it on the test set.

For example, run incremental update experiment with the same hyperparameter as the paper:

```
cd sscard/src
python inc_run.py --dname=DBLP_AN --cache_space=250000 --inc_h=5  --only_query=False
```

