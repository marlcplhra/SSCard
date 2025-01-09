### Building SSCard

---

Run the following command to build SSCard model and test its performance on a dataset:

```bash
 python run.py --dname=<data_name> --h=<tree_height> --buc=<buc_size> --l=<segment_size> --e=<error_bound> --fitting=<fitting_method>
```

- `<data_name>`: name of a dataset (`DBLP_AN`, `IMDB_AN`, `IMDB_MT`, `TPCH_PN`, or `WIKI`)
- `<tree_height>`: the maximum height of the suffix tree
- `<buc_size>`: the bucket size for a linear fitting function, only used for linear fitting
- `<segment_size>`: segment size of the pruned suffix tree
- `<error_bound>`: the error bound for greedy spline interpolation
- `<fitting>`: fitting method (`spline` or `linear`)