#!/bin/bash


for i in {0..2};
do
  python run_job.py like_queries_single sscard_pg_single.txt $i
  python run_job.py like_queries_single pg_pg_single.txt $i
  mv ./pg_exp/like_queries_single/pg/exe_time_pg_pg_single_$i.txt ./pg_exp/like_queries_single/sscard
  mv ./pg_exp/like_queries_single/pg/log_pg_pg_single_$i.txt ./pg_exp/like_queries_single/sscard
  rm -rf ./pg_exp/like_queries_single/pg
  # cp ./pg_exp/like_queries_single/exp2/exe_time_pg_pg_single_$i.txt ./pg_exp/like_queries_single/sscard
done

python analyse_results.py like_queries_single sscard_pg_single pg_pg_single 3