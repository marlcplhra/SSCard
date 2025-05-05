import pandas as pd
import numpy as np


def analyse_latency(name, path):
    print(name, ":", end = "")
    df = pd.read_csv(path)
    latencies = df['latency'].to_list()
    print(np.mean(latencies))
        
        
if __name__ == "__main__":
    analyse_latency("DREAM DBLP_AN", "./exp_result/DBLP_AN/DREAM_DBLP_AN_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_l2_1e-08_pat_10_clipGr_10.0_seed_1234_maxEpoch_100_maxD_0_bs_128/analysis.csv")
    analyse_latency("DREAM IMDB_AN", "./exp_result/IMDB_AN/DREAM_IMDB_AN_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_l2_1e-08_pat_10_clipGr_10.0_seed_1234_maxEpoch_100_maxD_0_bs_128/analysis.csv")
    analyse_latency("DREAM IMDB_MT", "./exp_result/IMDB_MT/DREAM_IMDB_MT_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_l2_1e-08_pat_10_clipGr_10.0_seed_1234_maxEpoch_100_maxD_0_bs_128/analysis.csv")
    analyse_latency("DREAM TPCH_PN", "./exp_result/TPCH_PN/DREAM_TPCH_PN_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_l2_1e-08_pat_10_clipGr_10.0_seed_1234_maxEpoch_100_maxD_0_bs_128/analysis.csv")
    analyse_latency("DREAM WIKI", "./exp_result/WIKI/DREAM_WIKI_cs_512_layer_1_predL_3_hDim_512_es_100_lr_0.001_maxC_200_l2_1e-08_pat_10_clipGr_10.0_seed_1234_maxEpoch_100_maxD_0_bs_128/analysis.csv")
    
    analyse_latency("LBS DBLP_AN", "./exp_result/DBLP_AN/LBS_DBLP_AN_N_5_PT_20_L_10_seed_0_maxD_0_pTest_0.1/analysis.csv")
    analyse_latency("LBS IMDB_AN", "./exp_result/IMDB_AN/LBS_IMDB_AN_N_5_PT_20_L_10_seed_0_maxD_0_pTest_0.1/analysis.csv")
    analyse_latency("LBS IMDB_MT", "./exp_result/IMDB_MT/LBS_IMDB_MT_N_5_PT_20_L_10_seed_0_maxD_0_pTest_0.1/analysis.csv")
    analyse_latency("LBS TPCH_PN", "./exp_result/TPCH_PN/LBS_TPCH_PN_N_5_PT_20_L_10_seed_0_maxD_0_pTest_0.1/analysis.csv")
    analyse_latency("LBS WIKI", "./exp_result/WIKI/LBS_WIKI_N_5_PT_20_L_10_seed_0_maxD_0_pTest_0.1/analysis.csv")