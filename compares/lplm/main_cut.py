import time
from collections import namedtuple

from pympler import asizeof
from torch.utils.data import DataLoader
import torch
import misc_utils
import selectivity_estimator
import numpy as np
from tqdm import tqdm
import pandas as pd

import datetime
import sys
import os


# This function load the saved model
def load_estimation_model(model_file_name, model_card, device):
    model_card.load_state_dict(torch.load(model_file_name))
    return model.to(device)


# This function trains and returns the embedding model
def modelTrain(train_data, model, device, learning_rate, num_epocs, model_save_path):
    model = selectivity_estimator.train_model(train_data, model, device, learning_rate, num_epocs, model_save_path)
    torch.save(model.state_dict(), model_save_path)
    return model


# This function estimate the cardinalities and saves them to a txt file
def estimate_cardinality(test_dataset, model, device, save_file_path, dataset_size, padding_latencies):
    write_to_file = open(save_file_path, 'w')
    qerror_list = []
    latencies = []
    estimates = []
    total_latencies = []
    print("Target device:", device)
    model = model.to(device)
    print("Tested in:", next(model.parameters()).device)
    with torch.no_grad():
        i = 0
        for name, mask, actual_card in tqdm(test_dataset, desc="Testing"):
            name = name.to(device)
            output = model(name)
            output = output.to(device)
            mask = mask.to(device)
            start = time.time()
            output = torch.prod(torch.pow(output, mask)) * dataset_size
            output = torch.clamp(output, min=1)
            end = time.time()
            estimates.append(output.item())
            latencies.append(end - start)
            total_latencies.append(end - start + padding_latencies[i])
            i += 1
            
            qerror = misc_utils.compute_qerrors(actual_card, output.item())
            qerror_list.append(qerror[0])
            # if i % 1000 == 0:
            #     print(f"{i} tested.")
            # write_to_file.write(str(output.item()) + '\n')
            # if qerror > 1000:
            #     print(f"est:{output.item()} card:{actual_card} q-error:{qerror[0]}")

    q_error_sum = 0
    for q in qerror_list:
        q_error_sum += q
    
    print(f"Sum:{q_error_sum} Avg:{q_error_sum / len(qerror_list)}")
    
    print("Mean", np.average(qerror_list))
    print(f'G-mean: {np.round(misc_utils.g_mean(qerror_list), 4)}')
    print(f'Mean: {np.round(np.average(qerror_list), 4)}')
    print(f'Median: {np.round(np.percentile(qerror_list, 50), 4)}')
    print(f'90th: {np.round(np.percentile(qerror_list, 90), 4)}')
    print(f'99th: {np.round(np.percentile(qerror_list, 99), 4)}')
    print(f'Max: {np.round(np.max(qerror_list), 4)}')
    print(f'Query time(no padding avg.): {np.mean(latencies)}')
    print(f'Query time(with padding avg.): {np.mean(total_latencies)}')
    return estimates, latencies


def get_vocab(trainfile, max_char=200):

    vocab_dict = {}
    for i in open(trainfile):
        i=i.strip().rsplit(':', 2)[0]
        # if i.find('\"') != -1:
        #     print(i)
        #     exit()
        i = i.replace('%', '\1')
        i = i.replace('$', '\2')
        i = i.replace('.', '\3')
        i = i.replace('#', '\4')
        for k in i:
            if k != '%':
                if k not in vocab_dict:
                    vocab_dict[k] = 1
                else:
                    vocab_dict[k] += 1
    vocab_dict[' '] = 0
    vocab_dict = dict(sorted(vocab_dict.items(), key=lambda item: item[1], reverse=True)[:max_char])
    # vocab_dict = list(zip(*sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)[:max_char]))[0]  # frequency order
    # print(vocab_dict)
    # return vocab_dict
    vocab_dict[' '] = 0
    vocab = ''
    for token in vocab_dict:
        vocab += token
    return vocab + '$.#'


def write_results_to_file(test_file, output_file, estimates, latencis, padding_latencies):
    query_strings = []
    ground_truths = []
    
    df = pd.read_csv(test_file, na_values=[], keep_default_na=False)
    # df = pd.read_csv(query_filename, dtype={'string': str, 'selectivity': float})
    query_strings = df['string'].astype(str).tolist()
    ground_truths = df['selectivity'].astype(int).tolist()
    
    # with open(test_file, "r") as f:
    #     first_line = True
    #     for line in f.readlines():
    #         if first_line == True:
    #             first_line = False
    #             continue
    #         temp = line.strip().rsplit(",", 1)
    #         if int(temp[1]) == 0:
    #             continue
    #         query_strings.append(temp[0])
    #         ground_truths.append(int(temp[1]))
    
    assert len(ground_truths) == len(estimates)
    qerror_list = []
    for i, (truth, est) in enumerate(zip(ground_truths, estimates)):
        qerror_list.append(max(truth / est, est / truth))
    for i, q in enumerate(qerror_list):
        if q > 10000:
            print(query_strings[i], ground_truths[i], estimates[i], q)
    total_latencies = [(x + y) for (x, y) in zip(latencies, padding_latencies)]
    print("========Final results========")
    print("Mean", np.average(qerror_list))
    print(f'G-mean: {np.round(misc_utils.g_mean(qerror_list), 4)}')
    print(f'Mean: {np.round(np.average(qerror_list), 4)}')
    print(f'Median: {np.round(np.percentile(qerror_list, 50), 4)}')
    print(f'90th: {np.round(np.percentile(qerror_list, 90), 4)}')
    print(f'99th: {np.round(np.percentile(qerror_list, 99), 4)}')
    print(f'Max: {np.round(np.max(qerror_list), 4)}')
    print(f'Query time(no padding avg.): {np.mean(latencies)}')
    print(f'Query time(with padding avg.): {np.mean(total_latencies)}')   
    
    with open(output_file, "w") as f:
        f.write("s_q, ground_truth, estimate, latency, padding_latency\n")
        for i, (s_q, ground_truth, estimate, latency, padding_latency) in \
            enumerate(zip(query_strings, ground_truths, estimates, latencis, padding_latencies)):
            f.write(s_q + "," + str(ground_truth) + ',')
            f.write(str(estimate) + ',' + str(latency) + ',' + str(padding_latency) + '\n')
            
            

if __name__ == "__main__":
    if sys.argv[1] == 'sample':
        vocabulary = get_vocab('sample_train_test/dblp_author_names_train.txt')
        trainpath = 'sample_train_test/dblp_author_names_train.txt'
        testpath = 'sample_train_test/dblp_author_test.txt'
        savepath = 'sample_train_test/estimated_cardinalities.txt'
        savemodel= 'sample_train_test/model.pth'
    elif sys.argv[1] == 'DBLP_AN_test':
        vocabulary = get_vocab('./Datasets/DBLP_AN/author_names_training_set_ground_truth2.txt')
        trainpath = './Datasets/DBLP_AN/author_names_training_set_ground_truth2.txt'
        # testpath = './Datasets/DBLP_AN/author_names_training_set.txt'
        testpath = '../../datasets/DBLP_AN/DBLP_AN_test.csv'
        savepath = './exp/DBLP_AN/author_names_test_estimated_cardinalities2.txt'
        savemodel= './exp/DBLP_AN/author_names_test_model.pth'
        datasize = 450000
    elif sys.argv[1] == 'DBLP_AN':
        vocabulary = get_vocab('./Datasets/DBLP_AN/DBLP_AN_train_ground_truth.txt')
        trainpath = './Datasets/DBLP_AN/DBLP_AN_train_ground_truth.txt'
        testpath = '../../datasets/DBLP_AN/DBLP_AN_test.csv'
        savepath = './exp/DBLP_AN/DBLP_AN_estimated_cardinalities.txt'
        savemodel= './exp/DBLP_AN/DBLP_AN_model.pth'
        datasize = 450000
    elif sys.argv[1] == 'IMDB_AN':
        vocabulary = get_vocab('./Datasets/IMDB_AN/IMDB_AN_train_ground_truth.txt')
        trainpath = './Datasets/IMDB_AN/IMDB_AN_train_ground_truth.txt'
        testpath = '../../datasets/IMDB_AN/IMDB_AN_test.csv'
        savepath = './exp/IMDB_AN/IMDB_AN_estimated_cardinalities_test.txt'
        savemodel= './exp/IMDB_AN/IMDB_AN_model.pth'
        datasize = 550000
    elif sys.argv[1] == 'IMDB_MT':
        vocabulary = get_vocab('./Datasets/IMDB_MT/IMDB_MT_train_ground_truth.txt')
        trainpath = './Datasets/IMDB_MT/IMDB_MT_train_ground_truth.txt'
        testpath = '../../datasets/IMDB_MT/IMDB_MT_test.csv'
        savepath = './exp/IMDB_MT/IMDB_MT_estimated_cardinalities.txt'
        savemodel= './exp/IMDB_MT/IMDB_MT_model.pth'
        datasize = 500000
    elif sys.argv[1] == 'TPCH_PN':
        vocabulary = get_vocab('./Datasets/TPCH_PN/TPCH_PN_train_ground_truth.txt')
        trainpath = './Datasets/TPCH_PN/TPCH_PN_train_ground_truth.txt'
        testpath = '../../datasets/TPCH_PN/TPCH_PN_test.csv'
        savepath = './exp/TPCH_PN/TPCH_PN_estimated_cardinalities.txt'
        savemodel= './exp/TPCH_PN/TPCH_PN_model.pth'
        datasize = 200000
    elif sys.argv[1] == 'WIKI':
        vocabulary = get_vocab('./Datasets/WIKI/WIKI_train_ground_truth_final.txt')
        trainpath = './Datasets/WIKI/WIKI_train_ground_truth_final.txt'
        testpath = '../../datasets/WIKI/WIKI_test.csv'
        savepath = './exp/WIKI/WIKI_estimated_cardinalities_finalll_0424.txt'
        savemodel= './exp/WIKI/WIKI_model_finalll.pth'
        resultspath='./exp/WIKI/WIKI_result_finalll_0424.txt'
        datasize = 1031930
    # elif sys.argv[1] == 'WIKI':
    #     vocabulary = get_vocab('likeCard_datasets/WIKI/WIKI_training_set_ground_truth_new.txt')
    #     trainpath = 'likeCard_datasets/WIKI/WIKI_training_set_ground_truth_new.txt'
    #     testpath = '../../datasets/WIKI/WIKI_valid.csv'
    #     savepath = './exp/WIKI/WIKI_estimated_cardinalities.txt'
    #     # savemodel= 'likeCard_datasets/DBLP/model.pth'
    #     savemodel= 'likeCard_datasets/WIKI/model.pth'
    #     datasize = 1031930
    
    load_model = True
    
    A_NLM_configs = namedtuple('A_NLM_configs', ['vocabulary', 'hidden_size', 'learning_rate', 'batch_size', 'datasize',
                                                 'num_epocs', 'train_data_path', 'test_data_path',
                                                 'save_qerror_file_path', 'device', 'save_path'])

    card_estimator_configs = A_NLM_configs(vocabulary= vocabulary, hidden_size=256,
                                           datasize=datasize, learning_rate=0.0001, batch_size=128, num_epocs=64,
                                           train_data_path=trainpath,
                                           test_data_path=testpath,
                                           save_qerror_file_path=savepath,
                                           device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                           save_path=savemodel)
    char2idx = {letter: (i+1) for i, letter in enumerate(card_estimator_configs.vocabulary)}
    # print(char2idx['$'])
    # print(char2idx)
    # exit()

    model = selectivity_estimator.Cardinality_Estimator(1, card_estimator_configs.hidden_size, card_estimator_configs.device,
                                              len(char2idx))
    
    if load_model:
        model.load_state_dict(torch.load(savemodel, map_location=card_estimator_configs.device))
        trained_model = model
        trained_model = trained_model.to(card_estimator_configs.device)
        # print(trained_model.device)
        print("Loading finished.")
    else:
        t_count1 = datetime.datetime.now()
        print("Start adding paddings...")
        train_data = misc_utils.addpaddingTrain(card_estimator_configs.train_data_path, char2idx)
        dataloadertrain = DataLoader(train_data, batch_size=card_estimator_configs.batch_size, shuffle=True)
        print("Start training....")
        trained_model = modelTrain(dataloadertrain, model, card_estimator_configs.device,
                                card_estimator_configs.learning_rate, card_estimator_configs.num_epocs,
                                card_estimator_configs.save_path)
        t_count2 = datetime.datetime.now()
        print("Training finished. Time:", t_count2 - t_count1)
        with open(resultspath, "w") as f:
            f.write(f"Training time: {t_count2 - t_count1}")
    
    print("Model size:", os.path.getsize(card_estimator_configs.save_path) / 1024 / 1024)
    # exit()
    
    if False and sys.argv[1] == 'WIKI':
        total_estimates = []
        total_latencies = []
        total_padding_latencies = []
        for set_idx in range(5):
            print(f"Testing {set_idx}th set...")
            tp = card_estimator_configs.test_data_path.rsplit(".", 1)
            path = tp[0] + str(set_idx) + ".csv"
            datasettest, padding_latencies = misc_utils.addpaddingTest(path, char2idx)
            dataloadertest = DataLoader(datasettest, batch_size=1)

            estimates, latencies = estimate_cardinality(dataloadertest, trained_model, "cpu",
                                card_estimator_configs.save_qerror_file_path, card_estimator_configs.datasize, padding_latencies)
            total_estimates.extend(estimates)
            total_latencies.extend(latencies)
            total_padding_latencies.extend(padding_latencies)
        write_results_to_file(card_estimator_configs.test_data_path, 
            card_estimator_configs.save_qerror_file_path, total_estimates, total_latencies, total_padding_latencies)
    else:
        datasettest, padding_latencies = misc_utils.addpaddingTest(card_estimator_configs.test_data_path, char2idx)
        dataloadertest = DataLoader(datasettest, batch_size=1)

        estimates, latencies = estimate_cardinality(dataloadertest, trained_model, "cpu",
                            card_estimator_configs.save_qerror_file_path, card_estimator_configs.datasize, padding_latencies)
        
        write_results_to_file(card_estimator_configs.test_data_path, 
            card_estimator_configs.save_qerror_file_path, estimates, latencies, padding_latencies)