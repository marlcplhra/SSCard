import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import pandas as pd
import misc_utils
from string_dataset_helpers import TripletStringDataset, StringSelectivityDataset
import EmbeddingLearner
import SupervisedSelectivityEstimator

import os
import datetime
import logging
import sys
import time

embedding_learner_configs, frequency_configs, selectivity_learner_configs = None, None, None


def setup_logging(path):
    # log_fn = "./logs/only_test"
    log_fn = path + "LOG_test.txt" #.format(datetime.date.today().strftime("%y%m%d"))
    if os.path.exists(log_fn):
        tp = input("Experiment already exists, redo? ")
        if tp != 'yes' and tp != 'y' and tp != "Yes" and tp != "Y":
            print("Exit")
            exit()
    logging.basicConfig(filename=log_fn, filemode='w', format='%(message)s',level=logging.INFO)
    # also log into console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    print(f"setup logging to {log_fn}.")

#This function gives a single place to change all the necessary configurations.
#Please see misc_utils for some additional descriptions of what these attributes mean
def setup_configs():
    global embedding_learner_configs, frequency_configs, selectivity_learner_configs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_learner_configs = misc_utils.AstridEmbedLearnerConfigs(embedding_dimension=64, batch_size=32,
        num_epochs=32, margin=0.2, device=device, lr=0.001, channel_size=8)
    # 其他数据集batch_size=128，WIKI要设置成32，要不cuda内存不够
    # path = "datasets/dblp/"
    # file_name_prefix = "IMDB_AN"
    file_name_prefix = sys.argv[1]
    path = "../../datasets/" + file_name_prefix + "/"
    #This assumes that prepare_dataset function was called to output the files.
    #If not, please change the file names appropriately
    # file_name_prefix = "dblp_titles"
    # query_type = "prefix"
    query_type = "substring"
    model_path = "./exp/" + file_name_prefix + "/"
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    frequency_configs = misc_utils.StringFrequencyConfigs(
        string_list_file_name= path + file_name_prefix + ".csv",
        selectivity_file_name= path + file_name_prefix + "_train.csv",
        selectivity_file_name_test= path + file_name_prefix + "_test_new.csv",
        triplets_file_name= path + file_name_prefix +  "_" +  query_type + "_triplets_p10.csv"
        # triplets_file_name=path + "test.csv"
    )

    selectivity_learner_configs = misc_utils.SelectivityEstimatorConfigs(
        embedding_dimension=64, batch_size=128, num_epochs=64, device=device, lr=0.001,
        #will be updated in train_selectivity_estimator
        min_val=0.0, max_val=1.0,
        embedding_model_file_name = model_path + file_name_prefix +  "_" +  query_type + "_embedding_model.pth",
        selectivity_model_file_name = model_path + file_name_prefix +  "_" +  query_type + "_selectivity_model.pth",
        estimation_file = model_path + "analyse_0424.csv"
        )

    setup_logging(model_path)
    return embedding_learner_configs, frequency_configs, selectivity_learner_configs



#This function trains and returns the embedding model
def train_astrid_embedding_model(string_helper, model_output_file_name=None):
    global embedding_learner_configs, frequency_configs

    #Some times special strings such as nan or those that start with a number confuses Pandas
    df = pd.read_csv(frequency_configs.triplets_file_name, na_values=["nan"], keep_default_na=False)
    df["Anchor"] = df["Anchor"].astype(str)
    df["Positive"] = df["Positive"].astype(str)
    df["Negative"] = df["Negative"].astype(str)
    # print(df[:10])
    # df = df[:100]

    # # 检查每一列是否包含"apple"
    # anchor_contains_apple = df["Anchor"].str.contains("nan", na=False)
    # positive_contains_apple = df["Positive"].str.contains("nan", na=False)
    # negative_contains_apple = df["Negative"].str.contains("nan", na=False)

    # # 将所有的检查结果合并起来，确保至少一个列包含"apple"
    # any_contains_apple = anchor_contains_apple | positive_contains_apple | negative_contains_apple

    # # 打印包含"apple"的行
    # df_with_apple = df[any_contains_apple]
    # print(df_with_apple)
    # exit()

    triplet_dataset = TripletStringDataset(df, string_helper)
    train_loader = DataLoader(triplet_dataset, batch_size=embedding_learner_configs.batch_size, shuffle=True)

    embedding_model = EmbeddingLearner.train_embedding_model(embedding_learner_configs, train_loader, string_helper, model_output_file_name)
    if model_output_file_name is not None:
        torch.save(embedding_model.state_dict(), model_output_file_name)
    return embedding_model

#This function performs min-max scaling over logarithmic data.
#Typically, the selectivities are very skewed.
#This transformation reduces the skew and makes it easier for DL to learn the models
def compute_normalized_selectivities(df, is_train):
    global selectivity_learner_configs
    normalized_selectivities, min_val, max_val = misc_utils.normalize_labels(df["selectivity"])
    df["normalized_selectivities"] = normalized_selectivities

    if is_train:
        #namedtuple's are immutable - so replace them with new instances
        selectivity_learner_configs = selectivity_learner_configs._replace(min_val=min_val)
        selectivity_learner_configs = selectivity_learner_configs._replace(max_val=max_val)
    return df


#This function trains and returns the selectivity estimator.
def train_selectivity_estimator(train_df, string_helper, embedding_model, model_output_file_name=None):
    global selectivity_learner_configs, frequency_configs

    string_dataset = StringSelectivityDataset(train_df, string_helper, embedding_model)
    train_loader = DataLoader(string_dataset, batch_size=selectivity_learner_configs.batch_size, shuffle=True)

    selectivity_model = SupervisedSelectivityEstimator.train_selEst_model(selectivity_learner_configs, train_loader, string_helper)
    if model_output_file_name is not None:
        torch.save(selectivity_model.state_dict(), model_output_file_name)
    return selectivity_model

#This is a helper function to get selectivity estimates for an iterator of strings
def get_selectivity_for_strings(strings, embedding_model, selectivity_model, string_helper):
    global selectivity_learner_configs
    from SupervisedSelectivityEstimator import SelectivityEstimator
    embedding_model.eval()
    selectivity_model.eval()
    strings_as_tensors = []
    with torch.no_grad():
        for string in strings:
            string_as_tensor = string_helper.string_to_tensor(string)
            #By default embedding mode expects a tensor of [batch size x alphabet_size * max_string_length]
            #so create a "fake" dimension that converts the 2D matrix into a 3D tensor
            string_as_tensor = string_as_tensor.view(-1, *string_as_tensor.shape)
            
            string_as_tensor = string_as_tensor.cuda()
            strings_as_tensors.append(embedding_model(string_as_tensor).cpu().numpy())
        strings_as_tensors = np.concatenate(strings_as_tensors)
        #normalized_selectivities= between 0 to 1 after the min-max and log scaling.
        #denormalized_predictions are the frequencies between 0 to N
        strings_as_tensors = torch.from_numpy(strings_as_tensors)
        strings_as_tensors = strings_as_tensors.cuda()
        normalized_predictions = selectivity_model(strings_as_tensors.clone().detach())
        denormalized_predictions = misc_utils.unnormalize_torch(normalized_predictions, selectivity_learner_configs.min_val,
            selectivity_learner_configs.max_val)
        return normalized_predictions, denormalized_predictions


#This is a helper function to get selectivity estimates for an iterator of strings
def get_selectivity_for_strings_cpu(strings, embedding_model, selectivity_model, string_helper):
    global selectivity_learner_configs
    from SupervisedSelectivityEstimator import SelectivityEstimator
    embedding_model.eval()
    selectivity_model.eval()
    embedding_model = embedding_model.to("cpu")
    selectivity_model = selectivity_model.to("cpu")
    
    latencies = []
    with torch.no_grad():
        normalized_predictions = []
        denormalized_predictions = []
        for string in strings:
            sta = time.time()
            string_as_tensor = string_helper.string_to_tensor(string)
            string_as_tensor = string_as_tensor.view(-1, *string_as_tensor.shape)
            string_as_tensor = embedding_model(string_as_tensor)
            normalized_prediction = selectivity_model(string_as_tensor)
            denormalized_prediction = misc_utils.unnormalize_torch(normalized_prediction, selectivity_learner_configs.min_val,
            selectivity_learner_configs.max_val)
            end = time.time()
            latencies.append(end - sta)
            normalized_predictions.append(normalized_prediction)
            denormalized_predictions.append(denormalized_prediction)

        normalized_predictions = torch.stack(normalized_predictions, dim=0)
        denormalized_predictions = torch.stack(denormalized_predictions, dim=0)
    
    return normalized_predictions, denormalized_predictions, latencies


def write_results_to_file(file, strings, actuals, estimates, latencies):
    with open(file, "w") as f:
        f.write("s_q, ground_truth, estimate, latency\n")
        for i, (s_q, ground_truth, estimate, latency) in \
            enumerate(zip(strings, actuals, estimates, latencies)):
                
            f.write(s_q + "," + str(ground_truth.item()) + ',')
            f.write(str(estimate.item()) + ',' + str(latency) + '\n')
       


def load_embedding_model(model_file_name, string_helper):
    from EmbeddingLearner import EmbeddingCNNNetwork
    embedding_model= EmbeddingCNNNetwork(string_helper, embedding_learner_configs)
    embedding_model.load_state_dict(torch.load(model_file_name))
    return embedding_model

def load_selectivity_estimation_model(model_file_name, string_helper):
    from SupervisedSelectivityEstimator import SelectivityEstimator
    selectivity_model = SelectivityEstimator(string_helper, selectivity_learner_configs)
    selectivity_model.load_state_dict(torch.load(model_file_name))
    return selectivity_model

def main():
    global embedding_learner_configs, frequency_configs, selectivity_learner_configs
    random_seed = 1234
    load_model = False
    misc_utils.initialize_random_seeds(random_seed)

    #Set the configs
    embedding_learner_configs, frequency_configs, selectivity_learner_configs = setup_configs()

    embedding_model_file_name = selectivity_learner_configs.embedding_model_file_name
    selectivity_model_file_name = selectivity_learner_configs.selectivity_model_file_name

    string_helper = misc_utils.setup_vocabulary(frequency_configs.string_list_file_name)

    #You can comment/uncomment the following lines based on whether you
    # want to train from scratch or just reload a previously trained embedding model.
    embed_sta_t = datetime.datetime.now()
    if load_model:
        embedding_model = load_embedding_model(embedding_model_file_name, string_helper)
        embedding_model = embedding_model.to(embedding_learner_configs.device)  
    else:
        embedding_model = train_astrid_embedding_model(string_helper, embedding_model_file_name)
    embed_end_t = datetime.datetime.now()

    print("Load/Train embedding model finished.")

    # #Load the input file and split into 50-50 train, test split
    # df = pd.read_csv(frequency_configs.selectivity_file_name)
    # #Some times strings that start with numbers or
    # # special strings such as nan which confuses Pandas' type inference algorithm
    # df["string"] = df["string"].astype(str)
    # df = compute_normalized_selectivities(df)
    # train_indices, test_indices = train_test_split(df.index, random_state=random_seed, test_size=0.5)
    # train_df, test_df = df.iloc[train_indices], df.iloc[test_indices]
    
    
    # train_df = pd.read_csv(frequency_configs.selectivity_file_name, na_values=["nan"], keep_default_na=False)
    # train_df["string"] = train_df["string"].astype(str)
    # train_df = compute_normalized_selectivities(train_df, True)
    # test_df = pd.read_csv(frequency_configs.selectivity_file_name_test, na_values=["nan"], keep_default_na=False)
    # test_df["string"] = test_df["string"].astype(str)
    train_df = pd.read_csv(frequency_configs.selectivity_file_name, na_values=[], keep_default_na=False)
    train_df["string"] = train_df["string"].astype(str)
    train_df = compute_normalized_selectivities(train_df, True)
    test_df = pd.read_csv(frequency_configs.selectivity_file_name_test, na_values=[], keep_default_na=False)
    test_df["string"] = test_df["string"].astype(str)
    # print(test_df[:50])
    # exit()
    # train_df = train_df[:100]
    # test_df = test_df[:100]
    test_df = compute_normalized_selectivities(test_df, False)

    #You can comment/uncomment the following lines based on whether you
    # want to train from scratch or just reload a previously trained embedding model.
    sel_sta_t = datetime.datetime.now()
    if load_model:
        selectivity_model = load_selectivity_estimation_model(selectivity_model_file_name, string_helper)
        selectivity_model = selectivity_model.to(selectivity_learner_configs.device)   
    else:
        selectivity_model = train_selectivity_estimator(train_df, string_helper, embedding_model, selectivity_model_file_name)
    sel_end_t = datetime.datetime.now()
    selectivity_model = load_selectivity_estimation_model(selectivity_model_file_name, string_helper)
    selectivity_model = selectivity_model.to(selectivity_learner_configs.device)
    print("Load/Train selectivity model finished.")
    
    emb_size = os.path.getsize(embedding_model_file_name)
    sel_size = os.path.getsize(selectivity_model_file_name)
    print("Model size", (emb_size + sel_size) / 1024 / 1024)
    # exit()

    #Get the predictions from the learned model and compute basic summary statistics
    test_gpu_sta_t = datetime.datetime.now()
    normalized_predictions, denormalized_predictions = get_selectivity_for_strings(
        test_df["string"].values, embedding_model, selectivity_model, string_helper)
    test_gpu_end_t = datetime.datetime.now()
    test_cpu_sta_t = datetime.datetime.now()
    normalized_predictions, denormalized_predictions, latencies = get_selectivity_for_strings_cpu(
        test_df["string"].values, embedding_model, selectivity_model, string_helper)
    test_cpu_end_t = datetime.datetime.now()
    
    # actual = torch.tensor(test_df["normalized_selectivities"].values)
    actual = torch.tensor(test_df["selectivity"].values)
    print(selectivity_learner_configs.min_val, selectivity_learner_configs.max_val)
    test_q_error = misc_utils.compute_qerrors(normalized_predictions, actual,
        selectivity_learner_configs.min_val, selectivity_learner_configs.max_val)
    
    write_results_to_file(selectivity_learner_configs.estimation_file, test_df["string"].values, actual, denormalized_predictions, latencies)
    logging.info(f"Test data: Mean q-error: {np.mean(test_q_error)}")
    logging.info(f"Test data: Percentile q-error: [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] {[np.quantile(test_q_error, q) for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]]}")
    percentiles = [round(np.quantile(test_q_error, q), 3) for q in [0.5, 0.9, 0.99]]
    logging.info(f"Test data: Percentile q-error: [0.5, 0.9, 0.99], Max: {percentiles} {np.max(test_q_error):.3f}")
    logging.info(f"Training embedding model time: {embed_end_t - embed_sta_t}")
    logging.info(f"Training selectivity model time: {sel_end_t - sel_sta_t}")
    logging.info(f"Testing time(GPU): {test_gpu_end_t - test_gpu_sta_t}")
    logging.info(f"Testing time(CPU): {test_cpu_end_t - test_cpu_sta_t}")
    logging.info(f"Latency (CPU): {np.mean(latencies)}")
    

if __name__ == "__main__":
    main()
