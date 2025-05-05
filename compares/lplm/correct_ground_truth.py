import os
import sqlite3
import multiprocessing
import traceback
import concurrent.futures
import datetime
import sys
import numpy as np
import time
import shutil
import pandas as pd


# def query_database(db_name, pattern, result_queue):
def query_database(db_name, pattern):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('SELECT count(*) FROM pattern WHERE trans LIKE ?', (pattern,))
    result = c.fetchone()[0]
    conn.close()
    return result
    # result_queue.put(result)

def parallel_query(dataset_path, num_dbs, pattern):
    path = dataset_path.rsplit('/', 1)[0]
    db_files = [path + f'/dbs2/database_{i}.db' for i in range(num_dbs)]
    
    processes = []
    result_queue = multiprocessing.Queue()
    
    for db_name in db_files:
        p = multiprocessing.Process(target=query_database, args=(db_name, pattern, result_queue,))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    results = [result_queue.get() for _ in processes]
    total_count = sum(results)
    return total_count
    
    
    p = multiprocessing.Pool()
    results = []
    for db_name in db_files:
        res = p.apply(query_database, args=(db_name, pattern,))
        results.append(res)
    total_count = sum(results)
    return total_count


# def return_cardinality(query_list, c, dataset_size):
def return_cardinality(c, query_list, dataset_size):
    if len(query_list) == 1:
        # cn = parallel_query(dataset_path, num_dbs, query_list[0])
        # cn = parallel_query(dataset_path, num_dbs, query_list[0])
        cn = c.execute('SELECT count(*) FROM pattern WHERE trans LIKE ?', (query_list[0],)).fetchall()[0][0]
        # conn.close()
        prob = cn / dataset_size
        return prob
    else:
        cn = c.execute('SELECT count(*) FROM pattern WHERE trans LIKE ?', (query_list[0],)).fetchall()[0][0]
        # cn = parallel_query(dataset_path, num_dbs, query_list[0])
        cn1 = c.execute('SELECT count(*) FROM pattern WHERE trans LIKE ?', (query_list[1],)).fetchall()[0][0]
        # cn1 = parallel_query(dataset_path, num_dbs, query_list[1])
        # conn.close()
        if cn1 == 0:
            print(f"Error: Like pattern actual card is {0}")
            return 0
            # cn1 = 1
        prob = float(cn) / cn1
        
        return prob


def find_all_possible_probabilities(like, all_con_prob_list):
    wildcard_list = ['$', '^']
    if len(like) == 0:
        return all_con_prob_list
    elif len(like) == 1:
        all_con_prob_list.append(('%' + like[-1] + '%',))
        return all_con_prob_list
    else:
        if like[-1] not in wildcard_list:
            if like[-2] not in wildcard_list:
                all_con_prob_list.append(('%' + like + '%', '%' + like[:-1] + '%'))
                return find_all_possible_probabilities(like[:-1], all_con_prob_list)
            else:
                if len(like) > 2:
                    if like[-2] == '^':
                        all_con_prob_list.append(
                            ('%' + like[:-2] + '^' + like[-1] + '%', '%' + like[:-2] + '%' + like[-1] + '%'))
                        all_con_prob_list.append(('%' + like[:-2] + '%' + like[-1] + '%', '%' + like[:-2] + '%'))

                        return find_all_possible_probabilities(like[:-2], all_con_prob_list)
                    elif like[-2] == '$':
                        all_con_prob_list.append(('%' + like + '%', '%' + like[:-2] + '%' + like[-1] + '%'))
                        all_con_prob_list.append(('%' + like[:-2] + '%' + like[-1] + '%', '%' + like[:-2] + '%'))

                        return find_all_possible_probabilities(like[:-2], all_con_prob_list)
                else:
                    all_con_prob_list.append(('^' + like[-1] + '%', '%' + like[-1] + '%'))
                    all_con_prob_list.append(('%' + like[-1] + '%',))

                    return find_all_possible_probabilities('', all_con_prob_list)
        else:
            if len(all_con_prob_list) == 0:
                all_con_prob_list.append(('%' + like[:-1] + '^', '%' + like[:-1] + '%'))
                return find_all_possible_probabilities(like[:-1], all_con_prob_list)


def language_to_query(list_languages):
    list_queries = []
    list_wildcards = ['$', '%', '_', '@']
    for con in list_languages:
        list_pairs = []
        for l in con:
            query = ''
            l_ = l.replace('%^', '_').replace('^%', '_').replace('^', '_').replace('@', ' ')
            for i in range(len(l_)):
                if len(query) == 0:
                    query += l_[i]
                else:

                    if query[-1] not in list_wildcards and l_[i] not in list_wildcards:
                        query += '%' + l_[i]
                    else:
                        query += l_[i]
            list_pairs.append(query.replace('$', ''))
        list_queries.append(list_pairs)
    return list_queries


# def LIKE_pattern_to_newLanguage(liste):
#     transformed_pattern = ''
#     for key in liste:
#         if len(key) == 1:
#             transformed_pattern += key
#         else:
#             new = ''
#             count = 0
#             for char in key:
#                 if count < 1:
#                     new += char
#                     count += 1
#                 else:
#                     if new[-1] != '_' and char != '_' and char != '@' and new[-1] != '@':
#                         new += '$' + char
#                     else:
#                         new += char
#             transformed_pattern += new
#     transformed_pattern = transformed_pattern.replace('_', '^')
#     return transformed_pattern
def LIKE_pattern_to_newLanguage(liste):
    transformed_pattern = ''
    for pattern in liste:
        if len(pattern) == 1:
            transformed_pattern += pattern
        else:
            new_pattern = ''
            count = 0
            for char in pattern:
                if count < 1:
                    new_pattern += char
                    count += 1
                else:
                    if (
                        new_pattern[-1] not in ('_', '@')
                        and char not in ('_', '@')
                    ):
                        # new_pattern += char + '$'
                        new_pattern += '$' + char
                    else:
                        new_pattern += char
            transformed_pattern += new_pattern
    transformed_pattern = transformed_pattern.replace('_', '^')
    return transformed_pattern


def inject_type(liste, type_):
    newliste = []
    if type_ == 'prefix' or type_ == 'end_underscore':
        liste.insert(-1, [liste[-1][0][1:], liste[-1][0]])
        for i in range(len(liste)):
            if i < len(liste) - 2:
                newliste.insert(0, [liste[i][0][1:], liste[i][1][1:]])
            else:
                newliste.insert(0, liste[i])
        return newliste
    if type_ == 'suffix' or type_ == 'begin_underscore':
        liste.insert(0, [liste[0][0][:-1], liste[0][0]])
        for i in range(len(liste)):
            newliste.insert(0, liste[i])
        return newliste

    elif type_ == 'prefix_suffix':
        liste.insert(-1, [liste[-1][0][1:], liste[-1][0]])
        liste.insert(0, [liste[0][0][:-1], liste[0][0]])
        for i in range(len(liste)):
            if i < len(liste) - 2:
                newliste.insert(0, [liste[i][0][1:], liste[i][1][1:]])
            else:
                newliste.insert(0, liste[i])
        return newliste
    else:
        for i in range(len(liste)):
            newliste.insert(0, liste[i])
        return newliste


def create_sqlite_db2(dataset_path, num_dbs):
    with open(dataset_path, 'r') as f:
        lines = f.readlines()

    chunk_size = len(lines) // num_dbs

    for i in range(num_dbs):
        path = dataset_path.rsplit('/', 1)[0]
        db_name = path + f'/dbs2/database_{i}.db'
        if os.path.exists(db_name):
            os.remove(db_name)
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS pattern (id INTEGER PRIMARY KEY, trans TEXT)''')

        chunk = lines[i * chunk_size: (i + 1) * chunk_size]
        for line in chunk:
            c.execute('INSERT INTO pattern (trans) VALUES (?)', (line.strip(),))
        
        conn.commit()
        conn.close()


def create_sqlite_db_repeat(dataset_path, db_path, chunk_num):
    with open(dataset_path, 'r') as f:
        lines = f.readlines()
    
    # x = 0
    # for line in lines:
    #     print(line.strip())
    #     x += 1
    #     if x > 10: exit()

    db = db_path + f'database_0.db'
    if not os.path.exists(db):
        conn = sqlite3.connect(db)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS pattern (id INTEGER PRIMARY KEY, trans TEXT)''')
        for line in lines:
            c.execute('INSERT INTO pattern (trans) VALUES (?)', (line.strip(),))
        conn.commit()
        conn.close()
    for i in range(1, chunk_num):
        # path = dataset_path.rsplit('/', 1)[0]
        # db_name = path + f'/dbs_repeat/database_{i}.db'
        db_name = db_path + f'database_{i}.db'
        if os.path.exists(db_name):
            continue
        shutil.copy2(db, db_name)
        if (i + 1) % 100 == 0:
            print(f"Creating db {i + 1} / {chunk_num}")


def create_sqlite_db(dataset_path):
    with open(dataset_path, 'r') as f:
        lines = f.readlines()

    path = dataset_path.rsplit('/', 1)[0]
    db_name = path + f'/dbs/database_0.db'
    if os.path.exists(db_name):
        os.remove(db_name)
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS pattern (id INTEGER PRIMARY KEY, trans TEXT)''')

    for line in lines:
        c.execute('INSERT INTO pattern (trans) VALUES (?)', (line.strip(),))
    
    conn.commit()
    conn.close()


results = []
# def get_likepatterns_ground_truth(like_idx, likepatterns, pattern_nums, dataset_path, datasetsize, file_to_save):
def get_likepatterns_ground_truth(c, likepatterns, datasetsize, path_file_tosave, lock=None):
    # print(likepatterns)
    # likepatterns = '%AB%C'
    # likepatterns = 'afdadfhahjdk'
    # print(likepatterns)
    newlike = likepatterns.strip().replace(' ', '@')
    likepatterns = ('%' + newlike + '%').replace('%%', '%').replace('%_', '_').replace('_%', '_')
    if likepatterns[0] == '%':
        likepatterns = likepatterns[1:]
    if likepatterns[-1] == '%':
        likepatterns = likepatterns[:-1]
    transformed_pattern = LIKE_pattern_to_newLanguage(likepatterns.split('%'))
    # print(likepatterns, transformed_pattern)
    all_con_language_prob = find_all_possible_probabilities(transformed_pattern, [])
    # print(all_con_language_prob)
    all_con_prob1 = language_to_query(all_con_language_prob)
    # exit()

    if (newlike[0] == '%' and newlike[-1] != '%' and newlike[-1] != '_'):
        all_con_prob = inject_type(all_con_prob1, 'suffix')
        newlike = likepatterns + ':' + 'suffix'

    elif (newlike[0] != '%' and newlike[-1] == '%' and newlike[0] != '_'):
        all_con_prob = inject_type(all_con_prob1, 'prefix')
        newlike = likepatterns + ':' + 'prefix'

    elif newlike[0] != '%' and newlike[-1] != '%' and newlike[0] != '_' and newlike[-1] != '_':
        all_con_prob = inject_type(all_con_prob1, 'prefix_suffix')
        newlike = likepatterns + ':' + 'prefix_suffix'

    elif newlike[0] != '%' and newlike[-1] == '_' and newlike[0] != '_':
        all_con_prob = inject_type(all_con_prob1, 'end_underscore')
        newlike = likepatterns + ':' + 'end_underscore'

    elif newlike[-1] != '%' and newlike[0] == '_' and newlike[-1] != '_':
        all_con_prob = inject_type(all_con_prob1, 'begin_underscore')
        newlike = likepatterns + ':' + 'begin_underscore'
    else:
        all_con_prob = inject_type(all_con_prob1, 'substring')
        newlike = likepatterns + ':' + 'substring'
    # print(all_con_language_prob)
    # print(newlike)
    # exit()
    liste = []  
    try:
        for pair in all_con_prob:
            # print(pair)
            # liste.append(return_cardinality(pair, c, datasetsize))
            tp = return_cardinality(c, pair, datasetsize)
            if tp == 0: 
                # print(f"====={likepatterns}: return=====")
                return
            liste.append(tp)
            # print(liste[-1])
        # conn.close()
        # print(like_idx % chunk_size, db)
        s = [str(k) for k in liste]
        # exit()
        
        sel = 1
        for i in liste:
            sel *= i
        # print(f"{likepatterns}: {sel * datasetsize}")
        # print(s)
        # return newlike + ':' + ' '.join(s)
        with lock:
            with open(path_file_tosave, "a") as f:
                f.write(newlike + ':' + ' '.join(s) + '\n')
                
    except Exception as e:
        print(f"Error occurred with likepatterns: {likepatterns}")
        print("Error message:", str(e))
        traceback.print_exc()


def solve_pattern_list(chunk_id, old_output, chunk, db_path, datasetsize, path_file_tosave, lock):
    # path = dataset_path.rsplit('/', 1)[0]
    # db = path + f'/dbs_repeat/database_{chunk_id}.db'
    db = db_path + f'database_{chunk_id}.db'
    conn = sqlite3.connect(db)
    c = conn.cursor()
    for i, likepatterns in enumerate(chunk):
        p_ori = likepatterns[1:-1]
        if p_ori in old_output:
            with lock:
                with open(path_file_tosave, "a") as f:
                    f.write(old_output[p_ori] + "\n")
        else:
            get_likepatterns_ground_truth(c, likepatterns, datasetsize, path_file_tosave, lock)
        if (i + 1) % 10 == 0:
            print(f"chunk {chunk_id}: {i+1}/{len(chunk)} finished.")
        
    conn.close()


def main(dataset_path, db_path, list_of_patterns, path_file_tosave, datasetsize):
    if os.path.exists(path_file_tosave):
        tp = input("Ground truth already exists, remove? ")
        if tp == 'y' or tp == 'yes':
            os.remove(path_file_tosave)
        else:
            exit()

    old_file = "./Datasets/WIKI/WIKI_train_ground_truth2.txt"
    old_output = {}
    aa = 0
    for line in open(old_file):
    # with open(old_file, "r") as f:
        # for line in f.readlines():
            tp = line.strip().rsplit(':', 2)
            old_output[tp[0]] = line.strip()
            # aa += 1
            # if aa > 100: break
    print("Loading old file finished.")

    # list_of_patterns = list_of_patterns[:100]
    pattern_nums = len(list_of_patterns)
    lock = multiprocessing.Lock()

    chunk_num = 500
    chunks = np.array_split(list_of_patterns, chunk_num)
    sta_t = datetime.datetime.now()
    print("Start creating sqlite dbs...")
    create_sqlite_db_repeat(dataset_path, db_path, chunk_num)
    print("Start computing...")
    processes = []
    for i, chunk in enumerate(chunks):
        p = multiprocessing.Process(target=solve_pattern_list, args=(i, old_output, chunk, \
                                            db_path, datasetsize, path_file_tosave, lock))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

        # print(f"{(i + 1) * chunk_size}/{pattern_nums} finished. Time:{end_t - sta_t}")
    
    end_t = datetime.datetime.now()
    print(f"Computing finished. Time:{end_t - sta_t}")


def load_like_patterns(filename):
    list_of_patterns = []
    df = pd.read_csv(filename, na_values=[], keep_default_na=False)
    # df = pd.read_csv(query_filename, dtype={'string': str, 'selectivity': float})
    pattern = df['string'].astype(str).tolist()
    num = df['selectivity'].astype(float).tolist()
    list_of_patterns = ['%' + p + '%' for p in pattern]
    # with open(filename, "r") as f:
    #     first_line = True
    #     for line in f.readlines():
    #         if first_line == True:
    #             first_line = False
    #             continue
    #         temp = line.strip().rsplit(",", 1)
    #         if int(temp[1]) == 0:
    #             continue
    #         s = '%' + temp[0] + '%'
    #         # s = temp[0]
    #         list_of_patterns.append(s) 
    # print(list_of_patterns[:100])
    return list_of_patterns


def create_like_patterns(input_file, output_file):
    pattern = []
    num = []
    with open(input_file, "r") as f:
        first_line = True
        for line in f.readlines():
            if first_line == True:
                first_line = False
                continue
            
            temp = line.strip().rsplit(",", 1)
            if int(temp[1]) == 0:
                continue
            pattern.append(temp[0])
            num.append(float(temp[1]))
    with open(output_file, "w") as f:
        for p in pattern:
            f.write('%' + p + '%\n')


def main_WIKI(dataset_path, db_path, list_of_patterns, path_file_tosave, datasetsize):
    if os.path.exists(path_file_tosave):
        tp = input("Ground truth already exists, remove? ")
        if tp == 'y' or tp == 'yes':
            os.remove(path_file_tosave)
        else:
            exit()

    old_file = "./Datasets/WIKI/WIKI_train_ground_truth2.txt"
    old_output = {}
    for line in open(old_file):
    # with open(old_file, "r") as f:
        # for line in f.readlines():
            tp = line.strip().rsplit(':', 2)
            old_output[tp[0]] = line.strip()
    print("Loading old file finished.")
    
    output = []
    db = db_path + f'database_0.db'
    conn = sqlite3.connect(db)
    c = conn.cursor()
    p_len = len(list_of_patterns)
    for i, p in enumerate(list_of_patterns):
        p_ori = p[1:-1]
        if p_ori in old_output:
            output.append(old_output[p_ori])
        else:
            print(p_ori)
            tp = get_likepatterns_ground_truth(c, p, datasetsize, path_file_tosave)
            output.append(tp)
        # print(output[-1])
        if i % 50 == 0:
            print(f"{i}/{p_len} finished.")
                    
    conn.close() 
    with open(path_file_tosave, "w") as f:
        for x in output:
            f.write(x + "\n")


if __name__ == "__main__":
    # db_path = 'path to database'
    # like_patterns_path = 'path to training dataset'
    # file_to_save_ground = 'path to save ground truth'
    if sys.argv[1] == "DBLP_AN_w":    
        # db_path = 'author_names.db'
        dataset_path = './Datasets/DBLP_AN/author_name.csv'
        db_path = './Datasets/DBLP_AN/dbs_repeat/'
        like_patterns_path = './Datasets/DBLP_AN/author_names_training_set_new.txt'
        file_to_save_ground = './Datasets/DBLP_AN/author_names_training_set_ground_truth_new.txt'
        datasetsize = 450000
    elif sys.argv[1] == "DBLP_AN":    
        # db_path = 'author_names.db'
        dataset_path = '../../datasets/DBLP_AN/DBLP_AN.csv'
        db_path = './Datasets/DBLP_AN/dbs_repeat/'
        like_patterns_path = '../../datasets/DBLP_AN/DBLP_AN_train.csv'
        file_to_save_ground = './Datasets/DBLP_AN/DBLP_AN_train_ground_truth3.txt'
        datasetsize = 450000
    elif sys.argv[1] == "IMDB_AN":    
        # db_path = 'author_names.db'
        dataset_path = '../../datasets/IMDB_AN/IMDB_AN.csv'
        db_path = './Datasets/IMDB_AN/dbs_repeat/'
        like_patterns_path = '../../datasets/IMDB_AN/IMDB_AN_train.csv'
        file_to_save_ground = './Datasets/IMDB_AN/IMDB_AN_train_ground_truth.txt'
        datasetsize = 550000
    elif sys.argv[1] == "IMDB_MT":    
        # db_path = 'author_names.db'
        dataset_path = '../../datasets/IMDB_MT/IMDB_MT.csv'
        db_path = './Datasets/IMDB_MT/dbs_repeat/'
        like_patterns_path = '../../datasets/IMDB_MT/IMDB_MT_train.csv'
        file_to_save_ground = './Datasets/IMDB_MT/IMDB_MT_train_ground_truth.txt'
        datasetsize = 500000
    elif sys.argv[1] == "TPCH_PN":    
        # db_path = 'author_names.db'
        dataset_path = '../../datasets/TPCH_PN/TPCH_PN.csv'
        db_path = './Datasets/TPCH_PN/dbs_repeat/'
        like_patterns_path = '../../datasets/TPCH_PN/TPCH_PN_train.csv'
        file_to_save_ground = './Datasets/TPCH_PN/TPCH_PN_train_ground_truth.txt'
        datasetsize = 20000
    elif sys.argv[1] == "WIKI":    
        # db_path = 'author_names.db'
        dataset_path = '../../datasets/WIKI/WIKI.csv'
        db_path = './Datasets/WIKI/dbs_repeat/'
        like_patterns_path = '../../datasets/WIKI/WIKI_train.csv'
        file_to_save_ground = './Datasets/WIKI/WIKI_train_ground_truth_final.txt'
        datasetsize = 1031930
    # elif sys.argv[1] == "WIKI":
    #     dataset_path = 'likeCard_datasets/WIKI/WIKI.txt'
    #     # db_path = 'likeCard_datasets/WIKI/dbs_repeat'
    #     like_patterns_path = 'likeCard_datasets/WIKI/WIKI_training_set.txt'
    #     file_to_save_ground = 'likeCard_datasets/WIKI/WIKI_training_set_ground_truth_test.txt'
    #     create_like_patterns('likeCard_datasets/WIKI/train_data.csv', like_patterns_path)
    #     datasetsize = 1031930
        
    if not os.path.exists(like_patterns_path):
        print(f"Error: Like patterns file not found at {like_patterns_path}")
    else:
        list_of_patterns = load_like_patterns(like_patterns_path)
        print("Load finished.")
        # main(db_path, list_of_patterns, file_to_save_ground, datasetsize)
        main(dataset_path, db_path, list_of_patterns, file_to_save_ground, datasetsize)
        
        # main_WIKI(dataset_path, db_path, list_of_patterns, file_to_save_ground, datasetsize)