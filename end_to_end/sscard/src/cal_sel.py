import os
import likecard
import psycopg2
import logging


def setup_logging(log_fn):
    # log_fn = "./logs/db_api_results_serial_new2.txt"
    # log_fn = "./results_test.txt"
    logging.basicConfig(filename=log_fn, filemode='w', format='%(message)s',level=logging.INFO)
    # also log into console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    print(f"setup logging to {log_fn}.")


def get_table_size(tables):
    conn = psycopg2.connect(database="imdb", host="127.0.0.1", port=5432, password='postgres', user='postgres')
    conn.set_client_encoding('UTF8') 
    cur = conn.cursor()
    table_sizes = dict()
    for table in tables:
        SQL = "SELECT COUNT(*) FROM {};".format(table)
        cur.execute(SQL)
        rows = cur.fetchall()
        table_sizes[table] = rows[0][0]
    return table_sizes


def get_name_map(folderpath):
    file_names = os.listdir(folderpath)
    file_names = sorted(file_names)
    name_map = dict()
    tables = set()
    for filename in file_names:
        with open(folderpath + filename, 'r') as f:
            q = ""
            for line in f.readlines():
                if 'AS' in line and 'MIN' not in line:
                    temp = line.strip().split('AS')
                    full_name = temp[0].strip().split(' ')[-1]
                    short_name = temp[1][:-1].strip()
                    name_map[short_name] = full_name
                    tables.add(full_name)
    return name_map, tables


def get_like_queries(filename):
    selectivities = []
    like_predicates = []    # [id, column, string, LIKE/NOT LIKE]
    cnt = 0
    with open(filename, 'r') as f:
        for line in f.readlines():
            temp = line.strip().rsplit(',', 1)
            exp = temp[0]
            sel = float(temp[1])
            selectivities.append(sel)
            if '~~' in exp:
                if '!~~' in exp:
                    tp2 = exp.split('!~~')
                    not_like = True
                else:
                    tp2 = exp.split('~~')
                    not_like = False
                column = tp2[0].strip()
                string = tp2[1].strip()[1:-1]
                if string[0] != '%' or string[-1] != '%': 
                    cnt += 1
                    continue
                like_predicates.append([cnt, column, string, not_like])
            cnt += 1
    return like_predicates, selectivities


def estimate_selectivity(estimator, s, not_like, tsize):
    strings = s.split('%')
    # print(strings)
    sel = 1
    for substr in strings:
        if len(substr) < 1: continue
        c, _ = estimator.estimate_for_single_pattern_with_suffix_tree(substr, "likecard_tree")
        print(c, tsize)
        sel *= c / tsize
    if not_like == True:
        sel = 1 - sel
    print(sel)
    return sel


def main():
    setup_logging("../../cards/like_queries_single/log_test.txt")
    like_queries_path = "../../cards/like_queries_single/predicates_single.txt"
    like_predicates, selectivities = get_like_queries(like_queries_path)
    # print(selectivities[:20])
    like_predicates = sorted(like_predicates, key=lambda x:x[1])
    # print(like_predicates[:10])
    name_map, tables = get_name_map("../../like_queries_single/")
    # table_sizes = get_table_size(tables)
    table_sizes = {'kind_type': 7, 'movie_companies': 2609129, 'company_name': 234997, 'company_type': 4, 'person_info': 2963664, 'comp_cast_type': 4, 'complete_cast': 135086, 'keyword': 134170, 'movie_info_idx': 1380035, 'name': 4167491, 'role_type': 12, 'title': 2528312, 'movie_info': 14835720, 'info_type': 113, 'char_name': 3140339, 'cast_info': 36244344, 'movie_keyword': 4523930, 'aka_name': 901343, 'movie_link': 29997, 'aka_title': 361472, 'link_type': 18}
    # print(name_map)
    # print(tables)
    # print(table_sizes)
    # exit()
    estimated_sels = []
    loaded_col = ""
    for i, (idx, col, s, not_like) in enumerate(like_predicates):
        print(like_predicates[i])
        tp = col.split('.')
        table = name_map[tp[0]]
        table_col = table + '.' + tp[1]
        if loaded_col != table_col:
            filename = "../exp/" + table_col + "/SSCard_h3_l5000_e16/SSCard_h3_l5000_e16_0"
            estimator = likecard.load(filename)
            loaded_col = table_col

        res = [idx, estimate_selectivity(estimator, s, not_like, table_sizes[table])]
        print(res)
        logging.info(f"{idx}: {col} {s} {not_like} {res[1]}")
        estimated_sels.append(res)
    # exit()
        
        # if i > 5: break
    like_selectivities = list(selectivities)
    for i, (idx, sel) in enumerate(estimated_sels):
        like_selectivities[idx] = sel
    
    output_file = "../../cards/like_queries_single/sscard_pg_single.txt"
    with open(output_file, "w") as f:
        for sel in like_selectivities:
            f.write("{:.6f}\n".format(sel))
            # f.write(str(sel) + "\n")
    
    
    
if __name__ == '__main__':
    main()