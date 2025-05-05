import psycopg2
import os
import sys
import logging


def setup_logging(log_fn):
    dir_path = os.path.dirname(log_fn)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    logging.basicConfig(filename=log_fn, filemode='w', format='%(message)s',level=logging.INFO)
    # also log into console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    print(f"setup logging to {log_fn}.")


def run(query_set, card_file, ID):
    mname = card_file.split('_')[0]
    cardname = card_file.split('.')[0]
    setup_logging(f"./pg_exp/{query_set}/{mname}/log_{cardname}_{ID}.txt")
    conn = psycopg2.connect(database="imdb", host="127.0.0.1", port=5434, password='postgres', user='postgres')
    cur = conn.cursor()

    cur.execute("SET ml_cardest_enabled=true;")
    # cur.execute("SET ml_joinest_enabled=false;")
    cur.execute("SET query_no=0;")
    # cur.execute("SET join_est_no=0;")
    cur.execute(f"SET ml_cardest_fname='{card_file}';")

    folderpath = "./" + query_set + "/"
    file_names = os.listdir(folderpath)
    file_names = sorted(file_names)
    # print(file_names)
    queries = []
    for filename in file_names:
        with open(folderpath + filename, 'r') as f:
            SQL = f.read()
            queries.append([filename, SQL])
    
    exe_times = []
    for i, (filename, SQL) in enumerate(queries):
        # if filename != '1a.sql': continue
        logging.info(filename)
        
        # if card_file == 'pg_pg.txt':
        #     cur.execute("SET ml_cardest_enabled=false;")    
        # else:
        cur.execute("EXPLAIN ANALYSE " + SQL)
        rows = cur.fetchall()
        for row in rows:
            logging.info(row[0])
            if 'Execution Time' in row[0]:
                t = float(row[0].strip()[15:-2].strip())
                exe_times.append([filename.split('.')[0], t])
        logging.info("=" * 20 + "\n\n")
        # if i > 0: break
        
    cur.close()
    conn.close()
    
    output_file = f"./pg_exp/{query_set}/{mname}/exe_time_{cardname}_{ID}.txt"
    total = 0
    with open(output_file, "w") as f:
        for i, (sql_name, t) in enumerate(exe_times):
            f.write("{}: {:.6f}\n".format(sql_name, t))
            total += t
        f.write("Total time:{:.6f}\n".format(total))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])