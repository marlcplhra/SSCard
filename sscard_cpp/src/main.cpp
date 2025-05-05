#include "common_structs.h"
#include "sscard.h"
#include <ctime>
#include <cstring>
#include <cstdio>
#include <fstream>
// #include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <numeric>
// #include <vector>


char* dname;
const int STRING_LEN = 500;
const int PATTERN_LEN = 50;
const int PATTERN_NUM = 1e6;

char* data_strings[Const::STRINGS_NUM];

typedef struct {
    char* pattern;
    int num;
} Pattern;
Pattern query_strings[PATTERN_NUM];
double q_error[PATTERN_NUM];


SSCard get_para_and_init_SSCard(char *argv[])
{
    dname = argv[1];
    int SIGMA_NUM = atoi(argv[2]);
    int tree_height = atoi(argv[3]);
    int prune_length = atoi(argv[4]);
    int cm = atoi(argv[5]);
    int error_bound = atoi(argv[6]);
    SSCard estimator = SSCard(SIGMA_NUM, tree_height, prune_length, cm, error_bound);
    return estimator;
}


int load_data_strings(char* dname)
{
    int n = 0;
    char filename[256];
    strcpy(filename, "../../datasets/");
    strcat(filename, dname);
    strcat(filename, "/");
    strcat(filename, dname);
    strcat(filename, ".csv");
    // strcat(filename, ".txt");
    printf("Data string file path:%s\n", filename);
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Failed to open file");
        return 0;
    }

    char buffer[STRING_LEN];
    
    while (fgets(buffer, STRING_LEN, file) != NULL) 
    {
        int len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') buffer[len - 1] = '\0';
        data_strings[n] = (char*)malloc((len + 2) * sizeof(char));
        if (data_strings[n] == NULL) 
        {
            perror("Failed to allocate memory for line");
            fclose(file);
            return 0;
        }
        strcpy(data_strings[n], buffer);
        n++;
        // if(n > 1000) break;
    }

    fclose(file);
    // for(int i = 0; i < 10; ++i) printf("%s\n", data_strings[i]);
    int buc[4000], ascii_cnt = 0;
    for(int i = 0; i < 4000; ++i) buc[i] = 0;
    int min_ascii = 10000, max_ascii = 0;
    for(int i = 0; i < n; ++i)
    {
        int m = strlen(data_strings[i]);
        for(int j = 0; j < m; ++j)
        {
            int c = data_strings[i][j];
            if(!buc[c])
            {
                ascii_cnt++;
            }
            buc[c]++;
            min_ascii = std::min(min_ascii, (int)data_strings[i][j]);
            max_ascii = std::max(max_ascii, (int)data_strings[i][j]);
        }
    }
    printf("total ascii:%d\n", ascii_cnt);
    printf("min ascii:%d max ascii:%d\n", min_ascii, max_ascii);
    return n;
}


int load_pattern_strings(char* dname)
{
    char filename[256];
    strcpy(filename, "../../datasets/");
    strcat(filename, dname);

    strcat(filename, "/");
    strcat(filename, dname);
    strcat(filename, "_test.csv");

    // strcat(filename, "/test_data.csv");
    printf("Query string file path:%s\n", filename);
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Unable to open file");
        return 0;
    }

    char line[PATTERN_LEN];
    if (fgets(line, sizeof(line), file) == NULL) {
        perror("Error reading file");
        fclose(file);
        return 0;
    }


    int entryCount = 0;

    char pattern[PATTERN_LEN];
    int num;
    while (fgets(line, sizeof(line), file)) {

        line[strcspn(line, "\n")] = '\0';
        // printf("%s\n", line);


        if (strlen(line) == 0) {
            continue;
        }

        char* last_comma = strrchr(line, ',');
        if (last_comma != NULL) {
            num = atoi(last_comma + 1);
            *last_comma = '\0';
            // printf("%s %d\n", line, (int)strlen(line));
            strncpy(pattern, line, PATTERN_LEN - 1);
            strncpy(pattern, line, strlen(line));
            int len = strlen(line);
            // printf("%d\n", len);

            query_strings[entryCount].pattern = (char *)malloc(len * sizeof(char));
            if (query_strings[entryCount].pattern == NULL) 
            {
                fprintf(stderr, "Memory allocation error\n");
                return 1;
            }
            strcpy(query_strings[entryCount].pattern, pattern);
            query_strings[entryCount].num = num;
            entryCount++;
            // if(entryCount > 10000) break;
            // printf("%s %d %d\n", pattern, num, entryCount);
        } else {
            fprintf(stderr, "Invalid line format: %s\n", line);
        }
    }
    fclose(file);
    return entryCount;
}


void solve_results(int pattern_n)
{
    double avg = 0;
    for(int i = 0; i < pattern_n; ++i) avg += q_error[i];
    avg /= pattern_n;
    std::sort(q_error, q_error + pattern_n);
    printf("Avg q-error:%.3lf\n", avg);
    int p50th = int(pattern_n * 0.5);
    int p90th = int(pattern_n * 0.9);
    int p99th = int(pattern_n * 0.99);
    printf("Percentile: [50, 90, 99, 100] [%.3lf, %.3lf, %.3lf, %.3lf]\n", 
        q_error[p50th], q_error[p90th], q_error[p99th], q_error[pattern_n - 1]);
}


void Serialize(SSCard model, char* filename)
{
    std::ofstream os(filename, std::ios::binary);
    cereal::BinaryOutputArchive archive(os);
    archive(model); 
    std::cout << "Saved model to " << filename << std::endl;
}


auto Deserialize(char* filename)
{
    SSCard model;
    std::ifstream is(filename, std::ios::binary);
    cereal::BinaryInputArchive archive(is);
    archive(model);
    model.setTreeFa();
    return model;
}


int main(int argc, char *argv[])
{
    SSCard estimator = get_para_and_init_SSCard(argv);

    int n = load_data_strings(dname);
    
    clock_t t_sta = clock();
    estimator.build_SSCard(n, data_strings);
    clock_t t_end = clock();
    printf("Building time:%.3lf(s)\n", (double)(t_end - t_sta) / CLOCKS_PER_SEC);

    // save and load model
    char model_path[255];
    sprintf(model_path, "../saved_models/SSCard_%s_o3.bin", dname);
    Serialize(estimator, model_path);
    SSCard estimator2 = Deserialize(model_path);


    int pattern_n = load_pattern_strings(dname);

    t_sta = clock();
    std::vector<double> query_times;
    for(int i = 0; i < pattern_n; ++i)
    {
        clock_t t_est_sta = clock();
        double est = estimator2.estimate_with_tree(query_strings[i].pattern);
        clock_t t_est_end = clock();
        q_error[i] = std::max(query_strings[i].num / est, est / query_strings[i].num);
        // printf("%s num:%d est:%.3lf\n", query_strings[i].pattern, query_strings[i].num, est);
        query_times.push_back((double)(t_est_end - t_est_sta) / CLOCKS_PER_SEC);
    }
    t_end = clock();
    printf("Query time:%.3lf(s)\n", (double)(t_end - t_sta) / CLOCKS_PER_SEC);
    double sum = std::accumulate(query_times.begin(), query_times.end(), 0.0);
    double average = sum / query_times.size();
    printf("Query time per q:%.6lf(ms)\n", average * 1000);
    solve_results(pattern_n);

    return 0;
}