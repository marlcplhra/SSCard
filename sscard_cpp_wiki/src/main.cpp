#include "common_structs.h"
#include "sscard.h"
#include <ctime>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <numeric>
#include "utf8.h"


void print_char32_string_as_utf8(char32_t* str) {
    std::vector<char32_t> unicode_vec;
    for (int i = 0; str[i] != U'\0'; ++i) {
        unicode_vec.push_back(str[i]);
    }

    std::string utf8_str;
    utf8::utf32to8(unicode_vec.begin(), unicode_vec.end(), std::back_inserter(utf8_str));

    std::cout << utf8_str << std::endl;
}


char* dname;
const int STRING_LEN = 500;
const int PATTERN_LEN = 50;
const int PATTERN_NUM = 1e6;

char32_t* data_strings[Const::STRINGS_NUM];

typedef struct {
    char32_t* pattern;
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

    printf("Data string file path:%s\n", filename);
    std::ifstream file(filename);
    if (!file) {
        perror("Failed to open file");
        return 0;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty() && line.back() == '\n') line.pop_back();
        if (!line.empty() && line.back() == '\r') line.pop_back();

        std::vector<char32_t> unicode_line;
        try {
            utf8::utf8to32(line.begin(), line.end(), std::back_inserter(unicode_line));
        } catch (...) {
            std::cerr << "Failed to decode line " << n << ": " << line << std::endl;
            continue;
        }

        size_t len = unicode_line.size();
        data_strings[n] = (char32_t*)malloc((len + 2) * sizeof(char32_t));
        if (data_strings[n] == NULL) {
            perror("Failed to allocate memory for line");
            file.close();
            return 0;
        }

        std::copy(unicode_line.begin(), unicode_line.end(), data_strings[n]);
        data_strings[n][len] = U'\0'; // null-terminate

        n++;
        // if(n > 10000) break;
    }

    file.close();

    int buc[0x110000] = {0};
    int ascii_cnt = 0;
    int min_cp = 0x10FFFF, max_cp = 0;

    for (int i = 0; i < n; ++i)
    {
        char32_t* s = data_strings[i];
        for (int j = 0; s[j] != U'\0'; ++j)
        {
            int cp = (int)s[j];
            if (!buc[cp]) ascii_cnt++;
            buc[cp]++;
            min_cp = std::min(min_cp, cp);
            max_cp = std::max(max_cp, cp);
        }
    }

    printf("total code points: %d\n", ascii_cnt);
    // printf("min code point: U+%04X max code point: U+%04X\n", min_cp, max_cp);
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

    printf("Query string file path:%s\n", filename);
    std::ifstream file(filename);
    if (!file) {
        perror("Unable to open file");
        return 0;
    }

    std::string line;
    std::getline(file, line);

    int entryCount = 0;

    while (std::getline(file, line)) {
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }

        if (line.empty()) continue;

        std::string utf8_part;
        std::string num_part;
        bool flg = 0;

        if (line[0] == '"') {
            size_t closing_quote = line.rfind('"');

            if (closing_quote == std::string::npos) {
                std::cerr << "Format error" << line << std::endl;
                continue;
            }

            utf8_part = line.substr(1, closing_quote - 1);

            if (closing_quote + 2 < line.size()) {
                num_part = line.substr(closing_quote + 2);
            } else {
                std::cerr << "Format error" << line << std::endl;
                continue;
            }
            size_t pos = 0;
            std::string target = "\"\"";
            std::string replacement = "\"";
            while ((pos = utf8_part.find(target, pos)) != std::string::npos) {
                utf8_part.replace(pos, target.length(), replacement);
                pos += replacement.length();  // Move past the replacement
            }

        } else {
            size_t comma_pos = line.rfind(',');
            if (comma_pos == std::string::npos) {
                std::cerr << "Format error" << line << std::endl;
                continue;
            }

            utf8_part = line.substr(0, comma_pos);
            num_part = line.substr(comma_pos + 1);
        }

        int num = std::atoi(num_part.c_str());

        // UTF-8 → UTF-32
        std::vector<char32_t> codepoints;
        try {
            utf8::utf8to32(utf8_part.begin(), utf8_part.end(), std::back_inserter(codepoints));
        } catch (...) {
            std::cerr << "UTF-8 tranformation error at line " << entryCount << ": " << utf8_part << std::endl;
            continue;
        }

        size_t len = codepoints.size();
        query_strings[entryCount].pattern = (char32_t*)malloc((len + 1) * sizeof(char32_t));
        if (!query_strings[entryCount].pattern) {
            perror("Memory error");
            return 1;
        }

        std::copy(codepoints.begin(), codepoints.end(), query_strings[entryCount].pattern);
        query_strings[entryCount].pattern[len] = U'\0';
        query_strings[entryCount].num = num;

        entryCount++;
    }

    file.close();
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
    sprintf(model_path, "../saved_models/SSCard_%s_wiki_o3.bin", dname);
    Serialize(estimator, model_path);
    SSCard estimator2 = Deserialize(model_path);


    int pattern_n = load_pattern_strings(dname);

    t_sta = clock();
    std::vector<double> query_times;
    for(int i = 0; i < pattern_n; ++i)
    {
        clock_t t_est_sta = clock();
        double est = estimator2.estimate_with_tree(query_strings[i].pattern);
        assert(est >= 1);
        clock_t t_est_end = clock();
        int num = std::max(1, query_strings[i].num);
        q_error[i] = std::max(num / est, est / num);
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