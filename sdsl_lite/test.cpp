#include <sdsl/suffix_arrays.hpp>
#include <fstream>
#include <string>

// using namespace std;
using namespace sdsl;

std::string read_string(std::string dname)
{
    std::string file_path = "../datasets/" + dname + "/" + dname + ".csv";
    std::ifstream file(file_path);
    if (!file) {
        std::cerr << "Error !" << std::endl;
        return "";
    }

    std::string content, line;
    while (std::getline(file, line)) {
        content += line + '\01';
    }
    return content;
}


char* dname;
const int STRING_LEN = 500;
const int PATTERN_LEN = 50;
const int PATTERN_NUM = 1e6;
typedef struct {
    std::string pattern;
    int num;
} Pattern;
Pattern query_strings[PATTERN_NUM];
double q_error[PATTERN_NUM];

int load_pattern_strings(std::string dname)
{
    std::string filename1 = "../datasets/" + dname + "/" + dname + "_test.csv";
    const char* filename = filename1.c_str();

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
        if (!line.empty() && line.back() == '\n') line.pop_back();
        if (!line.empty() && line.back() == '\r') line.pop_back();

        if (line.empty()) continue;

        size_t comma_pos = line.rfind(',');
        if (comma_pos == std::string::npos) {
            std::cerr << "Invalid line format: " << line << std::endl;
            continue;
        }

        std::string utf8_part = line.substr(0, comma_pos);
        std::string num_part = line.substr(comma_pos + 1);

        int num = std::atoi(num_part.c_str());

        query_strings[entryCount].pattern = utf8_part; 
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


int main(int argc, char* argv[]) 
{
    std::string dname = argv[1];
    std::string s = read_string(dname);

    clock_t t_sta = clock();
    csa_wt<> fm_index;
    construct_im(fm_index, s, 1);
    clock_t t_end = clock();

    printf("Building time:%.3lf(s)\n", (double)(t_end - t_sta) / CLOCKS_PER_SEC);

    std::string filename = "./saved_models/fm_index_" + dname + ".sdsl";
    store_to_file(fm_index, filename);
    std::ofstream out(filename + ".html");
    write_structure<HTML_FORMAT>(fm_index,out);

    csa_wt<> fm_index2;
    load_from_file(fm_index2, filename);

    int pattern_n = load_pattern_strings(dname);

    t_sta = clock();
    std::vector<double> query_times;
    for(int i = 0; i < pattern_n; ++i)
    {
        clock_t t_est_sta = clock();
        double est = count(fm_index2, query_strings[i].pattern);
        clock_t t_est_end = clock();
        // std::cout << query_strings[i].pattern << std::endl;
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
}