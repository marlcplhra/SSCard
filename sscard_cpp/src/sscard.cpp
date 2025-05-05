#include "sscard.h"
#include <cstdio>
#include <cstring>
#include <cstdlib> 
#include <ctime>
#include <algorithm>
#include <fstream>
#include <assert.h>

SSCard::SSCard() {}

SSCard::SSCard(int SIGMA_NUM, int tree_height, int prune_length, int cm, double error_bound): tree(SIGMA_NUM, tree_height, prune_length, cm, error_bound) 
{
    this->SIGMA_NUM = SIGMA_NUM;
    // C.resize(SIGMA_NUM);
}


char get_char(char *strings[], int idx, int pos)
{
    int m = strlen(strings[idx]);
    // if(pos >= m) return strings[idx][pos % m];
    if(pos >= m) return strings[idx][pos - m];
    else return strings[idx][pos];
}

char get_L(char *strings[], Idx_pos r)
{
    if(r.pos == 0) return Const::EOS;
    else return strings[r.idx][r.pos - 1];
}

int sorted_cnt = 0;
Idx_pos *tp_rotations = NULL;
Idx_pos *sorted_rotations = NULL;
Idx_pos *all_rotations = NULL;
int *sigma_cnt;
int *st, top = 0;
void radix_sort(char *strings[], const int& L, const int& R, const int& h)  //[L,R)
{
    // printf("==L:%d R:%d\n", L, R);
    if(R - L == 1 | h == (strlen(strings[all_rotations[0].idx]) << 1))
    {
        memcpy(sorted_rotations + sorted_cnt, all_rotations + L, (R - L) * sizeof(Idx_pos));
        sorted_cnt += R - L;
        return;
    }
    for(int i = L; i < R; ++i)
    {
        char c = get_char(strings, all_rotations[i].idx, all_rotations[i].pos + h);
        if(sigma_cnt[c] == 0) st[top++] = c;
        sigma_cnt[c]++;
    }
    std::sort(st, st + top); 
    int *buc = (int *)calloc((top + 1), sizeof(int));
    int las = 0, buc_len = top;
    for(int i = 0; i < top; ++i)
    {
        int tp = sigma_cnt[st[i]];
        sigma_cnt[st[i]] = las;
        buc[i] = L + las;
        las += tp;
    }
    buc[buc_len] = R;
    // printf("------\n");
    // for(int i = 0; i < buc_len; ++i) printf("%d %d\n", buc[i] + 1, buc[i + 1]);
    for(int i = L; i < R; ++i)
    {
        char c = get_char(strings, all_rotations[i].idx, all_rotations[i].pos + h);
        tp_rotations[L + sigma_cnt[c]] = all_rotations[i];
        sigma_cnt[c]++;
    }
    memcpy(all_rotations + L, tp_rotations + L, (R - L) * sizeof(Idx_pos));
    while(top) sigma_cnt[st[--top]] = 0;

    for(int i = 0; i < buc_len; ++i)
        radix_sort(strings, buc[i], buc[i + 1], h + 1);
    
    delete[] buc;
}


void SSCard::calc_C(int n, char* data_strings[])
{
    for(int i = 0; i < n; ++i)
    {
        int m = strlen(data_strings[i]);
        for(int j = 0; j < m; ++j)
        {
            char c = data_strings[i][j];
            if(!C.count(c))
            {
                C[c] = 1;
                sigma.push_back(c);
            } 
            else C[c]++;
        }
    }
    std::sort(sigma.begin(), sigma.end());
    int las = 0;
    for(char c: sigma)
    {
        int tp = C[c];
        C[c] = las;
        las += tp;
    }
}

void SSCard::build_SSCard(int n, char* data_strings[])
{
    int sum_len = 0;
    // int n = sizeof(strings) / sizeof(strings[0]);
    for(int i = 0; i < n; ++i) sum_len += strlen(data_strings[i]) + 1;
    SUM_DATA_LEN = sum_len;
    printf("sum len:%d\n", sum_len);
    clock_t t_sta = clock();
    all_rotations = (Idx_pos *)calloc(sum_len, sizeof(Idx_pos));
    int cnt = 0;
    for(int i = 0; i < n; ++i)
    {
        int len = strlen(data_strings[i]);
        // strings[i] = realloc(strings[i], len + 2);
        data_strings[i][len] = Const::EOS;
        data_strings[i][len + 1] = '\0';
        for(int j = 0; j < len + 1; ++j) all_rotations[cnt++] = (Idx_pos){i, j};
    }
    sorted_rotations = (Idx_pos *)calloc(sum_len, sizeof(Idx_pos));
    tp_rotations = (Idx_pos *)calloc(sum_len, sizeof(Idx_pos));
    // printf("---2---\n");
    sigma_cnt = (int*)calloc(SIGMA_NUM, sizeof(int));
    st = (int*)calloc(SIGMA_NUM, sizeof(int));
    radix_sort(data_strings, 0, sum_len, 0);
    clock_t t_end = clock();
    printf("Constructing L finished. Time:%.3lf(s)\n", (double)(t_end - t_sta) / CLOCKS_PER_SEC);
    // exit(0);
    
    // printf("-------------\n");
    // for(int i = 0; i < sum_len; ++i)
    //     printf("%c\n", get_L(data_strings,sorted_rotations[i]));
    // exit(0);

    calc_C(n, data_strings);
    t_sta = clock();
    puts("Start building suffix tree.");
    tree.buildSuffixTree(sigma, sum_len, sorted_rotations, data_strings);
    t_end = clock();
    printf("Building finished. Time:%.3lf(s)\n", (double)(t_end - t_sta) / CLOCKS_PER_SEC);
}



int SSCard::lf_mapping(int idx, char c)
{
    return std::min(1LL * SUM_DATA_LEN, 1LL * C[c] + tree.dfs_estimate_rk(tree.root, idx - 1, c));
}


QueryResults SSCard::estimate(int sta, int end, int p_sta, char* p)
{
    for(int i = p_sta; i >= 0; --i)
    {
        sta = lf_mapping(sta, p[i]);
        end = lf_mapping(end, p[i]);
        if(sta >= end) return (QueryResults){-1, -1};
    }
    return (QueryResults){sta, end};
}


double SSCard::estimate_with_tree(char* pattern)
{
    TreeResults tret = tree.count_with_tree(pattern);
    int m = strlen(pattern);
    
    if(tret.p_sta == 0) return tret.times;
    if(tret.sta == -1) return 0;

    QueryResults qret = estimate(tret.sta, tret.end, tret.p_sta - 1, pattern);
    double result = std::max(1.0, 1.0 * qret.end - qret.sta);
    return result;
}