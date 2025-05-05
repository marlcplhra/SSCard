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


char32_t get_char(char32_t *strings[], int idx, int pos)
{
    // int m = strlen(strings[idx]);
    int m = calc_len(strings[idx]);
    // if(pos >= m) return strings[idx][pos % m];
    if(pos >= m) return strings[idx][pos - m];
    else return strings[idx][pos];
}

char32_t get_L(char32_t *strings[], Idx_pos r)
{
    if(r.pos == 0) return Const::EOS;
    else return strings[r.idx][r.pos - 1];
}

int sorted_cnt = 0;
Idx_pos *tp_rotations = NULL;
Idx_pos *sorted_rotations = NULL;
Idx_pos *all_rotations = NULL;
// int *sigma_cnt;
std::unordered_map<char32_t, int> sigma_cnt;
// int *st, top = 0;
char32_t* st;
int top = 0;
void radix_sort(char32_t *strings[], const int& L, const int& R, const int& h)  //[L,R)
{
    // printf("==L:%d R:%d\n", L, R);
    if(R - L == 1 | h == (calc_len(strings[all_rotations[0].idx]) << 1))
    {
        memcpy(sorted_rotations + sorted_cnt, all_rotations + L, (R - L) * sizeof(Idx_pos));
        sorted_cnt += R - L;
        return;
    }
    for(int i = L; i < R; ++i)
    {
        char32_t c = get_char(strings, all_rotations[i].idx, all_rotations[i].pos + h);
        if(sigma_cnt[c] == 0) st[top++] = c;
        sigma_cnt[c]++;
    }
    std::sort(st, st + top); 
    int *buc = (int *)calloc((top + 1), sizeof(int));
    if (buc == NULL) {
        fprintf(stderr, "Memory allocation failed for buc! Size: %zu bytes\n",
                (size_t)(top + 1) * sizeof(int));
        exit(1);
    }
    int las = 0, buc_len = top;
    for(int i = 0; i < top; ++i)        // 现在sigma['c']表示'c'在这一轮的起始位置
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
        char32_t c = get_char(strings, all_rotations[i].idx, all_rotations[i].pos + h);
        tp_rotations[L + sigma_cnt[c]] = all_rotations[i];
        sigma_cnt[c]++;
    }
    memcpy(all_rotations + L, tp_rotations + L, (R - L) * sizeof(Idx_pos));
    while(top) sigma_cnt[st[--top]] = 0;

    for(int i = 0; i < buc_len; ++i)
        radix_sort(strings, buc[i], buc[i + 1], h + 1);
    
    delete[] buc;
}


void SSCard::calc_C(int n, char32_t* data_strings[])
{
    for(int i = 0; i < n; ++i)
    {
        int m = calc_len(data_strings[i]);
        for(int j = 0; j < m; ++j)
        {
            char32_t c = data_strings[i][j];
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
    for(char32_t c: sigma)
    {
        int tp = C[c];
        C[c] = las;
        las += tp;
    }
}

void SSCard::build_SSCard(int n, char32_t* data_strings[])
{
    int sum_len = 0;
    // int n = sizeof(strings) / sizeof(strings[0]);
    for(int i = 0; i < n; ++i) sum_len += calc_len(data_strings[i]) + 1;
    SUM_DATA_LEN = sum_len;
    printf("sum len:%d\n", sum_len);
    clock_t t_sta = clock();
    // exit(0);
    all_rotations = (Idx_pos *)calloc(sum_len, sizeof(Idx_pos));
    int cnt = 0;
    for(int i = 0; i < n; ++i)
    {
        int len = calc_len(data_strings[i]);
        // strings[i] = realloc(strings[i], len + 2);
        data_strings[i][len] = Const::EOS;
        data_strings[i][len + 1] = U'\x00';
        for(int j = 0; j < len + 1; ++j) all_rotations[cnt++] = (Idx_pos){i, j};
    }
    sorted_rotations = (Idx_pos *)calloc(sum_len, sizeof(Idx_pos));
    tp_rotations = (Idx_pos *)calloc(sum_len, sizeof(Idx_pos));
    // sigma_cnt = (int*)calloc(SIGMA_NUM, sizeof(int));
    st = (char32_t*)calloc(SIGMA_NUM, sizeof(char32_t));
    radix_sort(data_strings, 0, sum_len, 0);
    clock_t t_end = clock();
    printf("Constructing L finished. Time:%.3lf(s)\n", (double)(t_end - t_sta) / CLOCKS_PER_SEC);

    calc_C(n, data_strings);
    t_sta = clock();
    puts("Start building suffix tree.");
    tree.buildSuffixTree(sigma, sum_len, sorted_rotations, data_strings);
    t_end = clock();
    printf("Building finished. Time:%.3lf(s)\n", (double)(t_end - t_sta) / CLOCKS_PER_SEC);
}



int SSCard::lf_mapping(int idx, char32_t c)
{
    return std::min(1LL * SUM_DATA_LEN, 1LL * C[c] + tree.dfs_estimate_rk(tree.root, idx - 1, c));
}


QueryResults SSCard::estimate(int sta, int end, int p_sta, char32_t* p)
{
    for(int i = p_sta; i >= 0; --i)
    {
        sta = lf_mapping(sta, p[i]);
        end = lf_mapping(end, p[i]);
        if(sta >= end) return (QueryResults){-1, -1};
    }
    return (QueryResults){sta, end};
}


double SSCard::estimate_with_tree(char32_t* pattern)
{
    TreeResults tret = tree.count_with_tree(pattern);
    int m = calc_len(pattern);
    
    if(tret.p_sta == 0) return tret.times;
    if(tret.sta == -1) return 1;

    QueryResults qret = estimate(tret.sta, tret.end, tret.p_sta - 1, pattern);
    double result = std::max(1.0, 1.0 * qret.end - qret.sta);
    return result;
}