#ifndef SSCARD_H
#define SSCARD_H

#include "common_structs.h"
#include "suffix_tree.h"
#include <vector>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>


struct QueryResults{int sta, end;};


class SSCard
{
private:
    int SIGMA_NUM;
    int SUM_DATA_LEN;
    std::vector<char> sigma;
    // std::vector<int> C;
    std::unordered_map<char, int> C;

    SuffixTree tree;

public:
    SSCard();
    SSCard(int SIGMA_NUM, int tree_height, int prune_length, int cm, double error_bound);

    void build_SSCard(int n, char* data_strings[]);
    void calc_C(int n, char* data_strings[]);
    double estimate_with_tree(char* pattern);
    QueryResults estimate(int sta, int end, int p_sta, char* p);
    int lf_mapping(int idx, char c);

    template <class Archive>
    void serialize(Archive & archive)
    {
        archive(SIGMA_NUM, SUM_DATA_LEN, C, tree);
    }
    void setTreeFa()
    {
        tree.setFa(tree.root, nullptr);
    }
};


#endif /* SSCARD_H */