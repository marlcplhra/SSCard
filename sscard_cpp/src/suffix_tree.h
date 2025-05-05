#ifndef SUFFIX_TREE_H
#define SUFFIX_TREE_H

#include "common_structs.h"
#include "spline.h"
#include <vector>
#include <cstdlib> 
#include <unordered_map>
#include <fstream>
#include <memory>
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/unordered_map.hpp>


// estimating process
struct TreeResults
{
    int sta, end, p_sta, times;
};


class TreeNode
{
private:
    std::weak_ptr<TreeNode> fa;
    char c_fa;         //char to fa node
    int h;
    std::unordered_map<char, std::shared_ptr<TreeNode> > children;
    int L;
    int R;
    bool leaf;
    int occ_times;
    std::vector<R_child> children_list;
    std::unordered_map<char, std::vector<Pos_rk> > tp_positions;
    std::unordered_map<char, std::vector<Spline> > fun;
    std::unordered_map<char, int> first_pos;

    std::unordered_map<char, bool> need_build;

public:
    TreeNode();
    TreeNode(const std::shared_ptr<TreeNode>& _fa, char _c_fa, int _h, int _L, int _R, bool _leaf);

    void add_to_tp_positions(char c, Pos_rk x);

    template <class Archive>
    void serialize(Archive & archive)
    {
       archive(c_fa, h, children, L, R, leaf, occ_times, children_list, fun, first_pos);       
    }

    friend class SuffixTree;
};


class SuffixTree
{
private:
    int SIGMA_NUM;
    int SUM_DATA_LEN;
    std::vector<char> sigma;
    int tree_height;
    int prune_length;
    int cm;
    double error_bound;

    std::vector<int> building_steps;
    std::vector<int> total_cnt;
    std::vector<bool> occ_times;
    std::vector<int> st_occ;
    int top_occ;
public:

    std::shared_ptr<TreeNode> root;

    SuffixTree();
    SuffixTree(int SIGMA_NUM, int tree_height, int prune_length, int cm, double error_bound);

    void buildSuffixTree(std::vector<char> sigma, int sum_len, Idx_pos sorted_rotations[], char* data_strings[]);
    bool check_leaf(std::shared_ptr<TreeNode> cur_node);
    int get_occ_times(int L, int R, Idx_pos sorted_rotations[]);
    char next_char(int suffix_idx, int p, Idx_pos sorted_rotations[], char* data_strings[]);
    std::shared_ptr<TreeNode> create_node(std::shared_ptr<TreeNode> cur_node, char c, int L, int R);
    void batched_insert(int sum_len, Idx_pos sorted_rotations[], char* data_strings[]);
    char get_L(char *data_strings[], Idx_pos r);
    void get_L_rank_for_chars(std::shared_ptr<TreeNode> cur, Idx_pos sorted_rotations[], char* data_strings[]);
    void build_or_push_up(std::shared_ptr<TreeNode> cur);
    void learn_fun_with_pushup(std::shared_ptr<TreeNode> cur, Idx_pos sorted_rotations[], char* data_strings[]);

    void build_spline(std::shared_ptr<TreeNode> cur, char c);
    void check_spline(std::shared_ptr<TreeNode> cur);

    TreeResults count_with_tree(char* p);
    TreeResults dfs_occ_in_tree(std::shared_ptr<TreeNode> cur, int x, int len, char* p);
    int dfs_estimate_rk(std::shared_ptr<TreeNode> cur, int idx, char c);
    int get_rk_from_fa(std::shared_ptr<TreeNode> cur, int idx, char c);

    template <class Archive>
    void serialize(Archive & archive)
    {
       archive(SIGMA_NUM, SUM_DATA_LEN, tree_height, prune_length, cm, error_bound, root);
    }

    void setFa(std::shared_ptr<TreeNode> cur, std::shared_ptr<TreeNode> fa);
};


#endif