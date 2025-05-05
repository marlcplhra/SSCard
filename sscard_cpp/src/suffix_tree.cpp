#include "suffix_tree.h"
#include <iostream>
#include <queue>
#include <cstring>
#include <assert.h>
#include <fstream>

TreeNode::TreeNode() {}

TreeNode::TreeNode(const std::shared_ptr<TreeNode>& _fa, char _c_fa, int _h, int _L, int _R, bool _leaf): fa(_fa), c_fa(_c_fa), h(_h), L(_L), R(_R), leaf(_leaf)
{
    occ_times = 0;
    children_list.clear();
}


void TreeNode::add_to_tp_positions(char c, Pos_rk x)
{
    if(tp_positions.count(c) > 0) tp_positions[c].push_back(x);
    else 
    {
        std::vector<Pos_rk> v{x};
        tp_positions[c] = v;
    }
}


SuffixTree::SuffixTree() {}

SuffixTree::SuffixTree(int SIGMA_NUM, int tree_height, int prune_length, int cm, double error_bound)
{
    this->SIGMA_NUM = SIGMA_NUM;
    this->tree_height = tree_height;
    this->prune_length = prune_length;
    this->cm = cm;
    this->error_bound = error_bound;

    occ_times.resize(Const::STRINGS_NUM);
    st_occ.resize(Const::STRINGS_NUM);
    top_occ = 0;
}


void SuffixTree::buildSuffixTree(std::vector<char> sigma, int sum_len, Idx_pos sorted_rotations[], char* data_strings[])
{
    this->sigma = sigma;
    SUM_DATA_LEN = sum_len;
    building_steps.resize(sum_len);
    root = std::make_shared<TreeNode>(nullptr, ' ', 0, 0, sum_len, 0);
    batched_insert(sum_len, sorted_rotations, data_strings);
    total_cnt.resize(SIGMA_NUM);
    printf("Start building splines.\n");
    learn_fun_with_pushup(root, sorted_rotations, data_strings);
}


bool SuffixTree::check_leaf(std::shared_ptr<TreeNode> cur_node)
{
    return cur_node->h >= tree_height | cur_node->R - cur_node->L <= prune_length;
}


char SuffixTree::next_char(int suffix_idx, int p, Idx_pos sorted_rotations[], char* data_strings[])
{
    building_steps[suffix_idx] += 1;
    int idx = sorted_rotations[suffix_idx].idx;
    int len = strlen(data_strings[idx]);
    int pos = (sorted_rotations[suffix_idx].pos + p) % len;
    if(building_steps[suffix_idx] > len) return '\0';
    else return data_strings[idx][pos];
}


int SuffixTree::get_occ_times(int L, int R, Idx_pos sorted_rotations[])
{
    for(int i = L; i < R; ++i)
    {
        int idx = sorted_rotations[i].idx;
        if(!occ_times[idx]) 
        {
            occ_times[idx] = 1;
            st_occ[top_occ++] = idx;
        }
    }
    int cnt = top_occ;
    while(top_occ) occ_times[st_occ[--top_occ]] = 0;
    return cnt;
}


struct Queue
{
    std::weak_ptr<TreeNode> cur_node;
    int h, L, R;
    Queue(const std::shared_ptr<TreeNode>& _cur_node, int _h, int _L, int _R): cur_node(_cur_node), h(_h), L(_L), R(_R) {}
};


std::shared_ptr<TreeNode> SuffixTree::create_node(std::shared_ptr<TreeNode> cur_node, char c, int L, int R)
{
    cur_node->children[c] = std::make_shared<TreeNode>(cur_node, c, cur_node->h + 1, L, R, 0);
    cur_node->children_list.push_back(R_child(R, c));
    return cur_node->children[c];
}


void SuffixTree::batched_insert(int sum_len, Idx_pos sorted_rotations[], char* data_strings[])
{
    std::queue<Queue> q;
    q.push(Queue(root, 0, 0, sum_len));
    while(!q.empty())
    {
        Queue tp = q.front(); q.pop();
        std::shared_ptr<TreeNode> cur_node = tp.cur_node.lock();
        int h = tp.h, L = tp.L, R = tp.R;
        cur_node->occ_times = get_occ_times(L, R, sorted_rotations);
        if(check_leaf(cur_node))
        {
            cur_node->leaf = 1;
            continue;
        }
        char cur_c = next_char(L, h, sorted_rotations, data_strings);
        int beg = L;
        for(int i = L + 1; i < R; ++i)
        {
            char c = next_char(i, h, sorted_rotations, data_strings);
            if(c != cur_c)
            {
                if(cur_c != '\0')
                {
                    std::shared_ptr<TreeNode> new_node = create_node(cur_node, cur_c, beg, i);
                    q.push(Queue(new_node, h + 1, beg, i));
                }
                cur_c = c;
                beg = i;
            }
        }
        if(cur_c != '\0')
        {
            std::shared_ptr<TreeNode> new_node = create_node(cur_node, cur_c, beg, R);
            q.push(Queue(new_node, h + 1, beg, R));
        }
    }
}


char SuffixTree::get_L(char *data_strings[], Idx_pos r)
{
    if(r.pos == 0) return Const::EOS;
    else return data_strings[r.idx][r.pos - 1];
}


void SuffixTree::get_L_rank_for_chars(std::shared_ptr<TreeNode> cur, Idx_pos sorted_rotations[], char* data_strings[])
{
    int len = cur->R - cur->L;
    for(int i = 0; i < len; ++i)
    {
        char c = get_L(data_strings, sorted_rotations[cur->L + i]);
        cur->add_to_tp_positions(c, (Pos_rk){cur->L + i, ++total_cnt[c]});
    }
}


void SuffixTree::build_or_push_up(std::shared_ptr<TreeNode> cur)
{
    // printf("TreeNode L:%d R:%d\n", cur->L, cur->R);
    for(char i: sigma)
    // for(int i = 0; i < SIGMA_NUM; ++i)
    // for(const auto& pr: cur->tp_positions)
    {

        // char i = pr.first;
        if(!cur->tp_positions.count(i))
        {
            if(!cur->fa.expired()) cur->fa.lock()->need_build[i] = 1;
            continue;
        }
        int len = cur->tp_positions[i].size();
        bool need_build = cur->need_build[i];
        if(need_build && (len >= cm || cur->fa.expired()))
        {
            if(cur->fa.expired())
            {
                // printf("----%c---\n", i);
                assert(cur->h == 0);
            }
            build_spline(cur, i);
            cur->first_pos[i] = cur->tp_positions[i][0].pos;
            need_build = 0;
        }
        if(cur->fa.expired()) continue;
        std::shared_ptr<TreeNode> fa = cur->fa.lock();
        // if(len >= cm)
        if(!need_build)
        {
            fa->add_to_tp_positions(i, cur->tp_positions[i][0]);
            fa->add_to_tp_positions(i, cur->tp_positions[i].back());
        }
        else
        {
            fa->need_build[i] = 1;
            // fa->add_to_tp_positions(i, cur->tp_positions[i][0]);
            // if(len > 1)
            for(int j = 0; j < len; ++j) 
                fa->add_to_tp_positions(i, cur->tp_positions[i][j]);
        }
    }
    cur->tp_positions.clear();
}


void SuffixTree::learn_fun_with_pushup(std::shared_ptr<TreeNode> cur, Idx_pos sorted_rotations[], char* data_strings[])
{
    if(cur->leaf)
    {
        get_L_rank_for_chars(cur, sorted_rotations, data_strings);
        for(char i: sigma) cur->need_build[i] = 1;
        // for(int i = 0; i < SIGMA_NUM; ++i) cur->need_build[i] = 1;
        build_or_push_up(cur);
        return;
    }
    for(R_child x: cur->children_list)
        learn_fun_with_pushup(cur->children[x.c], sorted_rotations, data_strings);
    build_or_push_up(cur);
}


void SuffixTree::build_spline(std::shared_ptr<TreeNode> cur, char c)
{
    std::vector<Pos_rk>& pr = cur->tp_positions[c];
    // if(pr.size() == 1) printf("@@@@@\n");
    // assert(pr.size() > 1);
    GreedySpline spline = GreedySpline(SUM_DATA_LEN, error_bound, pr);
    cur->fun[c] = spline.build_greedyspline();
}


TreeResults SuffixTree::dfs_occ_in_tree(std::shared_ptr<TreeNode> cur, int x, int len, char* p)
{
    if(cur->leaf || x == len)
        return (TreeResults){cur->L, cur->R, x, cur->occ_times};
    if(!cur->children.count(p[x]))
        return (TreeResults){-1, -1, -1, 0};
    return dfs_occ_in_tree(cur->children[p[x]], x + 1, len, p);
}


TreeResults SuffixTree::count_with_tree(char* p)
{
    int m = strlen(p);
    int sta = std::max(0, m - tree_height);
    for(int i = sta; i < m; ++i)
    {
        TreeResults ret = dfs_occ_in_tree(root, i, m, p);
        if(ret.p_sta == m)
        {
            ret.p_sta = i;
            return ret;
        }
    }
    return (TreeResults){-1, -1, -1, 0};
}


int SuffixTree::get_rk_from_fa(std::shared_ptr<TreeNode> cur, int idx, char c)
{
    while(!cur->fun.count(c) && !cur->fa.expired()) 
        cur = cur->fa.lock();
    if(!cur->fun.count(c)) return 0;
    assert(c < SIGMA_NUM);
    std::vector<Spline>& S = cur->fun[c];

    std::vector<Pos_rk> hhh = cur->tp_positions[c];

    auto buc = std::lower_bound(S.begin(), S.end(), (Spline){idx, 0, 0});
    if(buc == S.end()) buc = std::prev(S.end());

    idx = std::max(idx, cur->first_pos[c]);
    idx = std::min(idx, S.back().pos);
    double k = (*buc).k, b = (*buc).b;
    int count = idx * k + b;
    // int count = int(std::min(1.0 * SUM_DATA_LEN, 1.0 * idx * k + b));
    return count;
}


int SuffixTree::dfs_estimate_rk(std::shared_ptr<TreeNode> cur, int idx, char c)
{
    if(idx <= 0) return 0;
    if(cur->leaf) return get_rk_from_fa(cur, idx, c);
    std::vector<R_child>& Ch = cur->children_list;
    auto child = std::upper_bound(Ch.begin(), Ch.end(), (R_child){idx, (char)(SIGMA_NUM - 1)});
    if(child == Ch.end()) *child = Ch.back();
    return dfs_estimate_rk(cur->children[(*child).c], idx, c);
}


void SuffixTree::setFa(std::shared_ptr<TreeNode> cur, std::shared_ptr<TreeNode> fa)
{
    cur->fa = fa;
    for(R_child x: cur->children_list)
        setFa(cur->children[x.c], cur); 
}