import sys
import math
import bisect
import heapq
import numpy as np
from queue import Queue

from numba import jit
    

bwt_EOS = "\0"
MAX_NUM = 1000000000


fitting_times = 0
leaf_fitting_times = 0
leaf_fitting_function_num = 0
fitting_function_num = 0
def ret_fitting_info():
    return fitting_function_num, fitting_times, leaf_fitting_times, leaf_fitting_function_num

spline_points = []
def ret_spline_points():
    return spline_points


def compare_first_dimension(item):
    return item[0]


@jit(nopython=True)
def ols(X, Y):
    if len(X) == 1:
        return 0, Y[0]
    offset_x = X[0]
    X = X - offset_x
    offset_y = Y[0]
    Y = Y - offset_y
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    n = len(X)
    tp = (np.sum(X * X) - n * x_mean * x_mean)
    if tp == 0:
        print(len(X))
        print(X)
    k = (np.sum(X * Y) - n * x_mean * y_mean) / (np.sum(X * X) - n * x_mean * x_mean)
    b = y_mean - k * x_mean
    return k, b - (offset_x * k) + offset_y
    # return k, b

def linear_regression(positions, bucket_size):
    global fitting_function_num
    regression_list = []
    # cur_x = [0]
    # cur_y = [0]
    cur_x = []
    cur_y = []
    for (x, y) in positions:
        if len(cur_x) > 0 and (x == cur_x[-1] or y == cur_y[-1]):
            continue
        cur_x.append(x)
        cur_y.append(y)
        # if len(cur_x) == bucket_size + 1:
        if len(cur_x) == bucket_size:
            X = np.array(cur_x)
            Y = np.array(cur_y)
            k, b = ols(X, Y)
            fitting_function_num += 1
            regression_list.append((cur_x[-1], k, b))
            # cur_x = [cur_x[-1]]
            # cur_y = [cur_y[-1]]
            cur_x = []
            cur_y = []
    # if len(cur_x) > 1:
    if len(cur_x) > 0:
        X = np.array(cur_x)
        Y = np.array(cur_y)
        k, b = ols(X, Y)
        fitting_function_num += 1
        regression_list.append((cur_x[-1], k, b))
    return regression_list


class Spline():
    def __init__(self, positions, max_error_):
        self.precision = 1e-16
        self.max_error_ = max_error_
        self.cur_num_keys = 0
        self.curr_num_distinct_keys_ = 0
        self.spline_points_ = []
        self.positions = positions
        
        for i, (x, y) in enumerate(positions):
            self.addKeytoSpline(x, y)
            self.cur_num_keys += 1
            self.prev_key_ = x
            self.prev_position_ = y
        self.spline_points_.append((positions[-1]))
    
    def get_kb(self, x1, y1, x2, y2):
        k = (y2 - y1) / (x2 - x1)
        b = y2 - k * x2
        return k, b
    
    def get_regression_list(self):
        regression_list = []
        for i, (x, y) in enumerate(self.spline_points_):
            if i == 0: continue
            k, b = self.get_kb(self.spline_points_[i - 1][0], self.spline_points_[i - 1][1], x, y)
            regression_list.append((x, k, b))
        return regression_list
    
    
    def try_to_merge(self, p1, p2, sta):
        k, b = self.get_kb(p1[0], p1[1], p2[0], p2[1])
        while self.positions[sta][0] != p2[0]:
            y = k * self.positions[sta][0] + b
            if math.fabs(y - self.positions[sta][1]) > self.max_error_:
                return False
            sta += 1
        return True
    
    
    def merge_splines(self):
        sta = 0
        self.merged_spline_points = []
        for i, point in enumerate(self.spline_points_):
            if i < 2:
                self.merged_spline_points.append(point)
            else:
               pre_point = self.merged_spline_points[-1]
               ppre_point = self.merged_spline_points[-2]
               if self.try_to_merge(ppre_point, point, sta):
                   self.merged_spline_points[-1] = point
               else:
                   self.merged_spline_points.append(point)
                   while self.positions[sta][0] != pre_point[0]:
                       sta += 1
        
        if len(self.spline_points_) != len(self.merged_spline_points):
            # print("merged!", len(self.spline_points_), len(self.merged_spline_points))
            self.spline_points_ = self.merged_spline_points
        else:
            assert self.spline_points_ == self.merged_spline_points
            
    
    def ComputeOrientation(self, dx1, dy1, dx2, dy2):
        expr = dy1 * dx2 - dy2 * dx1
        if expr > self.precision:
            return 0
        elif expr < -self.precision:
            return 2
        return 1
       
    def addKeytoSpline(self, x, y):
        if self.cur_num_keys == 0:
            self.spline_points_.append((x, y))
            self.curr_num_distinct_keys_ += 1
            self.prev_point_ = (x, y)
            return
        
        if x == self.prev_key_:
            return
        
        self.curr_num_distinct_keys_ += 1
        
        if self.curr_num_distinct_keys_ == 2:
            self.upper_limit_ = (x, y + self.max_error_)
            self.lower_limit_ = (x, max(0, y - self.max_error_))
            self.prev_point_ = (x, y)
            return

        last = self.spline_points_[-1]
        upper_y = y + self.max_error_
        lower_y = max(0, y - self.max_error_)
        
        assert self.upper_limit_[0] >= last[0]
        assert self.lower_limit_[0] >= last[0]
        assert x >= last[0]
        upper_limit_x_diff = self.upper_limit_[0] - last[0]
        lower_limit_x_diff = self.lower_limit_[0] - last[0]
        x_diff = x - last[0]
        
        assert self.upper_limit_[1] >= last[1]
        assert y >= last[1]
        upper_limit_y_diff = self.upper_limit_[1] - last[1]
        lower_limit_y_diff  = self.lower_limit_[1] - last[1]
        y_diff = y - last[1]
        
        assert self.prev_point_[0] != last[0]
        
        if self.ComputeOrientation(upper_limit_x_diff, upper_limit_y_diff, \
            x_diff, y_diff) != 0 or self.ComputeOrientation(lower_limit_x_diff, \
            lower_limit_y_diff, x_diff, y_diff) != 2:
            self.spline_points_.append(self.prev_point_)
            
            self.upper_limit_ = (x, upper_y)
            self.lower_limit_ = (x, lower_y)
        else:
            assert upper_y >= last[1]
            upper_y_diff = upper_y - last[1]
            if self.ComputeOrientation(upper_limit_x_diff, upper_limit_y_diff, \
                x_diff, upper_y_diff) == 0:
                self.upper_limit_ = (x, upper_y)
            
            lower_y_diff = lower_y - last[1]
            if self.ComputeOrientation(lower_limit_x_diff, lower_limit_y_diff, \
                x_diff, lower_y_diff) == 2:
                self.lower_limit_ = (x, lower_y)
        
        self.prev_point_ = (x, y)


# def spline_regression(positions, bucket_size):
def spline_regression(positions, error_bound, is_leaf = False):
    global fitting_function_num, leaf_fitting_function_num
    if len(positions) == 1:
        fitting_function_num += 1
        if is_leaf:
            leaf_fitting_function_num += 1
        return [(positions[0][0], 0, positions[0][1])]
    
    spline_builder = Spline(positions, error_bound)
    # spline_builder.merge_splines()
    regression_list = spline_builder.get_regression_list()
    fitting_function_num += len(regression_list)
    if is_leaf:
        leaf_fitting_function_num += len(regression_list)
        # spline_points.extend(spline_builder.spline_points_)
    return regression_list
            

def append_to_dict(_dict, c, val):
    if c not in _dict:
        _dict[c] = []
    _dict[c].append(val)


class TreeNode():
    def __init__(self, node_idx, fa_node, c, h, L, R, is_leaf):
        self.idx = node_idx
        self.fa = fa_node
        self.c = c          # edge to fa_node
        self.h = h
        self.char_to_children = {}
        # self.char_to_substrings = {}        # 'a': (idx, pos, len)
        self.L = L
        self.R = R
        self.is_leaf = is_leaf
        
        self.children_list = []         # (R, c)
        
        self.tp_positions = {}        # for pushup
        self.need_build = set()
        self.char_buckets = {}
        self.first_pos = {}
        
        self.offset = 0


class SuffixTree():
    def __init__(self, sorted_suffixes, strings, args):
        """

        Args:
            sorted_suffixes (list): [(idx, pos), ...]
            strings (list): [s, ...]
            args
        """
        self.regression_method = args.fitting
        self.error_bound = args.e
        self.node_cnt = 1
        self.total_node = 1
        self.prune_length = min(args.l, len(sorted_suffixes) - 1)
        self.regression_method = args.fitting
        if self.regression_method == "spline":
            self.error_bound = args.e
        elif self.regression_method == "linear":
            self.bucket = args.buc
        # self.prune_length = min(500, len(strings) - 1)
        # self.prune_length = 1
        self.root = TreeNode(self.node_cnt, None, None, 0, 0, len(sorted_suffixes), False)
        self.sorted_suffixes = sorted_suffixes
        self.building_steps = np.zeros(len(sorted_suffixes), dtype=int)
        self.strings = strings
        # self.L_id_list = L_id_list
        self.tree_height = args.h
        self.batched_insert()
        del self.building_steps
        # self.print_tree(self.root, None, None)
        print("Used index nums", self.node_cnt)
        print("Total Tree node:", self.total_node)
        
        self.leaves = []
    
    
    def __dfs(self, cur_node):
        self.total_node += 1
        for c, new_node in cur_node.char_to_children.items():
            self.__dfs(new_node)


    def printNode(self, cur_node, fa_c):
        if fa_c == bwt_EOS:
            fa_c = "EOS"
        print("-" * cur_node.h, end="")
            
        if cur_node.fa is not None:
            print(f">{fa_c}: Node {cur_node.idx} h:{cur_node.h}. fa:{cur_node.fa.idx} L:{cur_node.L} R:{cur_node.R}")
        else:
            print(f">{fa_c}: Node {cur_node.idx} h:{cur_node.h}. L:{cur_node.L} R:{cur_node.R}")
        
        
    def print_tree(self, cur_node, fa_c):
        self.printNode(cur_node, fa_c)
        las_R = 0
        for c, new_node in cur_node.char_to_children.items():
            assert las_R < new_node.R
            las_R = new_node.R
            self.print_tree(new_node, c)
  
    
    def special_check_for_EOS(self, cur_node):
        return cur_node.h == 1 and cur_node.c == bwt_EOS
    
    def check_leaf_with_len(self, cur_node):
        return cur_node.h >= self.tree_height or cur_node.R - cur_node.L <= self.prune_length
    
    
    def create_new_node(self, cur_node, c, idx, pos, L, R):
        self.node_cnt += 1
        self.total_node += 1
        if self.node_cnt % 5000 == 0: print("Tree node:", self.node_cnt)
        new_node = TreeNode(self.node_cnt, cur_node, c, cur_node.h + 1, L, R, False)
        cur_node.char_to_children[c] = new_node
        # cur_node.char_to_substrings[c] = (idx, pos, 1)
        cur_node.children_list.append((R, c))
        return new_node
    

    def next_char(self, suffix_idx, p):
        self.building_steps[suffix_idx] += 1
        idx = self.sorted_suffixes[suffix_idx][0]
        # pos = self.sorted_suffixes[suffix_idx][1] + p
        pos = (self.sorted_suffixes[suffix_idx][1] + p) % len(self.strings[idx])
        
        if self.building_steps[suffix_idx] > len(self.strings[idx]):
            return '!empty!', 0, 0
        # if pos >= len(self.strings[idx]):
            # return '!empty!', 0, 0
        # else:
        return self.strings[idx][pos], idx, pos
    
    
    def batched_insert(self):
        que = Queue()
        que.put((self.root, 0, 0, len(self.sorted_suffixes)))    
        while que.empty() == False:
            cur_node, p, L, R = que.get()
            if self.special_check_for_EOS(cur_node):
                cur_node.is_leaf = True
                continue
            if self.check_leaf_with_len(cur_node):                   # pruned with tree height and len
            # if self.check_leaf(cur_node):                   # pruned with tree height
                cur_node.is_leaf = True
                continue
            
            cur_char, cur_idx, cur_pos = self.next_char(L, p)
            beg = L
            idxs = {}
            idxs[self.sorted_suffixes[L][0]] = 1
            for i in range(L + 1, R):
                c, c_idx, c_pos = self.next_char(i, p)
                if c != cur_char:
                    if cur_char != '!empty!':
                        new_node = self.create_new_node(cur_node, cur_char, cur_idx, cur_pos, beg, i)
                        new_node.occ_times = len(idxs) / (i - beg)
                        que.put((new_node, p + 1, beg, i))
                    cur_char = c
                    cur_idx = c_idx
                    cur_pos = c_pos
                    beg = i
                    idxs = {}
                idxs[self.sorted_suffixes[i][0]] = 1
            if cur_char != '!empty!':
                new_node = self.create_new_node(cur_node, cur_char, cur_idx, cur_pos, beg, R)
                new_node.occ_times = len(idxs) / (R - beg)
                que.put((new_node, p + 1, beg, R))
                
    
    def get_L(self, x):
        tp = self.sorted_suffixes[x]
        idx = tp[0]
        pos = tp[1]
        if pos == 0:
            return bwt_EOS
        else:
            return self.strings[idx][pos - 1]
    
    
    def get_L_rank_for_chars(self, cur):
        for i in range(cur.L, cur.R):
            c = self.get_L(i)
            self.total_cnt[c] += 1
            append_to_dict(cur.tp_positions, c, (i, self.total_cnt[c]))
    
    
    def build_or_push_up(self, cur, cm=10):
        for c, positions in cur.tp_positions.items():
            p_len = len(positions)
            
            if c in cur.need_build and (p_len >= cm or cur.fa is None):
                if self.regression_method == "spline":
                    cur.char_buckets[c] = spline_regression(cur.tp_positions[c], self.error_bound)
                elif self.regression_method == "linear":
                    cur.char_buckets[c] = linear_regression(cur.tp_positions[c], self.bucket)
            
            if cur.fa is None: continue   
            if p_len >= cm:
                # assert c in cur.char_buckets
                append_to_dict(cur.fa.tp_positions, c, positions[0])
                append_to_dict(cur.fa.tp_positions, c, positions[-1])
            else:
                cur.fa.need_build.add(c)
                append_to_dict(cur.fa.tp_positions, c, positions[0])
                if len(positions) > 1:
                    cur.fa.tp_positions[c].extend(positions[1:])
        del cur.tp_positions
        del cur.need_build


    def build_or_push_up_new(self, cur, cm=10):
        for c, positions in cur.tp_positions.items():
            p_len = len(positions)
            
            cur_node_need_build = cur.need_build
            if c in cur.need_build and (p_len >= cm or cur.fa is None):
                if self.regression_method == "spline":
                    cur.char_buckets[c] = spline_regression(cur.tp_positions[c], self.error_bound)
                    cur.first_pos[c] = cur.tp_positions[c][0][0]
                elif self.regression_method == "linear":
                    cur.char_buckets[c] = linear_regression(cur.tp_positions[c], self.bucket)
                    cur.first_pos[c] = cur.tp_positions[c][0][0]
                cur_node_need_build = False
            
            if cur.fa is None: continue   
            if cur_node_need_build == False:
                # assert c in cur.char_buckets
                append_to_dict(cur.fa.tp_positions, c, positions[0])
                append_to_dict(cur.fa.tp_positions, c, positions[-1])
            else:
                cur.fa.need_build.add(c)
                append_to_dict(cur.fa.tp_positions, c, positions[0])
                if len(positions) > 1:
                    cur.fa.tp_positions[c].extend(positions[1:])
        del cur.tp_positions
        del cur.need_build
    
    
    def learn_fun_with_pushup(self, cur):
        if cur.is_leaf:
            self.get_L_rank_for_chars(cur)
            for c in self.C:
                cur.need_build.add(c)
            self.build_or_push_up_new(cur)
            return
        
        for c, new_node in cur.char_to_children.items():
            self.learn_fun_with_pushup(new_node)
        
        self.build_or_push_up_new(cur) 
        
    
    def direct_build(self, cur):
        for c in self.C:
            if c not in cur.tp_positions:
                cur.tp_positions[c] = [(cur.R - 1, self.total_cnt[c])]
            cur.char_buckets[c] = spline_regression(cur.tp_positions[c], self.error_bound)
        del cur.tp_positions
        del cur.need_build
    
    
    def learn_fun_wo_pushup(self, cur):
        if cur.is_leaf:
            self.get_L_rank_for_chars(cur)
            self.direct_build(cur)
            return
        for c, new_node in cur.char_to_children.items():
            self.learn_fun_wo_pushup(new_node)
            

    def __get_buc_from_fa(self, cur_node, idx, qc):
        
        idx -= cur_node.offset
        
        while qc not in cur_node.char_buckets and cur_node.fa is not None:
            cur_node = cur_node.fa
        if qc not in cur_node.char_buckets:
            return 0
        buc = bisect.bisect_left(cur_node.char_buckets[qc], (idx, 0, 0))
        if buc == len(cur_node.char_buckets[qc]):
            buc -= 1
        idx = max(idx, cur_node.first_pos[qc])
        # idx = max(idx, cur_node.char_buckets[qc][0][0])
        idx = min(idx, cur_node.char_buckets[qc][buc][0])
        _, k, b = cur_node.char_buckets[qc][buc]
        count = int(k * idx + b)
        return count
    
    
    def __dfs_suffix_occ(self, cur_node, p):
        if cur_node.is_leaf == True or p == "":
            return cur_node.L, cur_node.R, p, cur_node.occ_times
        
        if p[0] not in cur_node.char_to_children:
            return -1, -1, p, 0
        return self.__dfs_suffix_occ(cur_node.char_to_children[p[0]], p[1:])
    
    
    def count_suffix_occ(self, p):
        if len(p) > self.tree_height:
            suffix_p = p[len(p) - self.tree_height:]
            ret_p = p[:len(p) - self.tree_height]
        else:
            suffix_p = p
            ret_p = ""
        L, R, remain_p = self.__dfs_suffix_occ(self.root, suffix_p)
        # if remain_p != "":
        #     print(p, remain_p)
        return L, R, ret_p


    def count_suffix_occ_h2(self, p):
        sta = max(0, len(p) - self.tree_height)
        for i in range(sta, len(p)):
            L, R, remain_p, times = self.__dfs_suffix_occ(self.root, p[i:])
            if remain_p == "":
                return L, R, p[:i], times
        return -1, -1, p, 0
    
    
    def __dfs_occ(self, cur_node, idx, qc):
        if cur_node.is_leaf:
            count = self.__get_buc_from_fa(cur_node, idx, qc)
            return count
        # print(cur_node.children_list)
        # for i, (c, new_node) in enumerate(cur_node.char_to_children.items()):
        #     if new_node.R > idx or i == len(cur_node.char_to_children) - 1:
        #         # print(new_node.R, idx)
        #         return self.__dfs_occ(new_node, idx, qc)
        pos = bisect.bisect_right(cur_node.children_list, (idx, 'z'))
        if pos == len(cur_node.children_list):
            pos -= 1
        c = cur_node.children_list[pos][1]
        new_node = cur_node.char_to_children[c]
        return self.__dfs_occ(new_node, idx, qc)
    
    def count_occ(self, idx, qc):
        count = self.__dfs_occ(self.root, idx, qc)
        return count
