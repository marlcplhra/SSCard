import pickle
import datetime
import bisect
import numpy as np
import os

from collections import OrderedDict

from suffix_tree_sscard import SuffixTree
from suffix_tree_sscard import spline_regression


# EOS = "#"
EOS = "\0"
EOS_last = "\1"

flg_for_debug = False
debug_mode = False


def save(filename, idx):
    parent_directory = os.path.dirname(filename)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    f = open(filename, 'wb')
    pickle.dump(idx,f)


def load(filename):
    f = open(filename, "rb")
    ret = pickle.load(f)
    # estimator.print_tree_info()
    return ret


data_strings = []
    

def get_L(x):
    idx = x[0]
    pos = x[1]
    if pos == 0:
        return EOS
    else:
        return data_strings[idx][pos - 1]


def get_char(idx, pos):
    if pos >= len(data_strings[idx]):
        # print(pos, len(data_strings[idx]))
        return data_strings[idx][pos % len(data_strings[idx])]
    else:
        return data_strings[idx][pos]


def radix_sort(rotations, sorted_rotations, h):
    if len(rotations) == 1 or h == len(data_strings[rotations[0][0]]) * 2:
        # rotations = [(x[0], x[1]) for x in rotations]
        sorted_rotations.extend(rotations)
        return
    char_dict = {}
    for (idx, pos) in rotations:    
        c = get_char(idx, pos + h)
        if c not in char_dict:
            char_dict[c] = [(idx, pos)]
        else:
            char_dict[c].append((idx, pos))
    char_dict = OrderedDict(sorted(char_dict.items()))
    for (key, val) in char_dict.items():
        radix_sort(val, sorted_rotations, h + 1)


def multi_bwt_radix_sort(strings, args):
    """
    sort all cyclic shifts with the enhanced comparator mentioned in the paper

    Args:
        strings(tuple): set of data strings
    Return:
        ret(list): ret[i] = (L[i], idx)
    """
    global data_strings, string_length

    data_strings = [(s + EOS) for s in strings]

    L_path = os.path.join(args.load_path, "L")
    if args.load_L and os.path.exists(L_path):
        sorted_rotations = load(L_path)
        print("Loading L finished.")
        sorting_duration = 0
    else:
        beg = datetime.datetime.now()
        all_rotations = []
        for i, s in enumerate(strings):
            s += EOS
            rotation = [(i, j) for j in range(len(s))]
            all_rotations.extend(rotation)
        sorted_rotations = []
        radix_sort(all_rotations, sorted_rotations, 0)
        end = datetime.datetime.now()
        sorting_duration = end - beg
        print("Constructing L finished:", sorting_duration)
        if args.load_L:
            save(L_path, sorted_rotations)
    
    ret = [(get_L(x), x[0], x[1]) for x in sorted_rotations]
    # for i, x in enumerate(ret): print(i, x)
    
    mytree = SuffixTree(sorted_rotations, data_strings, args)
    print("Building tree finished.")
    
    return ret, mytree, sorting_duration


def multi_bwt_radix_sort_no_tree(strings, args):
    """
    Args:
        strings(tuple): set of data strings
    Return:
        ret(list): ret[i] = (L[i], idx)
    """
    global data_strings, string_length

    data_strings = [(s + EOS) for s in strings]

    L_path = os.path.join(args.load_path, "L")
    if args.load_L and os.path.exists(L_path):
        sorted_rotations = load(L_path)
        print("Loading L finished.")
        sorting_duration = 0
    else:
        beg = datetime.datetime.now()
        all_rotations = []
        for i, s in enumerate(strings):
            s += EOS
            rotation = [(i, j) for j in range(len(s))]
            all_rotations.extend(rotation)
        sorted_rotations = []
        radix_sort(all_rotations, sorted_rotations, 0)
        end = datetime.datetime.now()
        sorting_duration = end - beg
        print("Constructing L finished:", sorting_duration)
        if args.load_L:
            save(L_path, sorted_rotations)
    
    L = [(get_L(x), x[0], x[1]) for x in sorted_rotations]

    beg = datetime.datetime.now()
    occ_dict = {}
    for i, (c, idx, pos) in enumerate(L):
        if c == "\0": continue
        if c not in occ_dict:
            occ_dict[c] = [(i, 1)]
        else:
            count = len(occ_dict[c])
            occ_dict[c].append((i, count + 1))
    splines = {}
    fitting_function_num = 0
    for key, value in occ_dict.items():
        print("Spline interpolation", key)
        splines[key] = spline_regression(value, args.e, False)
        fitting_function_num += len(splines[key])
    end = datetime.datetime.now()
    spline_time = end - beg
    return L, splines, sorting_duration, spline_time, fitting_function_num
    


def calc_C(s):
    """ calculate the first occurance of a letter in sorted string s """
    A = {}
    for i, (c, idx, pos) in enumerate(s):
        if A.get(c):
            A[c] += 1
        else:
            A[c] = 1
    
    # sort the letters
    letters = sorted(A.keys())
    
    # first index of letter
    C = {}
    
    idx = 0
    for c in letters:
        C[c] = idx
        idx += A[c]
    # del A
    
    return C, A
    

class SSCard():
    """
        The fm-index part refers to https://github.com/egonelbre/fm-index
    
    """
    def __init__(self, data, args):
        self.bucket_size = args.buc
        
        self.h = args.h          # suffix tree height
        self.data_string_len = len(data)
        
        self.data, self.myTree, self.L_time = multi_bwt_radix_sort(data, args)
        
        self.C, self.char_cnt = calc_C(self.data)
        self.L_len = len(self.data)
       
        self.myTree.C = self.C
        self.myTree.total_cnt = {}
        for c in self.C:
            self.myTree.total_cnt[c] = 0
            
        self.myTree.bucket = args.buc
        
        beg = datetime.datetime.now()
        if args.pushup:
            print("Start learning functions and pushup...")
            self.myTree.learn_fun_with_pushup(self.myTree.root)
        else:
            print("Start learning functions")
            self.myTree.learn_fun_wo_pushup(self.myTree.root)
        end = datetime.datetime.now()
        self.spline_time = end - beg
        print("Building spline finished. Time:", end - beg)
        del self.myTree.strings
        del self.myTree.sorted_suffixes
        del self.myTree.C
        del self.data
        del self.myTree.total_cnt
            
    
    def _C(self, qc):
        """ get the first occurance of letter qc in left-column"""
        c = self.C.get(qc)
        if c == None:
            return 0
        return c
    

    def _lf(self, idx, qc, occ_method):
        """ get the nearset lf mapping for letter qc at position idx """
        C = self._C(qc)
        occ = self._occ(idx, qc, method=occ_method)
        return C + occ
       
        
    def _occ(self, idx, qc, method):
        """ count the occurances of letter qc (rank of qc) upto position idx (excluding qc)"""
        if idx <= 0:
            count = 0
        else:
            if qc not in self.C:
                return 0
            count = self.myTree.count_occ(idx - 1, qc)
            count = min(max(count, 0), self.char_cnt[qc])
                
        return count
    
    
    def __count_for_single_pattern(self, top, bot, q, occ_method):
        global flg_for_debug
        # top = 0
        # bot = self.L_len
        if flg_for_debug:
            print("  :", top, bot, bot - top)
        for i, qc in enumerate(q[::-1]):
            top = self._lf(top, qc, occ_method)
            bot = self._lf(bot, qc, occ_method)
            if flg_for_debug:
                print(qc, ":", top, bot, bot - top, self._C(qc))
            if top >= bot: 
                return (-1, -1)
        if flg_for_debug:
            flg_for_debug = False
        return (top,bot)
    
    
    def estimate_for_single_pattern(self, pattern, occ_method):
        top, bot = self.__count_for_single_pattern(0, self.L_len, pattern, occ_method)
        c = bot - top
        if c < 1:
            c = 1
        return c
    
    def estimate_for_single_pattern_with_suffix_tree(self, pattern, occ_method):
        original_pattern = pattern
        top_trie, bot_trie, pattern, times = self.myTree.count_suffix_occ_h2(pattern)
        if pattern == "":
            return times * (bot_trie - top_trie), False
        if top_trie != -1:
            # bot_trie = top_trie + (bot_trie - top_trie) * times
            top, bot = self.__count_for_single_pattern(top_trie, bot_trie, pattern, occ_method)
        else:
            top = -1
            bot = -1
        
        flg = False
        c = bot - top
        # if c < 1:
        #     c = 1
        return c, flg