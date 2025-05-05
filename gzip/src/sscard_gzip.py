import datetime
import bisect
import numpy as np
import os
import pickle
from collections import OrderedDict
import gzip


# from numba import jit


# EOS = "#"
EOS = "\0"
EOS_last = "\1"

flg_for_debug = False
debug_mode = False


def save(filename, model):
    parent_directory = os.path.dirname(filename)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    f = open(filename, 'wb')
    pickle.dump(model, f)

def load(filename):
    f = open(filename, "rb")
    ret = pickle.load(f)
    # estimator.print_tree_info()
    return ret

def gzip_save(filename, model):
    parent_directory = os.path.dirname(filename)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    with gzip.open(filename, "wb") as f:
        pickle.dump(model, f)

def gzip_load(filename):
    with gzip.open(filename, "rb") as f:
        model = pickle.load(f)
    return model


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
    
    # ret = [(get_L(x), x[0], x[1]) for x in sorted_rotations]
    ret = [get_L(x) for x in sorted_rotations]
    return ret, sorting_duration


def calc_C_and_checkpoints(L_array, args):
    """checkpoints up to idx (excluding idx)
    """
    A = {} # letter count
    cnt = 0
    las = 0
    for i, c in enumerate(L_array):
        if i % args.k == 0:
            gzip_save(f"{args.exp_path}checkpoints/checkpoint_{cnt}.pkl.gz", A)
            gzip_save(f"{args.exp_path}checkpoints/L_{cnt}.pkl.gz", L_array[las:i])
            cnt += 1
            las = i
        if A.get(c):
            A[c] += 1
        else:
            A[c] = 1
    gzip_save(f"{args.exp_path}checkpoints/checkpoint_{cnt}.pkl.gz", A)
    gzip_save(f"{args.exp_path}checkpoints/L_{cnt}.pkl.gz", L_array[las:])
    cnt += 1

    # sort the letters
    letters = sorted(A.keys())
    
    # first index of letter
    C = {}
    
    idx = 0
    for c in letters:
        C[c] = idx
        idx += A[c]
    # del A

    return C, cnt


class SSCard():
    
    def __init__(self, data, args):
        self.k = args.k
        self.savepath = args.exp_path
        
        self.data, self.L_time = multi_bwt_radix_sort(data, args)
        
        self.L_len = len(self.data)
        
        self.C, self.checkpoints_num = calc_C_and_checkpoints(self.data, args)

        del self.data
            
    
    def _C(self, qc):
        """ get the first occurance of letter qc in left-column"""
        c = self.C.get(qc)
        if c == None:
            return 0
        return c
    
    
    def _occ(self, idx, qc):
        """ Count the number of a letter upto idx in s using checkpoints.
            (excluding idx)
        """
        
        # find the nearest checkpoint for idx
        check = int((idx + (self.k / 2)) / self.k)
        if check >= self.checkpoints_num:
            check = self.checkpoints_num - 1
        pos = check * self.k
        
        # count of the letter s[idx] upto pos (not included)
        C = gzip_load(f"{self.savepath}checkpoints/checkpoint_{check}.pkl.gz")
        # count = C[check].get(qc)
        count = C.get(qc)
        if count == None:
            count = 0
        
        # range between pos and idx
        
        if pos < idx:
            # r = range(pos, idx)
            r = range(0, idx - pos)
            check += 1
        else:
            # r = range(idx, pos)
            r = range(idx - (pos - self.k), min(self.k, self.L_len - (pos - self.k)))
        
        # count of letters between pos, idx
        cnt = 0
        L = gzip_load(f"{self.savepath}checkpoints/L_{check}.pkl.gz")  
        # print(pos, idx, check)
        for i in r:
            if qc == L[i]:
                cnt += 1
        
        # calculate the letter count upto idx (not included)
        if pos < idx:
            count += cnt
        else:
            count -= cnt
        
        return count   
    

    def _lf(self, idx, qc):
        """ get the nearset lf mapping for letter qc at position idx """
        C = self._C(qc)
        occ = self._occ(idx, qc)
        return C + occ

    
    def estimate_for_single_pattern(self, pattern):
        top = 0
        bot = self.L_len
        for i, qc in enumerate(pattern[::-1]):
            top = self._lf(top, qc)
            bot = self._lf(bot, qc)
            if top >= bot: 
                break
        return max(1, bot - top)