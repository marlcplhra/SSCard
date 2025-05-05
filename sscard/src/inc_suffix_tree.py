class PatternTrieNode:
    def __init__(self):
        self.children = {}              # 'c': node 
        self.idx_list = []
        
class PatternTrie:
    def __init__(self, patterns):
        self.root = PatternTrieNode()
        self.total_nodes = 1
        
        for i, p in enumerate(patterns):
            self.insert(p, i)
          
    def insert(self, s, idx):
        cur = self.root
        for c in s:
            if c not in cur.children:
                new_node = PatternTrieNode()
                cur.children[c] = new_node
            cur = cur.children[c]
         
        cur.idx_list.append(idx)

    def query(self, s):
        cur = self.root
        idx_list = []
        for c in s:
            # print(c)
            if c not in cur.children: break
            # print(cur.idx_list)
            cur = cur.children[c]
            idx_list.extend(cur.idx_list)
        return idx_list

    def query_datastring(self, s):
        m = len(s)
        contain_set = set()
        for i in range(m):
            idx_list = self.query(s[i:])
            for idx in idx_list:
                contain_set.add(idx)
        return contain_set


class SuffixNode:
    def __init__(self, fa, is_leaf, cnt):
        # self.fa = fa
        # self.h = h
        self.children = {}              # 'c': node 
        # self.is_leaf = is_leaf
        self.cnt = cnt
        self.flg = 0
        
        
class SuffixTree:
    def __init__(self, tree_height):
        self.root = SuffixNode(None, True, 0)
        self.tree_height = tree_height
        self.total_nodes = 1
        self.string_num = 0
    
    
    def insert(self, s, string_idx):
        cur = self.root
        h = 0
        for c in s:
            if h >= self.tree_height:
                break
            if c not in cur.children:
                new_node = SuffixNode(cur, True, 0)
                cur.children[c] = new_node
                # cur.is_leaf = False
                self.total_nodes += 1
            cur = cur.children[c]
            if cur.flg != string_idx:
                cur.cnt += 1
                cur.flg = string_idx
            h += 1
    
    def insert_string(self, s):
        self.string_num += 1
        m = len(s)
        for i in range(m):
            self.insert(s[i:], self.string_num)    
    
    def query(self, s):
        cur = self.root
        for c in s:
            if c not in cur.children:
                return 0
            cur = cur.children[c]
        return cur.cnt
    
    def del_tree(self, cur):
        for c, new_node in cur.children.items():
            self.del_tree(new_node)
            # del cur.children[c]
        
        del cur