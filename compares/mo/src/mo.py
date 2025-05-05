import heapq

class PSTNode:
    def __init__(self, substring):
        self.substring = substring
        self.count = 0
        self.fa = None
        self.c = None
        self.children = {}
    
    def __lt__(self, other):
        return self.count < other.count

class PrunedSuffixTree:
    def __init__(self, data_len, prune_threshold=None):
        self.root = PSTNode("")
        self.total_cnt = data_len
        self.total_node = 0
        if prune_threshold:
            self.prune_threshold = prune_threshold

    def insert(self, string):
        seen_substrings = set()
        for i in range(len(string)):
            for j in range(i + 1, len(string) + 1):
                substr = string[i:j]
                if len(substr) > 20: break
                if substr not in seen_substrings:
                    self._insert_substring(substr)
                    seen_substrings.add(substr)

    def _insert_substring(self, substring):
        node = self.root
        for c in substring:
            if c not in node.children:
                new_node = PSTNode(node.substring + c)
                node.children[c] = new_node
                new_node.fa = node
                new_node.c = c
                self.total_node += 1
            node = node.children[c]
        node.count += 1

    def prune(self):
        def _prune_recursive(node):
            to_delete = []
            for c, child in node.children.items():
                _prune_recursive(child)
                if child.count < self.prune_threshold:
                    to_delete.append(c)
            for c in to_delete:
                del node.children[c]
        _prune_recursive(self.root)

    def prune_by_top_k_percent(self, top_k_percent=5):
        remain_node = int(self.total_node * top_k_percent / 100)
        
        remain_node = min(remain_node, 1000000)
        
        min_heap = []
        
        def collect_leaves(node):
            if len(node.children) == 0:
                heapq.heappush(min_heap, node)
                return
            for child in node.children.values():
                collect_leaves(child)
        collect_leaves(self.root)
        
        while len(min_heap) > 0 and self.total_node > remain_node:
            node = heapq.heappop(min_heap)
            if node.fa:
                del node.fa.children[node.c]
                if len(node.fa.children) == 0:
                    heapq.heappush(min_heap, node.fa)
            del node
            self.total_node -= 1
        
        node = min_heap[0]
        self.prune_threshold = node.count
        

def mo_estimate(pst: PrunedSuffixTree, query: str):
    i = 0
    substrings = []
    while i < len(query):
        node = pst.root
        j = i
        last_found = None
        while j < len(query) and query[j] in node.children:
            node = node.children[query[j]]
            j += 1
            last_found = node
        
        if last_found is None:
            substrings.append(PSTNode(query[i]))
            substrings[-1].count = pst.prune_threshold
        elif len(substrings) == 0 or (last_found.substring not in substrings[-1].substring):
            substrings.append(last_found)
            
        i += 1
    
    # subs = [x.substring for x in substrings]
    # print(subs)

    prob = substrings[0].count / pst.total_cnt
    for k in range(1, len(substrings)):
        overlap = find_overlap(substrings[k - 1].substring, substrings[k].substring)
        if overlap:
            joint = overlap
            prev_count = get_node_count(pst, joint)
            prob *= substrings[k].count / prev_count
        else:
            prob *= substrings[k].count / pst.total_cnt
    # return prob

    # cnt = max(1, prob * pst.total_cnt)
    cnt = prob * pst.total_cnt
    return cnt


def find_overlap(a, b):
    for i in range(1, min(len(a), len(b)) + 1):
        if a[-i:] == b[:i]:
            return a[-i:]
    return ""

def get_node_count(pst, substring):
    node = pst.root
    for c in substring:
        if c not in node.children:
            return pst.prune_threshold
        node = node.children[c]
    return node.count