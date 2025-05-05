import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import math
import numpy as np


from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.colors import to_rgba, rgb_to_hsv, hsv_to_rgb


def darken_color_hsv(color, factor=0.7):
    rgb = to_rgba(color)[:3] 
    hsv = rgb_to_hsv(rgb)
    hsv[2] *= factor 
    return hsv_to_rgb(hsv)


def blend_with_gray(color, factor=0.5):
    rgb = np.array(to_rgba(color)[:3]) 
    gray = np.array([0.5, 0.5, 0.5]) 
    return (1 - factor) * rgb + factor * gray


def format_func2(value, tick_position):
    return f'{value:.2f}' 

def format_func(value, tick_position):
    return f'{value:.1f}' 

def format_func0(value, tick_position):
    return f'{value:.0f}' 

def spline_painter_new_with_max_subplots():
    filepath = "../figures/"
    dnames = ["DBLP-AN", "IMDB-AN", "IMDB-MT", "TPCH-PN", "WIKI"]
    x_linear = []
    y_linear = []
    x_spline = []
    y_spline = []
    max_linear = []
    max_spline = []
    # DBLP_AN
    x_linear.append([3.6270, 4.33998, 5.90448, 9.09011, 15.3534])
    y_linear.append([4.1394, 3.96181, 3.61231, 3.09355, 2.47572])
    x_spline.append([3.6541, 4.39226, 5.94193, 9.12065, 14.9700])
    y_spline.append([3.2393, 2.92334, 2.57316, 2.18923, 1.82913])
    max_linear.append([1990.0, 1991.0, 2090.0, 2041.0, 1270.0])
    max_spline.append([141.0, 64.0, 35.0, 20.0, 11.0])
    
    # IMDB_AN
    x_linear.append([3.74697, 4.3528, 5.36696, 7.52201, 12.48910])
    y_linear.append([3.09617, 2.9451, 2.87047, 2.48677, 2.12488])
    x_spline.append([3.71346, 4.3019, 5.36576, 7.54434, 12.3715])
    y_spline.append([2.24108, 2.0489, 1.86034, 1.67262, 1.49806])
    max_linear.append([3592.0, 3604.0, 3927.0, 3292.0, 1837.0])
    max_spline.append([119.0, 70.0, 37.0, 24.0, 15.0])

    # IMDB_MT
    x_linear.append([4.13115, 4.76682, 5.98327, 8.71662, 14.6246])
    y_linear.append([2.97760, 2.90302, 2.71840, 2.32811, 1.99631])
    x_spline.append([4.08057, 4.73784, 6.02172, 8.77024, 14.51641])
    y_spline.append([2.33548, 2.13813, 1.92016, 1.698112, 1.502944])
    max_linear.append([6057.0, 6057.0, 6057.0, 1086.0, 3044.0])
    max_spline.append([143.0, 71.0, 40.0, 21.0, 12.0])

    # TPCH_PN
    x_linear.append([0.32080, 0.33819, 0.50141, 1.0699, 2.95406])
    y_linear.append([1928.3220, 1928.3220, 1869.7406, 1869.7375, 1713.6416])
    x_spline.append([0.31871, 0.36994, 0.53325, 1.07616, 2.91033])
    y_spline.append([1.00016, 1.00016, 1.00016, 1.00016, 1.00016])
    max_linear.append([21239.0, 21239.0, 21239.0, 21239.0, 21239.0])
    max_spline.append([1.0235, 1.0235, 1.0235, 1.0235, 1.0235])
    
    # WIKI
    x_linear.append([18.0869, 23.9140, 34.6829, 51.9197, 92.5138])
    y_linear.append([8.96920, 8.7310, 8.3102, 6.6148, 4.6890])
    x_spline.append([18.2953, 23.0627, 32.6633, 52.91644, 97.27057])
    y_spline.append([3.04790, 2.6126, 2.2258, 1.89442, 1.62710])
    max_linear.append([45417.0, 126541.666, 50139.0, 84035.0, 33663.0])
    max_spline.append([136.0, 73.0, 38.0, 21.0, 11.0])
    
    max_linear = [[math.log10(x) for x in tp] for tp in max_linear]
    max_spline = [[math.log10(x) for x in tp] for tp in max_spline]
    
    avg_ylims = [(1.5, 5), (1.4, 4), (1.4, 3.1), (-70, 2001), (1.4, 10)]
    max_ylims = [(0.1, 3.7), (0, 3.8), (0, 3.9), (-0.1, 5), (0, 5.3)]

    avg_yticks = [np.linspace(x[0]+0.1, x[-1]-0.1, 5) for x in avg_ylims]
    avg_yticks[3] = np.linspace(avg_ylims[3][0]+70, avg_ylims[3][-1]-1, 5)
    max_yticks = [np.linspace(x[0]+0.1, x[-1]-0.1, 5) for x in max_ylims]
    
    xlims = [(3, 15), (4, 12), (3, 15), (0.3, 2.7), (20, 100)]
    xticks = [np.linspace(x[0], x[-1], 5) for x in xlims]
    # print(xticks)
    # exit()
    
    # darken_cols = [darken_color_hsv('#1f77b4', 0.8), darken_color_hsv('#ff7f0e', 0.8)]
    darken_cols = [blend_with_gray('#1f77b4', 0.2), blend_with_gray('#ff7f0e', 0.2)]
    
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 4), tight_layout=False)
    fig.suptitle(' ', fontsize=18)

    for fi in range(0, 5):
        if fi == 0:
            axes[fi].plot(x_linear[fi], y_linear[fi], label='Linear fitting(avg.)', marker='D', markersize=8, linestyle='--', linewidth=2.5, color=darken_cols[0])
            axes[fi].plot(x_spline[fi], y_spline[fi], label='Spline interpolation(avg.)', marker='s', markersize=8, linewidth=2.5, color=darken_cols[0])
        else:
            axes[fi].plot(x_linear[fi], y_linear[fi], marker='D', markersize=8, linestyle='--', linewidth=2.5, color=darken_cols[0])
            axes[fi].plot(x_spline[fi], y_spline[fi], marker='s', markersize=8, linewidth=2.5, color=darken_cols[0])  
        
        axes[fi].set_xlabel('Size(MB)', fontsize=18)
        axes[fi].set_title(dnames[fi], fontsize=20)
        axes[fi].set_ylim(avg_ylims[fi]) 
        axes[fi].set_yticks(avg_yticks[fi])
        axes[fi].set_xticks(xticks[fi])
        axes[fi].tick_params(axis='x', labelsize=18)
        axes[fi].tick_params(axis='y', labelcolor=darken_cols[0], labelsize=18) 
        if fi == 3:
            axes[fi].yaxis.set_major_formatter(FuncFormatter(format_func0))
        else:
            axes[fi].yaxis.set_major_formatter(FuncFormatter(format_func))
        if fi == 0 or fi == 2:
            axes[fi].set_xlim(2.9, 15.4)
        
        axes[fi].grid(True, linestyle='--')
        

        ax2 = axes[fi].twinx() 


        if fi == 0:
            ax2.plot(x_linear[fi], max_linear[fi], label='Linear fitting(max.)', marker='D', markersize=8, linestyle='--', linewidth=2.5, color=darken_cols[1]) 
            ax2.plot(x_spline[fi], max_spline[fi], label='Spline interpolation(max.)', marker='s', markersize=8, linewidth=2.5, color=darken_cols[1])
        else:
            ax2.plot(x_linear[fi], max_linear[fi], marker='D', markersize=8, linestyle='--', linewidth=2.5, color=darken_cols[1])  
            ax2.plot(x_spline[fi], max_spline[fi], marker='s', markersize=8, linewidth=2.5, color=darken_cols[1])
        ax2.tick_params(axis='y', labelcolor=darken_cols[1], labelsize=12)
        if fi == 4:
            ax2.set_ylabel('Q-error(max.)(log10)', fontsize=18) 
        ax2.set_ylim(max_ylims[fi])
        ax2.set_yticks(max_yticks[fi])
        ax2.tick_params(axis='x', labelsize=18)
        ax2.tick_params(axis='y', labelsize=18)
        ax2.yaxis.set_major_formatter(FuncFormatter(format_func))
        # ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
    
    axes[0].set_ylabel('Q-error(avg.)',fontsize=18) 
    
    # fig.legend(loc='upper center', ncol=5, frameon=False, fontsize=18)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=5, fontsize=18, frameon=False)
    fig.tight_layout()
    
    filename = 'spline_subplots' 
    plt.savefig(filepath + filename + '_final.png', dpi=500)


class CustomTickFormatter(ticker.ScalarFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x, pos=None):
        if x >= 10000:
            return rf'$10^{{{int(x / 1000)}}}$'
        else:
            return rf'${{{int(x)}}}$'


def load_error_results(method, dname):
    filename = "./data/" + dname + "/results_per_length/" + method + ".csv"
    df = pd.read_csv(filename)
    df['length'] = df['length'].astype(int)
    df['Q_error'] = df['Q_error'].astype(float)
    df[''] = df['MAE'].astype(float)
    x = df['length'].tolist()
    y = df['Q_error'].tolist()
    y_MAE = df['MAE'].tolist()
    return x, y


def load_estimation_results(mname, dname):
    if mname not in ['MO', 'LBS', 'Astrid', 'DREAM', 'LPLM', 'SSCard']:
        print("Valid method: ", mname)
        exit()
    filepath = "../../results/" + dname + "/" + mname + ".csv"
    strings = []
    sels = []
    counts = []
    latencies = []
    
    if mname in ['DREAM', 'LBS']:
        df = pd.read_csv(filepath, na_values=[], keep_default_na=False)
        strings = df['s_q'].astype(str).tolist()
        sels = df['true_count'].astype(float).tolist()
        counts = df['est_count'].astype(float).tolist()
        latencies = df['latency'].astype(float).tolist()
    else:
    
        with open(filepath, "r") as f:
            first_line = True
            for line in f.readlines():
                if first_line:
                    first_line = False
                    continue
                # if dname != 'WIKI' and mname in ['LPLM']:
                if mname in ['LPLM']:
                    if dname == 'WIKI':
                        tmp_list = line.strip().rsplit(',', 4)
                        sels.append(float(tmp_list[1]))
                        counts.append(float(tmp_list[2]))
                        latencies.append(float(tmp_list[3]) + float(tmp_list[4]))
                    else:
                        tmp_list = line.strip().rsplit(',', 3)
                        tp2 = tmp_list[3].rsplit('.', 1)            # A comma is missing in the data...
                        # print(tp2)
                        tmp_list.append(tp2[0][-1] + '.' + tp2[1])
                        tmp_list[3] = tp2[0][:-1]
                        # print(tmp_list)
                        sels.append(float(tmp_list[1]))
                        counts.append(float(tmp_list[2]))
                    latencies.append(float(tmp_list[3]) + float(tmp_list[4]))
                else:
                    tmp_list = line.strip().rsplit(',', 3)
                    sels.append(float(tmp_list[1]))
                    counts.append(float(tmp_list[2]))
                    latencies.append(float(tmp_list[3]))
                
                strings.append(tmp_list[0])

            
    return strings, sels, counts, latencies


def cal_q_error_per_length(strings, sels, counts):
    q_error_length = {}
    for i, (s, n, c) in enumerate(zip(strings, sels, counts)):
        if len(s) > 12:
            continue
        c = max(c, 1)
        assert n > 0
        q_error = max(n / c, c / n)
        # if q_error > 1000:
            # print(s, n, c, q_error)
        if len(s) not in q_error_length:
            q_error_length[len(s)] = []
        q_error_length[len(s)].append(q_error)
        
        if len(s) > 8:
            print(s)
        
    x = []
    y = []
    for i in range(15):
        if i in q_error_length:
            x.append(i)
            y.append(np.mean(q_error_length[i]))
    for (xx, yy) in zip(x, y):
        print(xx, yy)
    return x, y

    
def error_per_length_painter_subplots():
    filepath = "../figures/"
    filename = 'Q_error_per_length_subplots_add_MO'
    methods = ['MO', 'LBS', 'Astrid', 'DREAM', 'LPLM', 'SSCard']
    markers = ['*', 'D', 'v', '^', 's', 'o']
    # colors = [plt.cm.Accent(x) for x in [0, 1, 4, 6, 7]]
    # colors = [plt.cm.Dark2(x) for x in [0, 1, 4, 5, 6]]
    dnames = ["DBLP_AN", "IMDB_AN", "IMDB_MT", "TPCH_PN", "WIKI"]
    D = 5

    colors = ['#8c564b', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # default colors
    darken_cols = [blend_with_gray(x, 0.2) for x in colors]
    
    fig, axes = plt.subplots(nrows=1, ncols=D, figsize=(25, 5), tight_layout=False)
    fig.suptitle(' ', fontsize=18)
    
    # ylims = [(0, 5), (1.4, 4), (1.4, 3.1), (-70, 2001), (1.4, 10)]
    # avg_yticks = [np.linspace(x[0]+0.1, x[-1]-0.1, 5) for x in avg_ylims]
    # avg_yticks[3] = np.linspace(avg_ylims[3][0]+70, avg_ylims[3][-1]-1, 5)
    yticks = [(0, 120), (0, 40), (0, 1800), (0, 30), (0, 15000)]
    yticks = [np.linspace(x[0], x[-1], 6) for x in yticks]
    
    for fi in range(0, D):
        dname = dnames[fi]
        X = []
        Y = []
        # if dname != "WIKI":
        x_ticks = [x for x in range(1, 9)]
        # else:
            # x_ticks = [2 * x for x in range(1, 7)]
        # if dname == "DBLP":
        
        for mname in methods:
            strings, sels, counts, latencies = load_estimation_results(mname, dname)
            print(mname)
            x, y = cal_q_error_per_length(strings, sels, counts)
            X.append(x)
            Y.append(y)
        
        axes[fi].set_title(dname.replace('_', '-'), fontsize=20)
        axes[fi].set_xlabel('Pattern string length',size=18)
        if fi == 0:
            axes[fi].set_ylabel('Q-error',size=18)
        axes[fi].set_xticks(ticks=x_ticks)
        # axes[fi].set_ylim(avg_ylims[fi]) 
        axes[fi].set_yticks(yticks[fi])
        axes[fi].tick_params(axis='x', labelsize=18)
        axes[fi].tick_params(axis='y', labelsize=18)
        
        axes[fi].grid(True, linestyle='--')
        
        
        axes[fi].set_yscale('log')

        for i in range(6):
            if fi == 0:
                axes[fi].plot(X[i], Y[i], label=methods[i], linewidth=2.5, marker=markers[i], markersize=8, color=darken_cols[i])
            else:
                axes[fi].plot(X[i], Y[i], linewidth=2.5, marker=markers[i], markersize=8, color=darken_cols[i])
    # exit()
        
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=6, fontsize=18, frameon=False)
    fig.tight_layout()
    
    plt.savefig(filepath + filename + '.png', dpi=100)
    
    plt.close()


def calc_q_error_per_card(strings, sels, counts, ranges):
    cards = {}
    for i, tp in enumerate(ranges):
        cards[i] = []
    min_n = 1000000000
    max_n = 0
    for i, (s, n, c) in enumerate(zip(strings, sels, counts)):
        c = max(1, c)
        # print(n)
        min_n = min(min_n, n)
        max_n = max(max_n, n)
        q_error = max(n / c, c / n)
        for j, (L, R) in enumerate(ranges):
            if n >= L and n <= R:
                cards[j].append(q_error)
    print(min_n, max_n)
    # exit()
    x = []
    y = []
    for i, tp in enumerate(ranges):
        x.append(i)
        y.append(np.mean(cards[i]))
    return x, y
  
    
def error_per_card_painter_subplots():
    filepath = "../figures/"
    filename = 'Q_error_per_card_subplots_add_MO'
    methods = ['MO', 'LBS', 'Astrid', 'DREAM', 'LPLM', 'SSCard']
    dnames = ["DBLP_AN", "IMDB_AN", "IMDB_MT", "TPCH_PN", "WIKI"]
    D = 5
    
    fig, axes = plt.subplots(nrows=1, ncols=D, figsize=(25, 5), tight_layout=False)
    fig.suptitle(' ', fontsize=18)
    
    # yticks_old = [(0, 25), (0, 40), (0, 30), (0, 15), (0, 80)]
    # yticks = [np.linspace(x[0], x[-1], 6) for x in yticks_old]
    # yticks[-1] = np.linspace(yticks_old[-1][0], yticks_old[-1][-1], 5)
    
    colors = ['#8c564b', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # default colors
    darken_cols = [blend_with_gray(x, 0.2) for x in colors]
    
    # for fi in [0, 3]:
    for fi in range(0, D):
        # fi = 3
        # if fi != 3: break
        dname = dnames[fi]
        X = [0, 1, 2, 3]
        Y = []
        for mname in methods:
            strings, sels, counts, latencies = load_estimation_results(mname, dname)
            print(mname)
            if dname == "TPCH_PN":
                ranges = [(10000, 20000), (20000, 50000), (50000, 100000), (100000, 500000)]
            else:
                ranges = [(0, 10), (11, 20), (21, 50), (51, 10000000)]
            x, y = calc_q_error_per_card(strings, sels, counts, ranges)
            print(x)
            print(y)
            # X.append(x)
            Y.append(y)

        axes[fi].set_title(dname.replace('_', '-'), fontsize=20)
        bar_width = 0.2
        x_ticks = [(7 * x + 5 / 2 - 0.5) * bar_width for x in X]
        # x_ticks = [i for i in range(len(ranges))]
        if dname == "TPCH_PN":
            labels = ["[1,2]", "[2,5]", "[5,10]", "[10,max]"]
        else:
            labels = ["[0,10]", "[11,20]", "[21,50]", "[51,max]"]
        axes[fi].set_xticks(ticks=x_ticks)
        axes[fi].set_xticklabels(labels, rotation=45)
        # axes[fi].set_yticks(yticks[fi])
        
        axes[fi].grid(True, linestyle='--')
        axes[fi].tick_params(axis='x', labelsize=12)
        axes[fi].tick_params(axis='y', labelsize=18)
        
        if dname == "TPCH_PN":
            axes[fi].set_xlabel(r'Actual cardinality($10^4$)',size=18)
        else:
            axes[fi].set_xlabel('Actual cardinality',size=18)
        if fi == 0:
            axes[fi].set_ylabel('Q-error',size=18)

        for i in range(6):
            if fi == 0:
                axes[fi].bar([(7 * x + i) * bar_width for x in X], Y[i], label=methods[i], width=bar_width, color=darken_cols[i])
            else:
                bars = axes[fi].bar([(7 * x + i) * bar_width for x in X], Y[i], width=bar_width, color=darken_cols[i])    
                    
        axes[fi].set_yscale('log')
                    
        
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=6, fontsize=18, frameon=False)
    fig.tight_layout()
    
    plt.savefig(filepath + filename + '.png', dpi=500)  


def suffix_tree_h_painter_subplots():
    filepath = "../figures/"
    dnames = ["DBLP-AN", "IMDB-AN", "IMDB-MT", "TPCH-PN", "WIKI"]
    x = [1, 2, 3, 4, 5]
    y_q_error = []
    y_space = []
    # DBLP_AN
    y_q_error.append([3.30262, 3.22093, 2.923418, 2.86047, 2.858871])
    y_space.append([1.24521, 1.75866, 4.392942, 5.78496, 5.94955])
    
    # IMDB_AN
    y_q_error.append([2.2225, 2.1805, 2.04898, 2.00661, 2.004791])
    y_space.append([1.03878, 1.57326, 4.29022, 6.21461, 6.456454])

    # IMDB_MT
    y_q_error.append([2.31765, 2.277883, 2.138057, 2.11432, 2.11407])
    y_space.append([1.13858, 1.80821, 4.751679, 6.47421, 6.63787])

    # TPCH_PN
    y_q_error.append([1.0102, 1.00116, 1.00016170, 1.000093231, 1.000000489])
    y_space.append([0.09509, 0.14639, 0.369044, 0.7163305, 1.012964])
    
    # WIKI
    y_q_error.append([2.883783, 2.820183, 2.61244, 2.361263, 2.310549])
    y_space.append([10.32533, 13.01476, 22.99634, 43.94808, 61.9411])
    
    q_error_ylims = [(2.5, 3.5), (1.8, 2.5), (1.8, 2.5), (0.98, 1.02), (2.0, 3.0)]
    space_ylims = [(0.0, 8.0), (0.0, 8.0), (0.0, 8.0), (0.0, 1.2), (5.0, 80.0)]

    q_error_yticks = [np.linspace(x[0], x[-1], 5) for x in q_error_ylims]
    space_yticks = [np.linspace(x[0], x[-1], 5) for x in space_ylims]
    
    # darken_cols = [darken_color_hsv('#1f77b4', 0.8), darken_color_hsv('#ff7f0e', 0.8)]
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    darken_cols = [blend_with_gray('#2ca02c', 0.2), blend_with_gray('#ffb517', 0.7)]
    
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 3.7), tight_layout=False)
    # fig.suptitle(' ', fontsize=18)

    for fi in range(0, 5):
        axes[fi].plot(x, y_q_error[fi], marker='s', markersize=8, linestyle='-', linewidth=2.5, color=darken_cols[0])
        
        axes[fi].set_xlabel('Tree height', fontsize=18)
        axes[fi].set_title(dnames[fi], fontsize=20)
        axes[fi].set_ylim(q_error_ylims[fi]) 
        axes[fi].set_yticks(q_error_yticks[fi])
        axes[fi].set_xticks(x)
        axes[fi].tick_params(axis='x', labelsize=18)
        axes[fi].tick_params(axis='y', labelcolor=darken_cols[0], labelsize=18)
        if fi == 3:
            axes[fi].yaxis.set_major_formatter(FuncFormatter(format_func2))
        else:
            axes[fi].yaxis.set_major_formatter(FuncFormatter(format_func))
        
        axes[fi].grid(True, linestyle='--')

        ax2 = axes[fi].twinx()

        # else:
        ax2.plot(x, y_space[fi], marker='D', markersize=8, linestyle='-', linewidth=2.5, color=darken_cols[1])
        ax2.tick_params(axis='y', labelcolor=darken_cols[1], labelsize=12)
        if fi == 4:
            ax2.set_ylabel('Space (MB)', fontsize=18)
        ax2.set_ylim(space_ylims[fi])
        ax2.set_yticks(space_yticks[fi])
        ax2.tick_params(axis='x', labelsize=18)
        ax2.tick_params(axis='y', labelsize=18)
        ax2.yaxis.set_major_formatter(FuncFormatter(format_func))
        # ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
    
    axes[0].set_ylabel('Q-error(avg.)',fontsize=18)
    
    fig.tight_layout()
    
    filename = 'suffix_tree_h_subplots' 
    plt.savefig(filepath + filename + '.png', dpi=500)


def load_inc_results_new(filename):
    
    m1_q_error = []
    m2_q_error = []
    m1_q_time = []
    m2_q_time = []
    rebuild_points = []
    m1_rebuild_t = []
    m2_rebuild_t = []
    m1_load_t = []
    m2_load_t = []
    
    with open(filename, 'r') as file:
        for line in file:
            line_list = line.strip().split(':')
            
            if len(line_list) == 5:
                m1_q_error.append(float(line_list[1].strip()))
                m2_q_error.append(float(line_list[2].strip()))
                m1_q_time.append(float(line_list[3].strip()))
                m2_q_time.append(float(line_list[4].strip()))
            elif "Rebuild====" in line:
                rebuild_points.append(len(m1_q_error))
            elif "Method 1" in line:
                m1_rebuild_t.append(float(line_list[1].strip()))
            elif "Method 2" in line:
                m2_rebuild_t.append(float(line_list[1].strip()))  
            elif len(line_list) == 3:
                m1_load_t.append(float(line_list[1].strip()))
                m2_load_t.append(float(line_list[2].strip()))
                
    
    return m1_q_error, m2_q_error, m1_q_time, m2_q_time, rebuild_points, m1_rebuild_t, m2_rebuild_t, m1_load_t, m2_load_t


def incremental_painter_q_error(dname):
    filepath = "../figures/"
    filename = 'inc_' + dname + "_q_error"
    if dname == "DBLP_AN":
        x_ticks = []
        # x_ticks = [10000, 20000, 30000, 40000, 50000]
        # x_labels = ['1', '2', '3', '4', '5']
        m1_q_error, m2_q_error, m1_q_time, m2_q_time, rebuild_points, _, _, _, _ = load_inc_results_new(f"../inc_exp/{dname}/SSCard_cache250000.0_5_with_pattern_trie_30000/output.log")
        
        X = [i+1 for i in range(len(m1_q_error))]
    
    print(rebuild_points)
    
    plt.figure(figsize=(5, 4))
    
    # plt.xticks(ticks=X, labels=[i * 3 for i in X], fontsize=15)
    plt.xticks(ticks=X[::2], labels=[i * 3 for i in X[::2]], fontsize=15)
    plt.yticks(fontsize=15)
    
    plt.ylabel("Q-error", fontsize=18)
    plt.xlabel(r'# Data strings($ \times 10^4$)', fontsize=18)
    plt.plot(X, m1_q_error, label="Single-SSCard update strategy", marker='o')
    plt.plot(X, m2_q_error, label = "Multiple-SSCard update strategy", marker='s')
    
    for x in rebuild_points:
        plt.axvline(x=x, color='gray', linestyle='--') 
    
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(filepath + filename + '.png', dpi=500)


def incremental_painter_rebuild_time(dname):
    filepath = "../figures/"
    filename = 'inc_' + dname + "_rebulid_time"
    if dname == "DBLP_AN":
        _, _, _, _, _, m1_rebuild_t, m2_rebuild_t, _, _ = load_inc_results_new(f"../inc_exp/{dname}/SSCard_cache250000.0_5_with_pattern_trie_30000/output.log")
        
        X = [i for i in range(len(m1_rebuild_t))]
        X = np.array(X)  # 关键一步，先转成 NumPy 数组
        x_ticks = [9, 18, 30, 39]
        
    
    # print(rebuild_points)
    
    plt.figure(figsize=(5, 4))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # default colors
    darken_cols = [blend_with_gray(x, 0.2) for x in colors]
    
    plt.xticks(ticks=X, labels=x_ticks, fontsize=15)
    plt.yticks(fontsize=15)
    
    plt.grid(True, axis='y', linestyle='--')
    
    plt.ylabel("Update time (s)", fontsize=18)
    plt.xlabel(r'# Data strings($ \times 10^4$)', fontsize=18)
    
    
    width = 0.35
    plt.bar(X - width/2, m1_rebuild_t, width, label="Single-SSCard update strategy", color=darken_cols[0])
    bar2 = plt.bar(X + width/2, m2_rebuild_t, width, label = "Multiple-SSCard update strategy",  color=darken_cols[1])
    
    for rect in bar2:
        height = rect.get_height()
        plt.text(
            rect.get_x() + rect.get_width() / 2, 
            height,                            
            f'{height:.1f}',                    
            ha='center', va='bottom', fontsize=10 
        )
    
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(filepath + filename + '.png', dpi=500)
    
    
def incremental_painter_query_time(dname):
    filepath = "../figures/"
    filename = 'inc_' + dname + "_query_time"
    if dname == "DBLP_AN":
        x_ticks = []
        # x_ticks = [10000, 20000, 30000, 40000, 50000]
        # x_labels = ['1', '2', '3', '4', '5']
        m1_q_error, m2_q_error, m1_q_time, m2_q_time, rebuild_points, _, _, _, _ = load_inc_results_new(f"../inc_exp/{dname}/SSCard_cache250000.0_5_with_pattern_trie_30000/output.log")
        
        X = [i+1 for i in range(len(m1_q_error))]
    
    print(rebuild_points)
    
    plt.figure(figsize=(5, 4))
    
    # plt.xticks(ticks=X, labels=[i * 3 for i in X], fontsize=15)
    plt.xticks(ticks=X[::2], labels=[i * 3 for i in X[::2]], fontsize=15)
    plt.yticks(fontsize=15)
    
    m1_q_time = [x * 1000 for x in m1_q_time]
    m2_q_time = [x * 1000 for x in m2_q_time]
    
    plt.ylabel("Query time (ms)", fontsize=18)
    plt.xlabel(r'# Data strings($ \times 10^4$)', fontsize=18)
    plt.plot(X, m1_q_time, label="Single-SSCard update strategy", marker='o')
    plt.plot(X, m2_q_time, label = "Multiple-SSCard update strategy", marker='s')
    
    for x in rebuild_points:
        plt.axvline(x=x, color='gray', linestyle='--') 
    
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(filepath + filename + '.png', dpi=500)


def incremental_painter_space(dname):
    filepath = "../figures/"
    filename = 'inc_' + dname + "_space"
    if dname == "DBLP_AN":
        
        m1_space = [1470960, 2702356, 3611635, 4387423]
        m2_space = [1470960, 1537018, 1494102, 1561727]
        
        las = 0
        for i, x in enumerate(m2_space):
            m2_space[i] += las
            las = m2_space[i]
        
        m1_space = [x / 1024 / 1024 for x in m1_space]
        m2_space = [x / 1024 / 1024 for x in m2_space]
        
        _, _, _, _, _, m1_rebuild_t, m2_rebuild_t, _, _ = load_inc_results_new(f"../inc_exp/{dname}/SSCard_cache250000.0_5_with_pattern_trie_30000/output.log")
        
        X = [i for i in range(len(m1_rebuild_t))]
        X = np.array(X) 
        x_ticks = [9, 18, 30, 39]
        
    
    # print(rebuild_points)
    
    plt.figure(figsize=(5, 4))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # default colors
    darken_cols = [blend_with_gray(x, 0.2) for x in colors]
    
    plt.xticks(ticks=X, labels=x_ticks, fontsize=15)
    plt.yticks(fontsize=15)
    
    plt.grid(True, axis='y', linestyle='--')
    
    plt.ylabel("Space (MB)", fontsize=18)
    plt.xlabel(r'# Data strings($ \times 10^4$)', fontsize=18)
    
    
    width = 0.35
    bar1 = plt.bar(X - width/2, m1_space, width, label="Single-SSCard update strategy", color=darken_cols[0])
    bar2 = plt.bar(X + width/2, m2_space, width, label = "Multiple-SSCard update strategy",  color=darken_cols[1])
    
    for rect in bar2:
        height = rect.get_height()
        plt.text(
            rect.get_x() + rect.get_width() / 2,  
            height,                            
            f'{height:.2f}',           
            ha='center', va='bottom', fontsize=10
        )
    for rect in bar1:
        height = rect.get_height()
        plt.text(
            rect.get_x() + rect.get_width() / 2,  
            height,                            
            f'{height:.2f}',           
            ha='center', va='bottom', fontsize=10
        )
    
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(filepath + filename + '.png', dpi=500)


def main():

    spline_painter_new_with_max_subplots()

    error_per_length_painter_subplots()

    error_per_card_painter_subplots()
    
    suffix_tree_h_painter_subplots()
    
    incremental_painter_q_error("DBLP_AN")
    incremental_painter_rebuild_time("DBLP_AN")
    incremental_painter_query_time("DBLP_AN")
    incremental_painter_space("DBLP_AN")



if __name__ == '__main__':
    main()