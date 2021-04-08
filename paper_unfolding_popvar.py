# Load data
import tools as tl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
# from scipy.stats import rankdata
# from mpl_toolkits.mplot3d import Axes3D
import os
import seaborn as sns
import benchmark_func as bf
sns.set(context="paper", font_scale=1, palette="husl", style="ticks",
        rc={'text.usetex': True, 'font.family': 'serif', 'font.size': 12,
            "xtick.major.top": False, "ytick.major.right": False})

is_saving = False
saving_format = 'png'

chosen_categories = ['Differentiable', 'Unimodal']  # 'Separable',
case_label = 'DU'

#  ----------------------------------
# Read operators and find their alias
collection_file = 'default.txt'
with open('./collections/' + collection_file, 'r') as operators_file:
    encoded_heuristic_space = [eval(line.rstrip('\n')) for line in operators_file]

# Search operator aliases
perturbator_alias = {
    'random_search': 'RS',
    'central_force_dynamic': 'CF',
    'differential_mutation': 'DM',
    'firefly_dynamic': 'FD',
    'genetic_crossover': 'GC',
    'genetic_mutation': 'GM',
    'gravitational_search': 'GS',
    'random_flight': 'RF',
    'local_random_walk': 'RW',
    'random_sample': 'RX',
    'spiral_dynamic': 'SD',
    'swarm_dynamic': 'PS'}

selector_alias = {'greedy': 'g', 'all': 'd', 'metropolis': 'm', 'probabilistic': 'p'}

operator_families = {y: i for i, y in enumerate(sorted([x for x in perturbator_alias.values()]))}

# Pre-build the alias list
heuristic_space = [perturbator_alias[x[0]] + selector_alias[x[2]] for x in encoded_heuristic_space]

# Find repeated elements
for heuristic in heuristic_space:
    concurrences = tl.listfind(heuristic_space, heuristic)
    if len(concurrences) > 1:
        for count, idx in enumerate(concurrences):
            heuristic_space[idx] += f'{count + 1}'


# Read basic metaheuristics
with open('collections/' + 'basicmetaheuristics.txt', 'r') as operators_file:
    basic_mhs_collection = [eval(line.rstrip('\n')) for line in operators_file]

# Read basic metaheuristics cardinality
basic_mhs_cadinality = [1 if isinstance(x, tuple) else len(x) for x in basic_mhs_collection]


dimensions = [2, 10, 30, 50]
num_dime = len(dimensions)


def filter_by_dimensions(dataset):
    allowed_dim_inds = [index for d in dimensions for index in tl.listfind(dataset['dimensions'], d)]
    return {key: [val[x] for x in allowed_dim_inds] for key, val in dataset.items()}


# Load data from basic metaheuristics
basic_mhs_data = filter_by_dimensions(tl.read_json('data_files/basic-metaheuristics-data_v2.json'))
basic_metaheuristics = basic_mhs_data['results'][0]['operator_id']

long_dimensions = basic_mhs_data['dimensions']
long_problems = basic_mhs_data['problem']


# Call the problem categories
problem_features = bf.list_functions(fts=chosen_categories)
categories = sorted(set([problem_features[x]['Code'] for x in basic_mhs_data['problem']]), reverse=True)

# --------------------------------
# Special adjustments for the plots
plt.rc('text', usetex=True)
plt.rc('font',  size=18)  # family='serif',

# Colour settings
cmap = plt.get_cmap('tab20')
colour_cat = [cmap(i)[:-1] for i in np.linspace(0, 1, len(categories))]
colour_dim = [cmap(i)[:-1] for i in np.linspace(0, 1, num_dime)]

# Saving images flag
folder_name = 'data_files/results_unfolding/'
if is_saving:
    # Read (of create if so) a folder for storing images
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

# Read the data file and assign the variables
data_filenames = {30: "unfolded_metaheuristics", 50: "unfolded_hhs_pop50", 100: "unfolded_hhs_pop100"}

data_tables = dict()

for population, filename in data_filenames.items():
    temporal_data = tl.read_json('data_files/' + filename + '.json')
    data_frame = filter_by_dimensions(temporal_data)

    # for basic metaheuristics
    current_performances = [x['performance'][-1] for x in data_frame['results']]
    basic_performances = [x['performance'] for x in np.copy(basic_mhs_data['results'])]

    # Compare the current metaheuristic against the basic metaheuristics
    performance_comparison = [np.copy(x - np.array(y)) for x, y in zip(current_performances, basic_performances)]

    # Success rate with respect to basic metaheuristics
    success_rates = [np.sum(x < 0.0) / len(x) for x in performance_comparison]

    # Create a data frame
    if population == 30:
        hFitness = [[y[-1] for y in x['hist_fitness'][-1]] for x in data_frame['results']]
        pValue = [stats.normaltest([y[-1] for y in x['hist_fitness'][-1]])[1] for x in data_frame['results']]
    else:
        hFitness = [[y[-1] for y in x['hist_fitness']] for x in data_frame['results']]
        pValue = [stats.normaltest([y[-1] for y in x['hist_fitness']])[1] for x in data_frame['results']]

    data_tables[population] = pd.DataFrame({
        'Dim': [str(x) for x in data_frame['dimensions']],
        'Problem': data_frame['problem'],
        'Cat': [problem_features[x]['Code'] for x in data_frame['problem']],
        'Performance': [x['performance'][-1] for x in data_frame['results']],
        'Steps': [x['step'][-1] for x in data_frame['results']],
        'Cardinality': [len(x['encoded_solution'][-1]) for x in data_frame['results']],
        'hFitness': hFitness,
        'pValue': pValue,
        'operatorFamily': [[operator_families[heuristic_space[y][:2]] for y in x['encoded_solution'][-1]]
                           for x in data_frame['results']],
        'successRate': success_rates
    })

# Melt data in one table
full_table = pd.concat(data_tables, axis=0, names=['Pop', 'RID']).reset_index(level=0)
full_table['Dim'] = full_table['Dim'].apply(lambda x: int(x))

full_table['Rank'] = full_table.groupby(by=['Dim', 'Problem'])['Performance'].rank(method='dense')
full_table['RankSR'] = full_table.groupby(by=['Dim', 'Problem'])['successRate'].rank(method='dense', ascending=False)
full_table['DimPop'] = full_table[['Dim', 'Pop']].agg(tuple, axis=1)


def app_time_complexity(row):
    fam_list = row['operatorFamily']
    dim = row['Dim']
    pop = row['Pop']
    tc_by_fam = {
        0: 2 * pop,                     # CF
        1: pop ** 2,                    # DM
        2: 2 * pop,                     # FD
        3: pop ** 2,                    # GC
        4: pop,                         # GM
        5: 2 * pop,                     # GS
        6: 2 * pop,                     # PS
        7: 1,                           # RF
        8: 1,                           # RS
        9: pop,                         # RW
        10: 1,                          # RX
        11: pop * (dim ** 2.3737)       # SD
    }

    return np.sum(np.array([tc_by_fam[x] for x in fam_list]))


full_table['tcMH'] = np.log10(full_table['Dim'] * full_table['Pop'] *
                              full_table.apply(app_time_complexity, axis=1))

full_table['tcHH'] = full_table['tcMH'] + np.log10((full_table['Steps'] + 1) * 50)

theo_limit = pd.DataFrame([(d, p, np.log10(50 * 200 * 100 * d * (p ** 3))) for p in [30, 50, 100] for d in dimensions],
                          columns=['Dim', 'Pop', 'Theoretical'])

# %% FIRST PLOT // Distributions Rank - [done]
p1 = sns.displot(data=full_table, hue='Pop', col='Dim', row='Cat', x='Rank', kind='hist', palette='tab10', fill=True,
                 row_order=categories, multiple="fill", height=1.8, aspect=0.8, legend=True, stat="density", discrete=True)
p1.set_xlabels(r'Rank by Performance')

if is_saving:
    p1.savefig(folder_name + 'Rank_vs_Pop_and_Dim_Cat-FillDistr.' + saving_format,
               format=saving_format, dpi=333, transparent=True)

plt.show()

# %% SECOND PLOT // Violin plot of rankings by perf and sr


def change_quartiles(ax):
    for l in ax.lines:
        l.set_linestyle('--')
        l.set_linewidth(0.6)
        l.set_color('red')
        l.set_alpha(0.8)
    for l in ax.lines[1::3]:
        l.set_linestyle('-')
        l.set_linewidth(0.6)
        l.set_color('black')
        l.set_alpha(0.8)

plt.figure(figsize=[5, 2.5])
p2 = sns.violinplot(data=full_table, hue='Pop', x='Dim', y='Rank', palette="pastel", #cut=0,
                    linestyle=':', scale="area", inner="quartile")
change_quartiles(p2)

plt.ylabel(r'Rank by Performance'), plt.xlabel(r'Dimensions')

if is_saving:
    plt.savefig(folder_name + 'RankPerf_vs_Dim-Violin.' + saving_format,
                format=saving_format, dpi=333, transparent=True)

plt.show()

plt.figure(figsize=[5, 2.5])
p3 = sns.violinplot(data=full_table, hue='Pop', x='Dim', y='RankSR', palette="pastel", #cut=0,
                    linestyle=':', scale="area", inner="quartile")
# sns.displot(data=full_table, hue='Pop', col='Dim', x='RankSR', kind='kde')
change_quartiles(p3)

plt.ylabel(r'Rank by Success Rate'), plt.xlabel(r'Dimensions')

if is_saving:
    plt.savefig(folder_name + 'RankSR_vs_Dim-Violin.' + saving_format,
                format=saving_format, dpi=333, transparent=True)

plt.show()

# %% THIRD PLOT // KDE of Card/Steps vs pop and dim

p4 = sns.displot(data=full_table, hue='Pop', col='Dim', x='Cardinality', kind='kde', palette='tab10', height=2.5,
                 aspect=0.8, fill=True)

if is_saving:
    p4.savefig(folder_name + 'Card_vs_Dim_Pop-KDE.' + saving_format, format=saving_format, dpi=333, transparent=True)

plt.show()

p5 = sns.displot(data=full_table, hue='Pop', col='Dim', x='Steps', kind='kde', palette='tab10', height=2.5,
                 aspect=0.8, fill=True)

if is_saving:
    p5.savefig(folder_name + 'Steps_vs_Dim_Pop-KDE.' + saving_format, format=saving_format, dpi=333, transparent=True)

plt.show()

# %% FOURTH PLOT // General rank vs dim and pop

plt.figure(figsize=[5.5, 4])
sns.lineplot(data=full_table, x='Dim', y='Rank', palette='tab10', hue='Pop')

if is_saving:
    plt.savefig(folder_name + 'Rank_vs_Dim_Pop-Line.' + saving_format, format=saving_format, dpi=333, transparent=True)

plt.show()

# %% p-Value per Dim and Cat


def plot_pvalue_boxplot(data_table, pop=30):
    fig = plt.figure(figsize=(5, 2.5))
    sns.boxplot(data=data_table, x='Dim', y='pValue', hue='Cat', hue_order=categories)
    plt.hlines(0.05, -0.5, num_dime-0.5)
    plt.ylabel(r'$p$-Value')
    plt.title(r'Pop = {}'.format(pop))
    plt.show()

    if is_saving:
        fig.savefig(folder_name + 'pValue-pop{}-CatDim-BoxPlot.'.format(pop) + saving_format,
                    format=saving_format, dpi=333, transparent=True)


for pop, data_table in data_tables.items():
    plot_pvalue_boxplot(data_table, pop)


# %% operator family per Cat and Dim

families = list(operator_families.keys())
operator_table = full_table.groupby(by=['Dim', 'Pop'])['operatorFamily']\
    .apply(list)\
    .transform(lambda x: np.concatenate(x))\
    .transform(lambda x: [families[y] for y in x])  # .reset_index(level=0)


# %%

place1st_table = full_table.groupby(by=['Dim', 'Pop', 'Cat'])['Rank']\
    .apply(list)\
    .transform(lambda x: len([xx for xx in x if xx < 1.5]) / len(x))\
    .reset_index(name='Ratio')

sns.catplot(data=place1st_table, x='Dim', y='Ratio', hue='Pop', col='Cat', kind='bar')
plt.show()

# %%


def get_count(lst):
    output = dict()
    total = len(lst)
    for oper in list(perturbator_alias.values()):
        output[oper] = tl.listfind(lst, oper).__len__() / total
    return output


pre_hist_heuristic_space = get_count([perturbator_alias[x[0]] for x in encoded_heuristic_space])


def plot_hist_ope_fam():
    # x_labels = list(operator_families.keys())
    bin_centres = np.arange(len(operator_families))
    bin_edges = np.arange(len(operator_families) + 1) - 0.5
    for pop in list(data_filenames.keys()):
        weights = list()
        vals = list()
        fig, ax = plt.subplots(figsize=(5, 4))
        # plt.ion()

        for dim in dimensions:
            # flat_list.append([sublist for sublist in operator_table.loc[(dim, pop)]])
            raw_hist = get_count(operator_table.loc[(dim, pop)])
            hist_data = {key: raw_hist[key] / pre_hist_heuristic_space[key] for key in pre_hist_heuristic_space.keys()}

            val, weight = zip(*[(k, v) for k, v in hist_data.items()])
            weights.append(weight)
            vals.append(bin_centres)

        # plt.ioff()
        plt.hist(vals, weights=weights, bins=bin_edges, label=[str(x) for x in dimensions], density=True)
        plt.title(r'Pop = {}'.format(pop))
        plt.xlabel(r'Operator Family')
        plt.ylabel(r'Frequency')
        plt.xticks(ticks=bin_centres, labels=[r'{}'.format(x) for x in val])
        plt.legend([r'{}'.format(x) for x in dimensions], title=r'Dim')
        plt.ylim([0, 0.15])

        plt.show()

        if is_saving:
            fig.savefig(folder_name + 'OperFam-Pop{}-Hist.'.format(pop) + saving_format,
                        format=saving_format, dpi=333, transparent=True)


plot_hist_ope_fam()


# %%

print(
    full_table.query(
        "Problem == ['Sphere', 'Rastrigin', 'Step', 'ZeroSum']"
    )[
        ['Problem', 'Pop', 'Dim', 'Performance']]
        .sort_values(by=['Problem', 'Pop', 'Dim'])
        .to_latex(index=False, multirow=True, header=False)
)


# %%
fig, ax = plt.subplots(figsize=(5, 2.5))
p9 = sns.catplot(data=full_table, hue='Pop', x='Dim', y='tcHH', kind='box', palette='pastel')\
    .set(xlabel=r'Dimensions', ylabel=r'Order of operations') #, $\log_{10}(T_{HH})$')  ,col='Cat'

plt.tight_layout()

if is_saving:
    p9.savefig(folder_name + 'OperNum-HH-Box.' + saving_format, format=saving_format, dpi=333, transparent=True)

plt.show()
