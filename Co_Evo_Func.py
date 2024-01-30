'''
Created 06-2020

@author:Nittaym
'''
import numpy as np
import pandas as pd
import scipy.stats
import math

from scipy.spatial.distance import dice
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
import itertools
import matplotlib as plt
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
from itertools import combinations
from scipy.stats import sem
import numpy as np
import random
from itertools import combinations
from itertools import combinations_with_replacement

import matplotlib.patches as mpatches



def add_pseudocounts(count_table, sample_col='sample', add = 1):
    '''Add pseudocounts for all species that should be in a specific community.
-----------------------------------------------------------------------------------------
    Parameters:
    counts_table:
        A pd dataframe with rows ad datapoints and columns as species
    sample_col:
        the column in the data frame that indicates which species should be in the sample
        a str in the form sp1_sp2
-----------------------------------------------------------------------------------------
    Return:
        pd.Dataframe with added pseodocounts'''

    count_table[count_table[sample_col].split('_')] = count_table[count_table[sample_col].split('_')] + add
    return count_table


def Shannon(prob, neff=True, base =2):
    '''Calcuate the Shannon entropy for an array of probabilities.
-----------------------------------------------------------------------------------------
    Parameters:
    prob:
        np.array of probabilies where sum(prob) = 1
    neff:
        if True, return the the number equivalent
    base:
        the log base
-----------------------------------------------------------------------------------------
    Return:
        float, the Shannon entropy
        '''

    prob = prob[prob!=0] # remove zeroes
    sh = np.sum(prob*-1*(np.log(list(prob)/np.log(base))))
    if neff == True:
        sh = base**sh
    return sh

def euc_change(d, n = 1):
    '''Calculate the euclidean distance from each timepoint to the last measurment.
-----------------------------------------------------------------------------------------
    Parameters:
    d:
        dataframe with species as columns
-----------------------------------------------------------------------------------------
    Return:
        a pd.Dataframe, in every timepoint a float indicating the distance from last measurment.
        first timepoint would return NaN'''

    cs = pd.DataFrame(index = d.index, columns=['c'])
    for i in d.index:
        if (d.shift().loc[i, :].isnull().any() | d.loc[i, :].isnull().any()) == False:
            cs.loc[i, 'c'] = euclidean(d.loc[i, :].values, d.shift().loc[i, :].values)
    return cs/np.sqrt(n)

def dist_from_med(d, n=1):
    '''Compute the mean distacne from a medioid.
-----------------------------------------------------------------------------------------
    Parameters:
    d:
        dataframe or np.matrix with rows as samples and columns as observations
    '''

    dm = distance_matrix(d, d)
    minimum = np.argmin(dm.sum(axis = 0))
    return dm[minimum][np.arange(0, len(dm))!=minimum].mean()/np.sqrt(n)


def get_species_fraction(df, species, ident, transfer):
    '''Return the the fractions of a specific ident (plate+well) at a certain transfer.
-----------------------------------------------------------------------------------------
    Parameters:
    df:
        a dataframe with fractions, and ident column, and transfer coulmn
    species:
        the count coulmns
    ident:
        str, plate+well as p1A1
    transfer:
        float or int, the transfer to take fractions from
    '''

    frac = df[species][df['ident'] == ident][df['transfer'] == transfer].values
    return frac


def get_well_OD(df, ident, transfer):
    ''' Get the OD of a well (ident) at a speciefic transfer.
-----------------------------------------------------------------------------------------
    Parameters:
    df:
        OD data frame, with coulmn OD as values
    ident:
        str, plate+well as p1A1
    transfer:
        float or int, the transfer to take fractions from'''

    return df['OD'][df['ident'] == ident][df['transfer'] == transfer].values


def get_fractional_OD(fractions, ods, species, ident, transfer):
    '''Return the the fractional OD of a specific ident (plate+well) at certain transfer.
-----------------------------------------------------------------------------------------
    Parameters:
    fractions:
        a dataframe with fractions, and ident column, and transfer coulmn
    ods:
        OD data frame, with coulmn OD as values
    species:
        the count coulmns
    ident:
        str, plate+well as p1A1
    transfer:
        float or int, the transfer to take fractions from'''

    frac = get_species_fraction(fractions, species, ident, transfer)
    if len(frac) > 0:
        frac = frac[0]
    od = get_well_OD(ods, ident, transfer)
    return np.array([f * od for f in frac]).squeeze()


def randomize_incs(sample, incs, shape, bysp=False):
    ''' Shuffle the increasing values
-----------------------------------------------------------------------------------------
    Parameters:
    bysp:
        if bysp = False, shuffles the values across all species and communities
        if bysp = True, shuffles each species only with its own increasing values
    sample:
        str, the community, written as sp1_sp2
    incs:
        array/dict, values to choose randomly from. if bysp = True, should be dict with
        species names as keys.
    shape:
        number of species in the community
    '''

    if bysp == False:
        new = np.random.choice(incs, shape)
    else:
        new = [np.random.choice(incs[sp]) for sp in sample.split('_')]
    return new

def randomize_counts(count_table, sample_col='sample', dist='dirichlet', costume_dist=None):
    '''Return random frequencies of species ,{only for present speices}
-----------------------------------------------------------------------------------------
    Parameters:
    count_table:
        a table with species as columns, and a sample column that indicates which species should be there
        sample_col should be with str and as sp1_sp2
    dist:
        'costume'or 'dirichlet', if 'costume' choose randomly numbers from costume_dist. if dirichlet use draw from dirichlet
    costume_dist:
        array, costume distribution to draw numbers from
-----------------------------------------------------------------------------------------
    Return:
        a pd.Datafram, structured like count_table, but with random fractions.
    '''
    if dist == 'dirichlet':
        si = len(count_table[sample_col].split('_'))
        ran = scipy.stats.dirichlet(np.repeat(1, si)).rvs().squeeze()
        count_table[count_table[sample_col].split('_')] = ran
    elif dist == 'costume':
        si = len(count_table[sample_col].split('_'))
        ran = np.random.choice(costume_dist[costume_dist.apply(lambda x: len(x) == si)])
        count_table[count_table[sample_col].split('_')] = ran
    return count_table

# def randomize_counts(count_table, sample_col='sample'):
#     '''Return random (drawn from dirichlet) frequencies of species , but only for present speices
#     count_table: a table with species as columns, and a sample column that indicates which species should be there
#     sample_col should be with str and as sp1_sp2'''
#     si = len(count_table[sample_col].split('_')) #number of species in the commnity
#     ran = scipy.stats.dirichlet(repeat(1, si)).rvs().squeeze() #
#     count_table[count_table[sample_col].split('_')] = ran
#
#     return count_table


def calculate_nestedness(comm_matrix):
    ''''''
    comm_matrix = comm_matrix.reindex(comm_matrix.median(axis=1).sort_values(ascending=True).index,
                                      axis=0).reindex(comm_matrix.median(axis=1).sort_values(ascending=True).index,
                                                      axis=1)
    tr = triu(comm_matrix.values)
    return len(tr[tr < 0]) / (len(tr[tr < 0]) + len(tr[tr > 0]))


def most_frequent(List):
    ''''''
    List = list(List)
    return max(set(List), key=List.count)


def zscore(distance, mean_distance, sd_distance):
    ''''''
    return (distance - mean_distance) / sd_distance

def pmax(l):
    '''Return the frequency of the maximum element'''
    return max(l)/sum(l)


def summarize_repeatability(df):
    df['max'] = df['fold_increase'].apply(np.argmax)
    df = df.groupby(['sample', 'max'])['max'].count().groupby('sample').apply(pmax).reset_index()
    return df['max'].values



def get_competative_scores(sp, df, logit=False):
    '''Get the mean relative abundace over all communities a species was a part of
    sp: str, the query species
    df: Dataframe, with counts and a sample column as community
    logit: if true transform to logit'''

    in_community = lambda x: sp in x['sample'].split('_')
    other_species = lambda x: np.array(x.split('_'))[np.array(x.split('_')) != sp][0]
    if logit == True:
        result = df[sp][df.apply(in_community, axis=1)].apply(lambda x: x / (1 - x))
    else:
        result = df[sp][df.apply(in_community, axis=1)]
    result = pd.DataFrame(result)
    result['partner'] = df['sample'][df.apply(in_community, axis=1)].apply(other_species)
    result.index = df['ident'][df.apply(in_community, axis=1)].values

    return result


def get_improvment_values(sp, df, ti, tf, logit=True):
    '''Get the mean  increase in relative abundace over all communities a species was a part of
    sp: str, the query species
    df: Dataframe, with counts and a sample column as community
    ti: trasfer to start with
    tf: final t
    logit: if true transform to logit'''

    start = get_competative_scores(sp, df[df['transfer'] == ti], logit=False).groupby('partner').mean()
    end = get_competative_scores(sp, df[df['transfer'] == tf], logit=False).groupby('partner').mean()
    if logit == True:
        start = start / (1 - start)
        end = end / (1 - end)
    imp = pd.DataFrame(end[sp] / start[sp])
    #     imp['partner'] = imp['index'].apply(lambda x:df['sample'][df['ident']==x].values[0])
    #     imp['partner'] = imp['partner'].apply(lambda x:array(x.split('_'))[array(x.split('_'))!=sp][0])
    return imp

### Predictions




def return_pairwise(community):
    '''Return all the pairs composing a with the form sp1_sp2
    community:str, built as 'sp1_sp2_sp3'
    '''
    sps = community.split('_')
    comb = list(itertools.combinations(sps, 2))

    return [pair[0] + '_' + pair[1] for pair in comb]


def all_pairs_present(trio, pair_list):
    '''True if all pairwise pairs composing a trio are present in pair_list
    trio: str, built as 'sp1_sp2_sp3
    pair_list: list/np.array of strings with pairs as sp1_sp2'''

    return all([pr in pair_list for pr in return_pairwise(trio)])


def build_community_matrix(community, counts_table):
    '''Return a community matrix with columns coresponding
    to a species fraction when grown with a certein partner (row)'''

    sp_in = lambda x: sp in x['sample']
    others = lambda x: np.array(x['sample'].split('_'))[np.array(x['sample'].split('_')) != sp][0]
    mat = pd.DataFrame(columns=community.split('_'), index=community.split('_'))
    df = counts_table[counts_table['sample'].isin(return_pairwise(community))]
    if len(df) != 0:
        for sp in mat.columns:
            mat.loc[df[df.apply(sp_in, axis=1)].apply(others, axis=1).values, sp] = df[sp][
                df.apply(sp_in, axis=1)].values
    return mat


def pair_trio_prediction(trio, counts_table):
    '''Predict a trio composition from pairwise outcomes
    trio: str, in the form sp1_sp2_sp3
    counts_table: pd.Dataframe, with coloumns as observations and rows as sample
    return: a pd.Dataframe, with one row and columns as species
    return the predicted value which is calculated as the weighted geometric mean of the pairwise competitions
    '''

    sps = np.array(trio.split('_'))
    mat = build_community_matrix(trio, counts_table).transpose()
    outcome = pd.DataFrame(columns=sps)
    for sp in sps:
        f12 = mat.loc[sp, sps[sps != sp][0]]
        f13 = mat.loc[sp, sps[sps != sp][1]]
        w2 = np.sqrt(mat.loc[sps[sps != sp][0], sp] * mat.loc[sps[sps != sp][0], sps[sps != sp][1]])
        w3 = np.sqrt(mat.loc[sps[sps != sp][1], sp] * mat.loc[sps[sps != sp][1], sps[sps != sp][0]])
        outcome.loc[0, sp] = ((f12 ** w2) * (f13 ** w3)) ** (1 / (w2 + w3))
    outcome.loc[0, :] = outcome.loc[0, :] / outcome.sum(axis=1).values
    if all_pairs_present(trio, counts_table) == False:
        outcome == np.nan
    return outcome


def predict_by_grates(sample, species, g_rates, ks):
    sps = sample.split('_')
    rates = [ks[species == sp].values for sp in sps] * (
                1 - (np.log2(1500) / 48) / [g_rates[species == sp].values for sp in sps])
    ratio = rates / sum(rates)
    return ratio.squeeze()


def predict_max_inc(trio, pair_majorities, pair_means):

    '''Predict which species in a trio would increase by the biggest factor
    here we predict that if a species increased in the pairs it was in, it would increase in the trio
    if in each pair a different species increase, go to pred_no_h, in print no hierarchy
    -------------------------------------------------------------------------------------------------------
    pair_majorities:
        pd.Dataframe, with a sample column as pairs and a column indicating which and column most_frequent
        which indicates which species increased by the biggest factor (indicated as an index)
    pair_mean:
        pd.Dataframe, indicating the mean increase value of each oe of the species in each one of the pairs,
        used only if there is no hierarchy
    -------------------------------------------------------------------------------------------------------
    return
        str, species which is predicted to increase by the biggest factor in the trio'''
    temp = pair_majorities[pair_majorities['sample'].isin(return_pairwise(trio))]
    outcomes = [temp.loc[i, 'sample'].split('_')[temp.loc[i, 'most_freq']] for i in temp.index]
    if len(outcomes) == 3:
        out = most_frequent(outcomes)
        if len(set(outcomes)) == 3:
            out = pred_no_h(trio, pair_means)
            print(trio, ':no hierarchy')

    elif (len(outcomes) == 2) & (len(set(outcomes)) == 1):
        out = most_frequent(outcomes)
    else:
        out = np.nan

    return out

def pred_no_h(trio, pair_means):
    '''Predict which species in a trio would increase by the biggest factor
    here we predict that with the highest mean increase in pairs it would increase in the trio
    -------------------------------------------------------------------------------------------------------
    pair_mean:
        pd.Dataframe, indicating the mean increase value of each oe of the species in each one of the pairs,
        used only if there is no hierarchy
    -------------------------------------------------------------------------------------------------------
    return
        str, species which is predicted to increase by the biggest factor in the trio'''
    sp_in = lambda x:sp in x['sample'].split('_')
    where_in = lambda x: x['fold_increase'][np.where(np.array(x['sample'].split('_'))==sp)[0][0]]
    temp = pair_means[pair_means['sample'].isin(return_pairwise(trio))]
    avrs = []
    for sp in trio.split('_'):
        avrs.append(temp[temp.apply(sp_in, axis = 1)].apply(where_in, axis =1).mean())
    return trio.split('_')[np.where(np.array(avrs)==max(avrs))[0][0]]


# def get_mean_dist(lis):
#
#
#     dists = distance_matrix(lis,
#                             lis)
#     return np.mean(np.triu(dists)[np.triu(dists) != 0])
#
#
# def get_sem_dist(lis):
#     dists = distance_matrix(lis,
#                             lis)
#     return sem(np.triu(dists)[np.triu(dists) != 0])

########################################################################################################################
def define_treat(comm, sp):

    if sp == comm:
        return 'Monoevolved'
    else:
        return 'Coevolved'

def get_interaction(x, od):
    sps = x['sample'].split('_')
    for sp in sps:
        x[sp] = (x[sp]*x['total'] + 1)/(x['total']+2) # add pseudocounts
        x[sp] = np.log2((x[sp]*x['OD'])/od.loc[od['sample']==sp, 'OD'].mean())

    return x


def stack_int_table(int_table, last_col = 18):
    int_table = int_table[int_table.columns[:last_col]]
    int_table[int_table == 0] = np.nan
    stk_co = int_table.copy()
    int_table  = pd.DataFrame(int_table.groupby('sample').mean().stack()).rename({0: 'mean'}, axis = 1)
    int_table['sem'] = stk_co.groupby('sample').sem().stack()
    int_table
    return  int_table



def define_treat_lvl(lvl1, lvl2):
    '''Combine levels'''
    if lvl1 == 'anc':
        return 'anc'

    else:
        return lvl1 + '_' + lvl2

    
def start_gr(od):
    th = od[:7].median() + .002
    under_th = od < th
    return under_th.where(under_th).last_valid_index()


def end_gr(od):
    th = od.tail(500).median() - .01
    ab_th = od >= th
    return ab_th.where(ab_th).first_valid_index()


def euc_nan(x1, x2):
    try:
        return euclidean(x1, x2)
    except ValueError:
        return np.nan

def mean_euc(lis):

    dists = []
    for comb in list(combinations_with_replacement(lis, 2)):
        dists.append(euclidean(comb[0], comb[1]))
    if len(dists) == 1:
        return np.nan
    else:
        return np.mean(dists)


def pars_by_sps(int_table):
    int_prep = pd.DataFrame(columns = ['species', 'treatment', 'pair',
                                   'm-c', 'strain','strain_part','exp','anc','Pre_anc',
                                   'fraction', 'total', 'effect'])
    j  = 0
    for i in int_table[int_table['Pair']].index:
        int_prep.loc[j, 'species'] = int_table.loc[i, 'sample'].split('_')[0]
        int_prep.loc[j, 'treatment'] = int_table.loc[i, 'treatment']
        int_prep.loc[j, 'pair'] = int_table.loc[i, 'sample']
        int_prep.loc[j, 'm-c'] = int_table.loc[i, 'm-c']
        int_prep.loc[j, 'strain'] = int_table.loc[i, 'strain_a']
        int_prep.loc[j, 'strain_part'] = int_table.loc[i, 'strain_b']

        int_prep.loc[j, 'exp'] = int_table.loc[i, 'experiment']

        int_prep.loc[j, 'anc'] = int_table.loc[i, 'anc']
        int_prep.loc[j, 'Pre_anc'] = int_table.loc[i, 'Pre_anc']
        int_prep.loc[j, 'fraction'] = int_table.loc[i, 'strain_a_count']
        int_prep.loc[j, 'total'] = int_table.loc[i, 'total']
        int_prep.loc[j, 'effect'] = int_table.loc[i, 'eff_on_a']
        j=j+1

        int_prep.loc[j, 'species'] = int_table.loc[i, 'sample'].split('_')[1]
        int_prep.loc[j, 'treatment'] = int_table.loc[i, 'treatment']
        int_prep.loc[j, 'pair'] = int_table.loc[i, 'sample']
        int_prep.loc[j, 'm-c'] = int_table.loc[i, 'm-c']
        int_prep.loc[j, 'strain'] = int_table.loc[i, 'strain_b']
        int_prep.loc[j, 'strain_part'] = int_table.loc[i, 'strain_a']

        int_prep.loc[j, 'exp'] = int_table.loc[i, 'experiment']

        int_prep.loc[j, 'anc'] = int_table.loc[i, 'anc']
        int_prep.loc[j, 'Pre_anc'] = int_table.loc[i, 'Pre_anc']
        int_prep.loc[j, 'fraction'] = int_table.loc[i, 'strain_b_count']
        int_prep.loc[j, 'total'] = int_table.loc[i, 'total']
        int_prep.loc[j, 'effect'] = int_table.loc[i, 'eff_on_b']
        j=j+1

    int_prep['anc'] = int_prep['anc'].astype('bool')
    int_prep['Pre_anc'] = int_prep['Pre_anc'].astype('bool')
    int_prep['fraction'] = int_prep['fraction'].astype('float64')
    int_prep['total'] = int_prep['total'].astype('float64')
    int_prep['effect'] = int_prep['effect'].astype('float64')
    return int_prep


def fanc_int(exp, sp, treat, samp, inter, anc_table, ):
    try:
        return inter - anc_table.loc[exp, sp, treat, samp]['effect']
    except KeyError:
        return np.nan


def get_dups(gr_table):
    occur = gr_table.droplevel('exp').index.value_counts()
    return occur[occur>1]


def remove_dups(gr_table, dups):
    rem_dup = []
    for pair in dups.index:
        exp = gr_table.loc[:, pair[0], pair[1], pair[2]][('effect', 'count')].mean(axis=1).idxmin()
        rem_dup.append((exp, pair[0], pair[1], pair[2]))
    return rem_dup


def update_intersections(strain_on, ref, up_table, agg_lv = 'strain'):
    """Updates the intersection between two strains such that if an event (mutation)
    effects more then one gene, it could intersect with only one gene in the oterh strain
    Example:    strain 1 mutations [a, b, c]
                strain 2 mutations [[b, c], d]
                intersection [b] or [c] but not [b, c]"""

    for mutation in up_table.loc[
        (up_table[agg_lv] == strain_on) & (up_table['gene_product'].apply(lambda x: isinstance(x, list)))].index:
        inter = set(up_table.loc[mutation, 'gene_product']).intersection(ref['gene_product'])
        if len(list(inter)) > 1:
            #             print('yep')
            up_table.loc[mutation, 'gene_product'] = list(inter)[0]
            compl = ref.loc[ref['gene_product'] == list(inter)[0]].index[0]
            up_table.loc[compl, 'gene_product'] = list(inter)[0]
        else:
            up_table.loc[mutation, 'gene_product'] = up_table.loc[mutation, 'gene_product'][0]
    return up_table


def calculate_dice_sim(strain_a, strain_b, md, agg_lv = 'strain', cluster_mut = False):
    """Caluclate the dice similarity between two strains"""

    has_list = lambda x: isinstance(x, list) == False
    all_single = md.loc[md[agg_lv].isin([strain_a, strain_b]), 'gene_product'].apply(has_list)
    ### if all mutations in both strains affect only one gene
    if all(all_single):

        temp = md[md[agg_lv].isin([strain_a, strain_b])].groupby(agg_lv).agg(
            {'gene_product': 'value_counts'}).unstack().fillna(0)
        if cluster_mut:
            temp[temp > 1] = 1

        return (1 - dice(temp.loc[strain_a], temp.loc[strain_b]))

    ### if some mutation affect more then one gene
    else:
        fin_table = md.loc[
            md[agg_lv].isin([strain_a, strain_b]), [agg_lv, 'gene_product', 'mutation_id']].copy()
        table_a = fin_table[fin_table[agg_lv] == strain_a].explode('gene_product')
        table_b = fin_table[fin_table[agg_lv] == strain_b].explode('gene_product')
        fin_table = update_intersections(strain_a, table_b, fin_table, agg_lv)
        fin_table = update_intersections(strain_b, table_a, fin_table, agg_lv)
        temp = fin_table.groupby(agg_lv).agg({'gene_product': 'value_counts'}).unstack().fillna(0)
        if cluster_mut:
            temp[temp > 1] = 1

        return (1 - dice(temp.loc[strain_a], temp.loc[strain_b]))

def calculate_dice_dist(strain_a, strain_b, md):
    """Caluclate the dice similarity between two strains"""

    has_list = lambda x: isinstance(x, list) == False
    all_single = md.loc[md['strain'].isin([strain_a, strain_b]), 'gene_product'].apply(has_list)
    ### if all mutations in both strains affect only one gene
    if all(all_single):
        temp = md[md['strain'].isin([strain_a, strain_b])].groupby('strain').agg(
            {'gene_product': 'value_counts'}).unstack().fillna(0)
        return dice(temp.loc[strain_a], temp.loc[strain_b])

    ### if some mutation affect more then one gene
    else:

        fin_table = mut_table.loc[
            mut_table['strain'].isin([strain_a, strain_b]), ['strain', 'gene_product', 'mutation_id']].copy()
        table_a = fin_table[fin_table['strain'] == strain_a].explode('gene_product')
        table_b = fin_table[fin_table['strain'] == strain_b].explode('gene_product')
        fin_table = update_intersections(strain_a, table_b, fin_table)
        fin_table = update_intersections(strain_b, table_a, fin_table)
        temp = fin_table.groupby('strain').agg({'gene_product': 'value_counts'}).unstack().fillna(0)
        return dice(temp.loc[strain_a], temp.loc[strain_b])


def get_sum_dists(X, lvl,bw ,method):
    '''Get summary statistic for distances between strains
    X: pd.Dataframe, distance matrix, multilevelindex with index (and columns) levels indicating the different grouping levels
    lvl: the level in the multiindex that the calculations are preformed on
    bw: string  between or within. wether to get the summary stats for for distances of pairs that has the same lvl value(within), or viceversa
    method: method to use'''

    if bw == 'between':
        mask = pd.DataFrame(X.apply(lambda y: y.index.get_level_values(lvl) != y.name[lvl], axis=1).tolist(),
                        index=X.index, columns=X.index)
    elif bw == 'within':
        mask = pd.DataFrame(X.apply(lambda y: y.index.get_level_values(lvl) == y.name[lvl], axis=1).tolist(),
                        index=X.index, columns=X.index)
    else:
        print(bw, 'is not an option')
    calc = np.asarray(X[mask].values.flatten(), dtype = float)
    return method(calc[~np.isnan(calc)])

def same_sign(x, y):

    return math.copysign(1, x) == math.copysign(1, y)

def calculate_nestedness(mat):
    first = np.array(list(mat.values[triu_indices(mat.shape[0])]))
    first = np.array(list(mat.values[triu_indices(mat.shape[0])]))
    upper = first[~np.isnan(first)]
    return (upper>0).mean()


def get_gene_frac(x, frac_table):
    if x in list(frac_table['locus_tag']):
        return frac_table.loc[frac_table['locus_tag'] == x, 'fraction_of_genome'].values[0]
    else:
        return frac_table.loc[frac_table['gene'] == x, 'fraction_of_genome'].values[0]


def permutate_scores(dist_matrix, lvl_on, lvl_stat, permut=1000):
    mat_copy = dist_matrix.copy()
    shuffled_ind = list(mat_copy.index.get_level_values(lvl_on))
    scores = []

    for i in range(permut):
        random.shuffle(shuffled_ind)
        new_ind = pd.MultiIndex.from_tuples(tuple(zip(shuffled_ind, mat_copy.index.get_level_values(lvl_stat))),
                                            names=[lvl_on, lvl_stat])
        mat_copy = mat_copy.set_index(new_ind)
        mat_copy.columns = new_ind

        scores.append(get_sum_dists(mat_copy, 0, 'between', np.mean) - get_sum_dists(mat_copy, 0, 'within', np.mean))

    return np.array(scores)



def build_par_mat(ev_strains, anc_strain,
                  eps, inds, ret_both, reshape):
    ### Distance matrix of evolved strains
    ev_strains = ev_strains.dropna()
    if reshape:
        dist_mat = pd.DataFrame(distance_matrix(ev_strains.values.reshape(-1, 1),
                                                ev_strains.values.reshape(-1, 1)),
                                index=inds, columns=inds)
    else:
        dist_mat = pd.DataFrame(distance_matrix(ev_strains, ev_strains),
                                index=inds, columns=inds)

    ### Mask the diagonal and bellow to avoid duplicating data points and distance to self
    mask = np.tri(dist_mat.shape[0], dist_mat.shape[1], k=0, dtype=bool)
    dist_mat[mask] = np.nan

    ### Distance from ancestor
    if ret_both!='dist':
        if reshape:
            dist_anc = ev_strains.apply(lambda x: math.sqrt((x - anc_strain)**2))
        else:
            dist_anc = ev_strains.apply(lambda x: euclidean(x, anc_strain), axis=1)

        #     dist_anc   = dist_anc.apply(lambda x:x[(x.name[0], x.name[1])], axis =1)
        dist_anc = pd.DataFrame(np.add.outer(dist_anc.values,
                                             dist_anc.values),
                                index=inds, columns=inds)

        par_mat = ((dist_anc - dist_mat) + eps) / (dist_anc + eps)
        par_mat[mask] = np.nan

    if ret_both == True:
        return dist_mat, par_mat
    elif ret_both == 'par':
        return par_mat
    elif ret_both == 'dist':
        return dist_mat


def prop_error_sub(x_err, y_err, n):
    return ((x_err ** 2 + y_err ** 2) ** 0.5) / (n ** 0.5)


def construct_sum_stat(w_ind, dists, pars,  agg_lv=0, measures = ['Distance', 'Par'],
                       avrage = np.mean, error  = scipy.stats.sem):

    cols = pd.MultiIndex.from_tuples([('Distance', 'mean', 'between'),
                                      ('Distance', 'mean', 'within'),
                                      ('Distance', 'mean', 'score'),
                                      ('Distance', 'sem', 'between'),
                                      ('Distance', 'sem', 'within'),
                                      ('Distance', 'sem', 'score'),

                                      ('Distance', 'permutations', 'p-value'),
                                      ('Distance', 'permutations', 'n'),

                                      ('Par', 'mean', 'between'),
                                      ('Par', 'mean', 'within'),
                                      ('Par', 'mean', 'score'),
                                      ('Par', 'sem', 'between'),
                                      ('Par', 'sem', 'within'),
                                      ('Par', 'sem', 'score'),
                                      ('Par', 'permutations', 'p-value'),
                                      ('Par', 'permutations', 'n')])

    sum_table = pd.DataFrame(index=pd.MultiIndex.from_tuples([w_ind]),
                             columns=cols)
    if 'Distance' in measures:
    ### Distance and Standart error between treatments:
        sum_table[('Distance', 'mean', 'between')] = get_sum_dists(dists, agg_lv, 'between', avrage)
        sum_table[('Distance', 'sem', 'between')] = get_sum_dists(dists, agg_lv, 'between', error)

        ### Distance and Standart error within treatments:
        sum_table[('Distance', 'mean', 'within')] = get_sum_dists(dists, agg_lv, 'within', avrage)
        sum_table[('Distance', 'sem', 'within')] = get_sum_dists(dists, agg_lv, 'within', scipy.stats.sem)

        ### compute scores:
        sum_table[('Distance', 'mean', 'score')] = sum_table[('Distance', 'mean', 'between')] - sum_table[
            ('Distance', 'mean', 'within')]
        sum_table[('Distance', 'sem', 'score')] = prop_error_sub(get_sum_dists(dists, agg_lv, 'between', np.std),
                                                                 get_sum_dists(dists, agg_lv, 'within', np.std),
                                                                 get_sum_dists(dists, agg_lv, 'between', len))
    if 'Par' in measures:
    
        ### Par and Standart error between treatments:
        sum_table[('Par', 'mean', 'between')] = get_sum_dists(pars, agg_lv, 'between', avrage)
        sum_table[('Par', 'sem', 'between')] = get_sum_dists(pars, agg_lv, 'between', error)

        ### Par and Standart error within treatments:
        sum_table[('Par', 'mean', 'within')] = get_sum_dists(pars, agg_lv, 'within', avrage)
        sum_table[('Par', 'sem', 'within')] = get_sum_dists(pars, agg_lv, 'within', error)

        ### compute scores:
        sum_table[('Par', 'mean', 'score')] = sum_table[('Par', 'mean', 'within')] - sum_table[('Par', 'mean', 'between')]
        sum_table[('Par', 'sem', 'score')] = prop_error_sub(get_sum_dists(pars, agg_lv, 'within', np.std),
                                                            get_sum_dists(pars, agg_lv, 'between', np.std),
                                                            get_sum_dists(pars, agg_lv, 'between', len))
    return sum_table


def sum_stat(ev_strains, anc_strain, eps, inds, ret, agg_lv,reshape, avrage = np.mean, erorr  = scipy.stats.sem):

    if ret == 'par':
        par_mat = build_par_mat(ev_strains, anc_strain, eps, inds, ret, reshape)
        return get_sum_dists(par_mat, agg_lv, 'within', avrage) - get_sum_dists(par_mat, agg_lv,
                                                                                     'between', avrage)
    if ret == 'dist':
        dist_mat = build_par_mat(ev_strains, anc_strain, eps, inds, ret,  reshape)
        return get_sum_dists(dist_mat, agg_lv, 'between', avrage) - get_sum_dists(dist_mat, agg_lv, 'within', avrage)

    
def get_pairwise_gene_dist(mut_table, dist_mat, sp, treat, comm):
    temp_ind = mut_table[(mut_table['Species'] == sp) & (mut_table['Pre-Na'] == treat) & (mut_table['ev-comm'] == comm)]

    if len(temp_ind) > 1:

        bet = dist_mat.loc[sp, treat, sp][sp, treat, comm].values.flatten()
        wit = np.concatenate([dist_mat.loc[sp, treat, sp][sp, treat, sp].values.flatten(),
                           dist_mat.loc[sp, treat, comm][sp, treat, comm].values.flatten()])

        return np.nanmean(list(wit)) - np.nanmean(list(bet))
    else:
        return np.nan

def get_pairwise_gr_dist(gr_table,anc_table, sp, treat, comm):
    if (sp, treat) in gr_table.index:
        ev_strains = gr_table.loc[sp, treat, :, [sp, comm]][['K', 'eff_R']].droplevel('m-c')
        anc_strains = anc_table.loc[sp, treat]

        if len(ev_strains['K'])>1:
            d_mat, p_mat = build_par_mat(ev_strains,
                                             anc_strains,
                                             0,
                                             ev_strains.index, True, False)
            sc = construct_sum_stat((comm, sp),d_mat, p_mat, agg_lv = 2).loc[(comm, sp)][("Par", 'mean', 'score')]
            return sc
    else:
        return np.nan

    



def get_common(ind):
    if ind[1] == 'Naive':
        sec = '_Exp_14-'
    else:
        sec = '_Exp_26-'
    return ind[1] + sec


def turn_to_set(inds):
    est = set([])
    inds = sorted(inds)
    for ev_comm in set(item[0] for item in inds):
        est.add("".join([item[1] for item in inds if item[0] == ev_comm]))

    return frozenset(est)

def perm_scores(ev_strains, anc_strain, eps,save_as,
                ret='par', agg_lv=0, iters=100,
                reshape = False):

    # file = open(save_as, 'w')
    scores = []
    inds_lev2 = list(ev_strains.index.get_level_values(agg_lv + 1))
    if isinstance(ev_strains, pd.core.series.Series):
        cols = ['d']
    else:
        cols = ev_strains.columns

    # commo = get_common(ev_strains.index)
    ind_set = set([turn_to_set(ev_strains.index)])

    for i in range(iters):
        if i%1000 == 0:
            print(save_as, i)
        inds_perm = list(ev_strains.index.get_level_values(agg_lv))
        random.shuffle(inds_perm)
        new_index = pd.MultiIndex.from_arrays([inds_perm, inds_lev2], names = ev_strains.index.names)
        check_new = turn_to_set(new_index)
        # print(check_new)
        if check_new in ind_set:
            continue
        else:
            ind_set.add(check_new)
            if isinstance(ev_strains, pd.core.series.Series):
                shuf_data = pd.DataFrame(ev_strains.values,
                                         columns=cols,
                                         index=new_index)['d']
            else:
                shuf_data = pd.DataFrame(ev_strains.values,
                                         columns=cols,
                                         index=new_index)
            ss = sum_stat(ev_strains = shuf_data,
                          anc_strain = anc_strain,
                          eps = eps, inds = new_index,
                          ret = ret, agg_lv =agg_lv, reshape = reshape)

            scores.append(ss)
    #     if i%500 == 0:
    #         file.writelines(scores)
    # file.close()
    return np.array(scores)

# def agg_for_corrs(int_table, mut_table, dist_mat, growth_table)

def plot_par(par_table, measure, pv, species_dict,
             paired=False, def_lims=(-0.02, 1.02), fmt='o',
             ticks=[0, .5, 1], par=True,ax = None):
    if paired:
        par_ns = par_table[par_table[pv] > .05]
        par_s = par_table[par_table[pv] < .05]

        ax.errorbar(x=par_ns[(measure, 'mean', 'within')],
                 y=par_ns[(measure, 'mean', 'between')],
                 xerr=par_ns[(measure, 'sem', 'within')],
                 yerr=par_ns[(measure, 'sem', 'between')],
                 fmt=fmt, mfc='w',
                 color='k', mew=3,
                 ms=13, capsize=4)

        ax.errorbar(x=par_s[(measure, 'mean', 'within')],
                 y=par_s[(measure, 'mean', 'between')],
                 xerr=par_s[(measure, 'sem', 'within')],
                 yerr=par_s[(measure, 'sem', 'between')],
                 fmt=fmt, mfc='k',
                 color='k', mew=3,
                 ms=13, capsize=4)
    else:
        par_ns = par_table[par_table[pv] > .05]
        par_s = par_table[par_table[pv] < .05]

        for sp in par_table.index.get_level_values('species').unique():

            if sp in par_ns.index:
                ax.errorbar(x=par_ns.loc[sp][(measure, 'mean', 'within')],
                         y=par_ns.loc[sp][(measure, 'mean', 'between')],
                         xerr=par_ns.loc[sp][(measure, 'sem', 'within')],
                         yerr=par_ns.loc[sp][(measure, 'sem', 'between')],
                         fmt=fmt, mfc='w',
                         color=species_dict[sp], mew=3,
                         ms=13, label=sp
                         , capsize=4)
            if sp in par_s.index:
                ax.errorbar(x=par_s.loc[sp][(measure, 'mean', 'within')],
                         y=par_s.loc[sp][(measure, 'mean', 'between')],
                         xerr=par_s.loc[sp][(measure, 'sem', 'within')],
                         yerr=par_s.loc[sp][(measure, 'sem', 'between')],
                         fmt=fmt, mfc=species_dict[sp],
                         color=species_dict[sp], mew=3,
                         ms=13, label=sp
                         , capsize=4)
    if measure == 'Par':
        ax.fill_between(x=[def_lims[0], def_lims[1]], y2=[def_lims[0], def_lims[1]],
                     y1=[def_lims[1], def_lims[1]], color='grey', alpha=0.2)
    elif measure == 'Distance':
        ax.fill_between(x=[def_lims[0], def_lims[1]], y1=[def_lims[0], def_lims[0]],
                     y2=[def_lims[0], def_lims[1]], color='grey', alpha=0.2)
    ax.plot([def_lims[0], def_lims[1]], [def_lims[0], def_lims[1]], '--', color='k', alpha=0.6)
    ax.set_xlim(def_lims[0], def_lims[1])
    ax.set_ylim(def_lims[0], def_lims[1])

    ax.set_xlabel('Within treatments')
    ax.set_ylabel('Between treatments')
    sns.despine(offset=10, ax = ax)



def pec_score_inset(dt, param, pv, species_dict,
                    fig, fmt='o', paired=False, left=0.25, bottom=0.5, width=0.1, height=.3):
    ax2 = fig.add_axes([left, bottom, width, height])
    sns.despine(ax=ax2, top=False, right=False)

    if paired:

        par_ns = dt[dt[pv] > .05]
        par_s = dt[dt[pv] < .05]

        ax2.errorbar(x=np.random.normal(0, .075, len(par_ns)),
                     y=par_ns[(param, 'mean', 'score')],
                     mfc='w', fmt=fmt,
                     color='k', mew=1.5,
                     ms=7)
    else:

        par_ns = dt[dt[pv] > .05]
        par_s = dt[dt[pv] < .05]

        for sp in dt.index.get_level_values('species').unique():

            if sp in par_ns.index:

                ax2.errorbar(x=np.random.normal(0, .075, len(par_ns[par_ns.index.get_level_values('species') == sp])),
                             y=par_ns.loc[sp][(param, 'mean', 'score')],
                             mfc='w', fmt=fmt,
                             color=species_dict[sp], mew=1.5,
                             ms=7)

            if sp in par_s.index:
                ax2.errorbar(x=np.random.normal(0, .075, len(par_s[par_s.index.get_level_values('species') == sp])),
                             y=par_s.loc[sp][(param, 'mean', 'score')],
                             mfc=species_dict[sp], fmt=fmt,
                             color=species_dict[sp], mew=1.5,
                             ms=7)

    ax2.plot([-1, 1], [0, 0], '--', color='k', alpha=0.6)
    sns.boxplot(y=dt[(param, 'mean', 'score')],
                x=np.zeros(len(dt)), ax=ax2,
                color='w')

    ax2.fill_between(x=[-1, 1], y2=[0, 0],
                     y1=dt[(param, 'mean', 'score')].min() - 0.05, color='grey', alpha=0.2)
    # ax2.fill_between(x=[-1, 1], y2=[0, 0],
    #                  y1=dt[(param, 'mean', 'score')].max() + 0.05, color='w')
    # ax2.set_facecolor('w')

    ax2.set_xlim(-.6, .6)
    ax2.set_ylim(dt[(param, 'mean', 'score')].min() - 0.05,
                 dt[(param, 'mean', 'score')].max() + 0.05)
    ax2.set_ylabel('Specificity')
    ax2.set_title(round(dt[(param, 'mean', 'score')].median(), 2), fontdict={'size': 20})
    ax2.set_xticks([])
    ax2.set_facecolor('#FDFEFF')


def plot_sp(df_means, df_sems,df_anc, sp, treat,colors, ax, score, leg=True):

    for ev_comm in df_means.loc[sp, treat].index.get_level_values('ev_comm').unique():

        if ev_comm == sp:
            ax.errorbar(x=df_means.loc[sp, treat, :, ev_comm]['K'],
                        y=df_means.loc[sp, treat, :, ev_comm]['eff_R'],
                        xerr=df_sems.loc[sp, treat,:,  ev_comm]['K'],
                        yerr=df_sems.loc[sp, treat, :, ev_comm]['eff_R'],
                        fmt='o', color=colors[sp], mfc='None', ms=7, mew=2)

        else:

            partner = np.array(ev_comm.split('_'))[np.array(ev_comm.split('_')) != sp][0]
            # print(partner)
            ax.errorbar(x=df_means.loc[sp, treat, :, ev_comm]['K'],
                        y=df_means.loc[sp, treat, :, ev_comm]['eff_R'],
                        xerr=df_sems.loc[sp, treat, :, ev_comm]['K'],
                        yerr=df_sems.loc[sp, treat,:, ev_comm]['eff_R'],
                        fmt='o', color=colors[partner], mfc='None', ms=7, mew=2)

    ax.errorbar(df_anc.loc[(sp, treat)]['K'],
                df_anc.loc[(sp, treat)]['eff_R'], color='k', fmt='^', ms=7, mew=2,
                label='Ancestor')
    ax.set_ylabel('Growth rate (1/h)')
    ax.set_xlabel('Carrying capacity')
    ax.set_title(sp + ',' + str(score), fontdict = {'size':20})
    for spine in ax.spines.values():
        spine.set_edgecolor(colors[sp])
        spine.set_linewidth(3)
    sns.despine(right = False, top = False, ax = ax)
    ax.set_xticks([round(df_means.loc[sp, treat]['K'].min(), 2), round(df_means.loc[sp, treat]['K'].max(), 2)])
    ax.set_yticks([round(df_means.loc[sp, treat]['eff_R'].min(), 2), round(df_means.loc[sp, treat]['eff_R'].max(), 2)])
    ax.set_xticklabels([round(df_means.loc[sp, treat]['K'].min(), 2), round(df_means.loc[sp, treat]['K'].max(), 2)], fontdict = {'size':17})
    ax.set_yticklabels([round(df_means.loc[sp, treat]['eff_R'].min(), 2), round(df_means.loc[sp, treat]['eff_R'].max(), 2)], fontdict = {'size':17})
    if leg:
        ax.legend(frameon=False, loc=(1, .5), title='Evolved with')

def short_name(gene_name, species):
    fir = gene_name.split('_')[0]
    if fir in species:
        return gene_name
    else:
        return fir

def add_colored_title(ax, title, color):
    # Add a colored box using patches
    rect = mpatches.Rectangle((0, 1.02), 1, 0.15, transform=ax.transAxes,
                              color=color, clip_on=False, alpha = 0.5)
    ax.add_patch(rect)

    # Add the title
    ax.text(0.1, 1.08, title, color='k',
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)




def plot_param_bysp(gm_table, bc_table,
                    level, param, ax_minx,
                    ax_maxx, ax_miny, ax_maxy,
                    ax, species_dict, ofs=10):

    for sp in bc_table.loc[:, level, :].index.get_level_values(0).unique():
        ax.errorbar(x=gm_table.unstack().loc[sp, level][(param, 'mean', 'Monoevolved')]
                    , y=gm_table.unstack().loc[sp, level][(param, 'mean', 'Coevolved')],
                    xerr=gm_table.unstack().loc[sp, level][(param, 'sem', 'Monoevolved')],
                    fmt='s', color=species_dict[sp], mfc='w', ms=13, alpha=.7, capsize=8, mew=3)

        ax.errorbar(x=bc_table.loc[sp, level][(param, 'ref')]
                    , y=bc_table.loc[sp, level][(param, 'mean')],
                    yerr=bc_table.loc[sp, level][(param, 'sem')],
                    fmt='o', color=species_dict[sp], mfc='w', ms=10, capsize=5, mew=3)

    ax.plot([0, 1000], [0, 1000], color='k', alpha=.7, linestyle='--')
    ax.plot([1, 1], [0, 1000], color='k', alpha=.7, linestyle='--')
    ax.plot([0, 1000], [1, 1], color='k', alpha=.7, linestyle='--')

    ax.set_yscale('log', base=2)
    ax.set_xscale('log', base=2)
    ax.set_ylim(ax_miny, ax_maxy)
    ax.set_xlim(ax_minx, ax_maxx)
    sns.despine(offset=ofs, ax=ax)


def plot_trajectory(community, ax, cf, cm,
                    ides=None, ticksfont={},
                    labelfont={}, alpha='changing',
                    sps='com', xtick=[0, 200, 400], anc = False):
    d = cf[(cf['sample'] == community)&((cf['total'] > 10))]
    al = alpha
    if sps == 'com':
        sps = community.split('_')
        if anc == True:
            sps = [sp.split('-')[0] for sp in sps]
    for i, ide in enumerate(set(d['ident'][:ides])):

        data = d[(d['ident'] == ide)]
        for st in sps:
            if alpha == 'changing':
                al = (i + 1) / len(set(d['ident'][:ides])) - 0.01
            colr = cm[st]
            ax.errorbar(y = data[st], x= data["Generation"], yerr=data.apply(lambda x:np.sqrt(x[st]*(1-x[st])/x['total']), axis = 1).values,
                        ecolor= colr, color = colr, alpha = al)
            # data.plot(y=st, x='Generation', ax=ax, legend=False,
            #           colors=colr, alpha=al)
            data.plot.scatter(y=st, x='Generation', ax=ax, legend=False,
                              color=colr, alpha=al, s = 40)
    ax.set_ylim(0, 1.)
    ax.set_xticks(xtick);
    ax.set_xticklabels(xtick, fontdict=ticksfont)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels([0, 0.5, 1.], fontdict=ticksfont)
    ax.set_ylabel('Relative abundance', fontdict=labelfont)
    ax.set_xlabel('Generation', fontdict=labelfont)
    ax.set_xlim(0, 400)


def plot_trajectory2(community, ax, cf, cm,
                    ides=None, ticksfont={},
                    labelfont={}, alpha='changing',
                    sps='com', xtick=[0, 200, 400]):
    d = cf[cf['sample'] == community]
    al = alpha
    if sps == 'com':
        sps = community.split('_')
    for i, ide in enumerate(set(d['ident'][:ides])):
        data = d[d['ident'] == ide][cf['total'] > 10]
        for i, c in enumerate(['count1', 'count2']):
            if alpha == 'changing':
                al = (i + 1) / len(set(d['ident'][:ides])) - 0.01
            colr = cm[sps[i]]
            #             ax.errorbar(y = data[st], x= data["Generation"], yerr=data['std'],fmt='|',
            #                         ecolor= colr, alpha = al)
            data.plot(y=c, x='Generation', ax=ax, legend=False,
                      color=colr, alpha=al)
            data.plot.scatter(y=c, x='Generation', ax=ax, legend=False,
                              color=colr, alpha=al)
    ax.set_ylim(0, 1.)
    ax.set_xticks(xtick);
    ax.set_xticklabels(xtick, fontdict=ticksfont)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels([0, 0.5, 1.], fontdict=ticksfont)
    ax.set_ylabel('Relative \nabundance', fontdict=labelfont)
    ax.set_xlabel('Generation', fontdict=labelfont)
    ax.set_xlim(0, 60)


def plot_pairwise_dists(ax, mat, dot_dict, species_dict, sp, ang_dict, tloc_dict):


    inds = list(mat.index)
    edges = dot_dict[len(mat)]
    angs = ang_dict[len(mat)]
    tlocs = tloc_dict[len(mat)]

    for ang, comb in enumerate(list(combinations(np.arange(len(mat)), 2))):

        angle = angs[ang]
        if mat.loc[inds[comb[0]], inds[comb[1]]] > 0:

            ax.errorbar(x=[edges[0][comb[0]], edges[0][comb[1]]],
                        y=[edges[1][comb[0]], edges[1][comb[1]]], fmt='--',
                        color='k', lw=mat.loc[inds[comb[0]], inds[comb[1]]] * 10)

        l1 = tlocs[ang]
        ax.text(*l1, str(round(mat.loc[inds[comb[0]], inds[comb[1]]], 2)),
                rotation=angle,
                fontdict={'ha': 'center', 'size': 20})

    for edge in range(len(mat)):

        if len(inds[edge].split('_')) == 1:
            colr = species_dict[inds[edge]]
        else:
            colr = species_dict[inds[edge].replace('_', '').replace(sp, '')]
        if np.isnan(mat.loc[inds[edge], inds[edge]]):

            ax.errorbar(x=edges[0][edge], y=edges[1][edge],
                        ms=30, fmt='o', mfc='grey', mec=colr,alpha = 0.4,
                        mew=(0.01) * 20)
        else:
            ax.errorbar(x=edges[0][edge], y=edges[1][edge],
                        ms=30, fmt='o', mfc='white', mec=colr,
                        mew=(mat.loc[inds[edge], inds[edge]] + 0.01) * 20)
        if edge in [0, 1, 2]:
            ax.text(x=edges[0][edge], y=edges[1][edge] + 0.3,
                    s=round(mat.loc[inds[edge], inds[edge]], 2),
                    fontdict={'ha': 'center', 'size': 20})
        else:
            ax.text(x=edges[0][edge], y=edges[1][edge] - 0.4,
                    s=round(mat.loc[inds[edge], inds[edge]], 2),
                    fontdict={'ha': 'center', 'size': 20})

    sns.despine(top=False, right=False, ax=ax)

    for spine in ax.spines.values():
        spine.set_edgecolor(species_dict[sp])
        spine.set_linewidth(3)
    ax.set_title(sp)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)




