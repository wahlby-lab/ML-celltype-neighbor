import networkx as nx
import os, datetime
import pandas as pd
import numpy as np
import random as rn
import matplotlib.pyplot as plt
import sys
import scipy
from scipy.spatial import cKDTree as KDTree
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
pd.options.display.max_columns = None
pd.options.display.max_rows = 100

def _kdistances(elems_df,x,y):
    kdT = KDTree(np.array([elems_df[x].values,elems_df[y].values]).T)
    d,i = kdT.query(np.array([elems_df[x].values,elems_df[y].values]).T,k=2)
    return d

# Auxiliary function to compute d_max
def findDmax(elems_df,x,y,groupby=None,debug=False, plot=False):
    d_list=[]

    if groupby is not None:
        for elem in elems_df[groupby].unique():
            elems_df_tmp = elems_df.loc[elems_df[groupby]==elem].copy()
            elems_df_tmp.reset_index(drop=True, inplace=True)
            if not elems_df_tmp.empty:
                # Find mean distance to nearest neighbor
                d=_kdistances(elems_df_tmp,x,y)
                d_list.append(d)
    else:
        d_list=_kdistances(elems_df,x,y)
    
    d = np.vstack(d_list)
    d_th= np.percentile(d[:,1],97)
    
    if debug:
        print(d.shape)
        print(d_th)
    
    if plot:     
        plt.hist(d[:,1],bins=200)
        plt.axvline(x=d_th,c='r')
        plt.show()

    return d_th


#BE careful1! this procedure just appends tings to the graph, maybe I should fix that
def buildGraph(elems_df,d_th,main,x,y,groupby=None):
    if "feature" in elems_df:
        print("The column feature will be re written")
        elems_df.drop(columns=["feature"],inplace=True)
    G = nx.Graph()
    n =0
    label_df =pd.DataFrame(elems_df[main])
    main_list=label_df[main].unique()
    mlshape=main_list.shape[0]
    one_hot_encoding = dict(zip(main_list,to_categorical(np.arange(mlshape),num_classes=mlshape).tolist()))
    if groupby is not None:
        for elem in elems_df[groupby].unique():
            elems_df_tmp = elems_df.loc[elems_df[groupby]==elem].copy()
            elems_df_tmp.reset_index(drop=True, inplace=True)
            if not elems_df_tmp.empty:
                elems_df_tmp["feature"] = elems_df_tmp[main].map(one_hot_encoding).tolist()
                kdT = KDTree(np.array([elems_df_tmp[x].values,elems_df_tmp[y].values]).T)
                res = kdT.query_pairs(d_th)
                res = [(x[0]+n,x[1]+n) for x in list(res)]
                # Add nodes
                G.add_nodes_from((elems_df_tmp.index.values+n), test=False, val=False, label=0)
                nx.set_node_attributes(G,dict(zip((elems_df_tmp.index.values+n), elems_df_tmp.feature)), 'feature')
                # Add edges
                G.add_edges_from(res)
                n = n + elems_df_tmp.shape[0]
    else:
        elems_df.reset_index(drop=True, inplace=True)
        elems_df["feature"] = elems_df[main].map(one_hot_encoding).tolist()

        kdT = KDTree(np.array([elems_df[x].values,elems_df[y].values]).T)
        res = kdT.query_pairs(d_th)
        res = [(x[0],x[1]) for x in list(res)]
        # Add nodes to graph
        G.add_nodes_from((elems_df.index.values), test=False, val=False, label=0)
        # Add node features to graph
        nx.set_node_attributes(G,dict(zip((elems_df.index.values), elems_df["feature"])), 'feature')
        # Add edges to graph
        G.add_edges_from(res)

    return G#, elems_df


def post_merge(df, labels, post_merge_cutoff, linkage_method='single', 
               linkage_metric='correlation', fcluster_criterion='distance', name='', save=False):
    
    Z = scipy.cluster.hierarchy.linkage(df.T, method=linkage_method, metric=linkage_metric)
    merged_labels_short = scipy.cluster.hierarchy.fcluster(Z, post_merge_cutoff, criterion=fcluster_criterion)

    #Update labels  
    label_conversion = dict(zip(df.columns, merged_labels_short))
    label_conversion_r = dict(zip(merged_labels_short, df.columns))
    new_labels = [label_conversion[i] for i in labels] 

    #Plot the dendrogram to visualize the merging
    fig, ax = plt.subplots(figsize=(20,10))
    scipy.cluster.hierarchy.dendrogram(Z, labels=df.columns ,color_threshold=post_merge_cutoff)
    ax.hlines(post_merge_cutoff, 0, ax.get_xlim()[1])
    ax.set_title('Merged clusters')
    ax.set_ylabel(linkage_metric, fontsize=20)
    ax.set_xlabel('pre-merge cluster labels', fontsize=20)
    ax.tick_params(labelsize=10)
    
#     if save == True:
#         fig.savefig('../figures/{}.svg'.format(name), dpi=500)

    return new_labels