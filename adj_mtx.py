import os
import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as ss
# from MVGCN.graph import distance_scipy_spatial as dss

def build_A(nodes,maindir):
    all_nodes = dict.fromkeys(sorted([int(v) for v in set([list(nodes[sub][run].keys()) for sub in nodes.keys() for run in nodes[sub].keys()][0])]))
    for vol in all_nodes.keys():
        all_nodes[vol] = np.mean(np.array([float(c) for cntr in [(nodes[sub][run][str(vol)]).strip('[]').split() for sub in nodes.keys() for run in nodes[sub].keys() if str(vol) in nodes[sub][run].keys()] for c in cntr]).reshape((-1,3)), axis=0)
    cntrs=[]
    vols=[]
    for k,v in all_nodes.items():
        # print(k)
        cntrs.append(v)
        vols.append(k)
    np.savetxt(os.path.join(maindir,'vols.txt'),vols)

    dists = squareform(pdist(cntrs,metric='euclidean'))
    dists = dists + np.eye(len(all_nodes))*np.mean(dists,axis=1)

    kdists =  dists / np.amax(dists, axis=1)[:,np.newaxis]

    idx = np.argsort(dists)[:,1:11]
    # kdists = np.array([[dists[n][i] for i in idx[n]] for n in range(idx.shape[0])])
    # # wt_kdists = kdists / np.amax(kdists,axis=1)[:,np.newaxis]

    A = np.zeros(dists.shape)
    for n in range(idx.shape[0]):
        for i in idx[n]:
            A[n][i] = kdists[n][i]
    return A



if __name__ == "__main__":

    main_dir='/Volumes/ElementsExternal/mridti_test2'

    with open(os.path.join(main_dir,'center_vxls.json'),'r') as n:
        nodes_dict = json.load(n)

    A = build_A(nodes_dict, main_dir)

    np.save(os.path.join(main_dir,'adj_mtrx.npy'), A)

    ax = sns.heatmap(A)
    plt.savefig(os.path.join(main_dir,'adj_mtrx.png'))
