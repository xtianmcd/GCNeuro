import os
import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as ss

def build_A(nodes,maindir):
    all_nodes = dict.fromkeys(sorted(set([int(v) for vols in [list(nodes[sub][run].keys()) for sub in nodes.keys() for run in nodes[sub].keys()] for v in vols])))
    print(len(all_nodes.keys()))
    for vol in all_nodes.keys():
        all_nodes[vol] = np.mean(np.array([float(c) for cntr in [(nodes[sub][run][str(vol)]).strip('[]').split() for sub in nodes.keys() for run in nodes[sub].keys() if str(vol) in nodes[sub][run].keys()] for c in cntr]).reshape((-1,3)), axis=0)
    cntrs=[]
    vols=[]
    for k,v in all_nodes.items():
        cntrs.append(v)
        vols.append(k)
    np.savetxt(os.path.join(maindir,'vols.txt'),vols)

    dists = squareform(pdist(cntrs,metric='euclidean'))

    kdists = np.divide(dists,np.amax(dists))
    idx = np.argsort(dists)[:,1:11]

    A = np.zeros(dists.shape)
    for n in range(idx.shape[0]):
        for i in idx[n]:
            A[n][i] = kdists[n][i]

    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            if A[r][c]>0 and A[r][c]!=A[c][r]:
                A[c][r]=A[r][c]

    A = A + np.eye(len(all_nodes))*np.mean(kdists,axis=1) # add self-connections

    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            if A[r][c]!=A[c][r]:
                print(f'assymetry at row {r}, col {c}')
    return A



if __name__ == "__main__":

    #main_dir='/Volumes/ElementsExternal/mridti_test2'
    main_dir = '/data/brain/mridti_small'

    with open(os.path.join(main_dir,'center_vxls.json'),'r') as n:
        nodes_dict = json.load(n)

    A = build_A(nodes_dict, main_dir)

    np.save(os.path.join(main_dir,'adj_mtrx.npy'), A)

    ax = sns.heatmap(A)
    plt.savefig(os.path.join(main_dir,'adj_mtrx.png'))
