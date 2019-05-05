import os
import json
import numpy as np
import scipy.sparse as sp
import torch

# def load_data(path="../data/cora/", dataset="cora"):
#     """Load citation network dataset (cora only for now)"""
#     print('Loading {} dataset...'.format(dataset))
#
#     idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
#                                         dtype=np.dtype(str))
#     features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     labels = encode_onehot(idx_features_labels[:, -1])
#
#     # build graph
#     idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#     idx_map = {j: i for i, j in enumerate(idx)}
#     edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
#                                     dtype=np.int32)
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                      dtype=np.int32).reshape(edges_unordered.shape)
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                         shape=(labels.shape[0], labels.shape[0]),
#                         dtype=np.float32)
#
#     # build symmetric adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#
#     features = normalize(features)
#     adj = normalize(adj + sp.eye(adj.shape[0]))
#
#     idx_train = range(140)
#     idx_val = range(200, 500)
#     idx_test = range(500, 1500)
#
#     features = torch.FloatTensor(np.array(features.todense()))
#     labels = torch.LongTensor(np.where(labels)[1])
#     adj = sparse_mx_to_torch_sparse_tensor(adj)
#
#     idx_train = torch.LongTensor(idx_train)
#     idx_val = torch.LongTensor(idx_val)
#     idx_test = torch.LongTensor(idx_test)
#
#     return adj, features, labels, idx_train, idx_val, idx_test

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def gen_feats(rundir):
    # for sub in os.listdir(maindir):
    #     if os.path.isdir(os.path.join(maindir,sub)):
    #         for run in os.listdir(os.path.join(maindir,sub,'dwi','tracks')):
    #             if os.path.isdir(os.path.join(maindir,sub,'dwi','tracks',run)):
    # f_mtx = np.array([])
    f_mts = []
    try:
        with open(os.path.join(rundir,'algo_vol-trk_map.json'),'r') as f:
            feats = json.load(f)
        for algo in feats.keys():
            # print(algo)
            # print(feats[algo].keys())
            fts = np.zeros((len(feats[algo]),len(feats[algo])))
            # print(len(feats[algo].keys()))
            for k in range(len(feats[algo].keys())):
                # print(k)
                for l in range(len(feats[algo].keys())):
                    if k!=l:
                        cnnxn = len(set(feats[algo]\
                                    [list(feats[algo].keys())[k]]).\
                                    intersection(feats[algo]\
                                    [list(feats[algo].keys())[l]]))
                        # print(cnnxn)
                        fts[k,l] = cnnxn

            fts =  fts / np.maximum(1,np.amax(fts, axis=1))[:,np.newaxis]
            # if not f_mtx.size: f_mtx = fts
            # else: f_mtx = np.dstack((f_mtx,fts))
            f_mts.append(fts)
            # print(f_mtx.shape)
        f_mts = np.array(f_mts)
        # print(f_mts.shape)
        np.save(os.path.join(rundir,'feature_mts.npy'),f_mts)
    except:
        print('No file "algo_vol-trk_map.json" in ', os.path.join(rundir))
    return f_mts

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(np.maximum(1,mx.sum(1)))
    # print(rowsum)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(maindir):
    # print('Loading {} dataset...'.format(dataset))

    # build graph
    adj = np.load(os.path.join(maindir,'adj_mtrx.npy'))
    
    ftrs_dict={}
    for subdir in os.listdir(maindir):
        if 'sub-' in subdir:
            for run in os.listdir(os.path.join(maindir,subdir,'dwi','tracks')):
                if 'run-' in run:
                    #print(run)
                    imdir =  os.path.join(maindir,subdir,'dwi','tracks',run)
                    if os.path.exists(os.path.join(imdir,'algo_vol-trk_map.json')):
                        with open(os.path.join(imdir,'algo_vol-trk_map.json'),'r') as a:
                            j = json.load(a)
                        #for algo in j.keys():
                            #print('\t',algo,' ',algo,' ',len(j[algo].keys()))
                    if os.path.exists(os.path.join(imdir,'feature_mts.npy')):
                        features = np.load(os.path.join(imdir,'feature_mts.npy')) 
                    else:
                        features = gen_feats(imdir)
                    if len(features):
                        features = [sp.csr_matrix(ftrs, dtype=np.float32) for ftrs in features]
                        features = [normalize(ftrs) for ftrs in features]
                        features = [torch.FloatTensor(np.array(ftrs.todense())) for ftrs in features]
                        ftrs_dict[run] = features

    adj = sp.coo_matrix(adj)
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # # idx_train = range(140)
    # # idx_val = range(200, 500)
    # # idx_test = range(500, 1500)
    #
    # features = torch.cat([ftrs.view(ftrs.size()[0],ftrs.size()[1],1) for ftrs in features],-1)
    labels=np.genfromtxt(os.path.join(maindir,'labels.txt'),dtype=np.dtype(str))
    #with open(os.path.join(maindir,'labels.txt'),'r') as l:
    #    for line in l.readlines():
    #        labels[line.split('  ')[0]] = line.split('  ')[1].strip().strip('[]')

    ftrs = []
    lbls = []
    for key in ftrs_dict.keys():
        #print(key)
        ftrs.append(ftrs_dict[key])
        for i in labels:
            #print(i[0].decode("utf-8"))
            if i[0]==key.split('_')[0]:
                #print(i[0].decode("utf-8"))
                #print(i)
                lbls.append([int(i[1]),int(i[2])])
    
    #print(lbls)
    #lbls = encode_onehot(lbls)
    #print(np.where(lbls)[1])
    lbls = torch.LongTensor(np.where(lbls)[1])
    # print(features)
    # print(features.size())
    #
    # # idx_train = torch.LongTensor(idx_train)
    # # idx_val = torch.LongTensor(idx_val)
    # # idx_test = torch.LongTensor(idx_test)

    return adj, ftrs, lbls #,idx_train, idx_val, idx_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)

    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def plot_training(tr_acc, v_acc, tr_objs, v_obj, dp=False,ep=0.1,es=False, wt_decay=False, dropout=False):
    tr_maxiter = tr_acc.index(np.max(tr_acc))
    v_maxiter = v_acc.index(np.max(v_acc))
    f,axrow = plt.subplots(1,2)
    axrow[0].plot(range(len(tr_acc)),tr_acc, label="Training Acc")
    axrow[0].plot(range(len(v_acc)),v_acc, label="Validation Acc", color='g')
    axrow[0].set(xlabel='Training Epochs',ylabel='Accuracy')
    axrow[0].set_title("Accuracy")
    axrow[0].set_ylim(0,100)
    axrow[0].annotate(f'{tr_maxiter},{np.max(tr_acc):.2f}%', xy=(tr_maxiter,
            np.max(tr_acc)),xytext=(tr_maxiter+0.04*len(tr_acc),
            np.max(tr_acc)+7),arrowprops=dict(facecolor='black', shrink=0.05),
            )
    axrow[0].annotate(f'{v_maxiter},{np.max(v_acc):.2f}%', xy=(v_maxiter,
            np.max(v_acc)),xytext=(v_maxiter+0.01*len(v_acc),
            np.max(v_acc)+10),arrowprops=dict(facecolor='black', shrink=0.05),
            )
    #     annotation source; https://matplotlib.org/users/annotations_intro.html
    axrow[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
            ncol=2, fancybox=True, shadow=True)
    axrow[1].plot(range(len(tr_objs)),tr_objs, label="Training Loss")
    axrow[1].plot(range(len(v_obj)),v_obj, label="Validation Loss",color='g')
    axrow[1].set(xlabel='Training Epochs',ylabel="Loss")
    axrow[1].set_title('Objective Function')
    axrow[1].legend(loc='upper right', bbox_to_anchor=(1, 1),
            ncol=1, fancybox=True, shadow=True)
    #legend source: https://pythonspot.com/matplotlib-legend/
    if dp: plt.savefig(f'./gcn_dp_plot_{ep}ep.png')
    elif wt_decay: plt.savefig(f'./gcn_wt_decay.png')
    elif dropout: plt.savefig(f'./gcn_dropout.png')
    else: plt.savefig('./gcn_nonpriv_plot.png')
    return

def plot_hps(WD_acc,D_acc):
    WD_maxiter = WD_acc.index(np.max(WD_acc))
    D_maxiter = D_acc.index(np.max(D_acc))

    f,axrow = plt.subplots(1,2)
    axrow[0].plot(range(len(WD_acc)),WD_acc)
    axrow[0].set(xlabel='Weight Decay',ylabel='Accuracy')
    axrow[0].set_title("Weight Decay")
    axrow[0].set_ylim(0,100)
    axrow[0].annotate(f'{WD_maxiter},{np.max(WD_acc):.2f}%', xy=(WD_maxiter,
                np.max(WD_acc)),xytext=(WD_maxiter+0.04*len(WD_acc),
                np.max(WD_acc)+7),arrowprops=dict(facecolor='black', shrink=0.05),
                )
    #annotation source; https://matplotlib.org/users/annotations_intro.html
    axrow[1].plot(range(len(D_acc)),D_acc)
    axrow[1].set(xlabel='Probabilities',ylabel="Accuracy")
    axrow[1].set_title('Dropout')
    axrow[1].set_ylim(0,100)
    axrow[1].annotate(f'{D_maxiter},{np.max(D_acc):.2f}%', xy=(D_maxiter,
                np.max(D_acc)),xytext=(D_maxiter+0.04*len(D_acc),
                np.max(D_acc)+7),arrowprops=dict(facecolor='black', shrink=0.05),
                )
    plt.savefig('./gcn_WD_D.png')
    return

if __name__=="__main__":

    #main_dir = '/Volumes/ElementsExternal/mridti_test2/'
    main_dir = '/data/brain/mridti_small'
    load_data(main_dir)
