import os
import json
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import torch.optim as optim
import time
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import sklearn
from torch.autograd import Variable
from NN import *

def encode_onehot(lbls,folds=False,n_folds=1):
    if folds:
        classes = set([label for labels in lbls for label in labels])
        labels_onehot=[]
        for f in range(n_folds):
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
            print(classes_dict)
            fold_labels_onehot = np.array(list(map(classes_dict.get, lbls[f])),
                             dtype=np.int32)
            labels_onehot.append(fold_labels_onehot)
    else:
        classes = set(lbls)
        classes_dict =
            {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot =
            np.array(list(map(classes_dict.get, lbls)), dtype=np.int32)

    return labels_onehot

def gen_feats(rundir):
    f_mts = []
    print(f' generating features for {rundir}')
    try:
        with open(os.path.join(rundir,'algo_vol-trk_map.json'),'r') as f:
            feats = json.load(f)
        for algo in feats.keys():
            fts = np.zeros((len(feats[algo]),len(feats[algo])))
            for k in range(len(feats[algo].keys())):
                for l in range(len(feats[algo].keys())):
                    if k!=l:
                        cnnxn = len(set(feats[algo]\
                                    [list(feats[algo].keys())[k]]).\
                                    intersection(feats[algo]\
                                    [list(feats[algo].keys())[l]]))
                        fts[k,l] = cnnxn

            fts =  fts / np.maximum(1,np.amax(fts, axis=1))[:,np.newaxis]
            f_mts.append(fts)
        f_mts = np.array(f_mts)
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

def traintest(f_keys, subs):
    shuffle_index = np.random.permutation(subs)
    tr_s = shuffle_index[:int(0.8*len(subs))]
    te_s = shuffle_index[int(0.8*len(subs)):]
    tr_k = np.random.permutation([k for k in f_keys if k.split('_')[0] in tr_s])
    te_k = np.random.permutation([k for k in f_keys if k.split('_')[0] in te_s])
    return tr_k, te_k

def load_data(maindir):
    # build graph
    adj = np.load(os.path.join(maindir,'adj_mtrx20.npy'))
    n_subs=0
    ftrs_dict={}
    labels=[]
    participants=pd.read_csv(os.path.join(maindir,'participants.tsv'),
                                                                delimiter='\t')
    for subdir in os.listdir(maindir):
        if 'sub-' in subdir and os.path.exists(os.path.join(maindir,subdir,'dwi')):
            n_subs+=1
            for run in os.listdir(os.path.join(maindir,subdir,'dwi','tracks')):
                if 'run-' in run:
                    #print(run)
                    imdir =  os.path.join(maindir,subdir,'dwi','tracks',run)
                    if os.path.exists(os.path.join(imdir,'feature_mts.npy')):
                        features = np.load(os.path.join(imdir,'feature_mts.npy'))
                    else:
                        features = gen_feats(imdir)
                    if len(features) and features.shape!=(0,):
                        features = [sp.csr_matrix(ftrs, dtype=np.float32) for
                                    ftrs in features]
                        features = [normalize(ftrs) for ftrs in features]
                        ###features = [torch.FloatTensor(np.array(ftrs.todense())) for ftrs in features]
                        features = [np.array(ftrs.todense()) for ftrs in features]
                        ftrs_dict[run] = features
                        labels.append([run,\
                            participants.query(\
                            'Subject_Num == @subdir & Run_Num == @run.split("_")[1].split("-")[1]'\
                            ).Group.values[0]])
    adj = sp.coo_matrix(adj)
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    tr_ftrs=[]
    tr_lbls=[]
    te_ftrs=[]
    te_lbls=[]
    if len(set(l[1] for l in labels))==2:
        folds=True
        print('2 classes')
        fold_size = sum([1 for l in labels if l[1]=='Control'])
        print(f'{fold_size} examples in Control class')
        n_folds = math.ceil(len(labels)/fold_size)
        print(f'There are {len(labels)} examples total')
        print(f'There will be {n_folds} folds')
        cntrl_keys = [k for k in ftrs_dict.keys() for l in labels if k==l[0]
                        and l[1]=='Control']
        #cntrl_tr_keys, cntrl_te_keys = traintest(cntrl_keys,np.array(list(set(k.split('_')[0] for k in cntrl_keys))))

        pd_keys = [k for k in ftrs_dict.keys() for l in labels if k==l[0]
                        and l[1]=='PD']
        pd_subs = np.random.permutation(list(set(k.split('_')[0]
                        for k in pd_keys)))
        fold_nsubs = math.ceil(len(pd_subs)/n_folds)
        print(fold_nsubs)
        for _ in range(n_folds):
            sh_cntrl = np.random.permutation(cntrl_keys)
            cntrl_k_shuff = []
            for sc in sh_cntrl:
                cntrl_k_shuff.append(sc)
            tr_fold_ftrs=[]
            tr_fold_lbls=[]
            te_fold_ftrs=[]
            te_fold_lbls=[]
            cntrl_subs=[]
            for key in ftrs_dict.keys():
                if key in cntrl_k_shuff[:int(0.8*len(cntrl_keys))]:
                    tr_fold_ftrs.append(ftrs_dict[key])
                    for i in labels:
                        if i[0]==key:
                            tr_fold_lbls.append(i[1])
                            cntrl_subs.append(i[0].split('_')[0])
                elif key in cntrl_k_shuff[int(0.8*len(cntrl_keys)):]:
                    te_fold_ftrs.append(ftrs_dict[key])
                    for i in labels:
                        if i[0]==key:
                            te_fold_lbls.append(i[1])
                            cntrl_subs.append(i[0].split('_')[0])
                fold_subs = np.random.choice(pd_subs,fold_nsubs)
                #print(len(set(cntrl_subs)),' subjects in control set')
                for s in fold_subs[:int(0.8*len(set(cntrl_subs)))]: #pd_subs[start:start+fold_nsubs]:
                    if key.split('_')[0]==s:
                        tr_fold_ftrs.append(ftrs_dict[key])
                        for i in labels:
                            if i[0]==key:
                                tr_fold_lbls.append(i[1])
                for s in fold_subs[int(0.8*len(set(cntrl_subs))):]:
                    if key.split('_')[0]==s:
                        te_fold_ftrs.append(ftrs_dict[key])
                        for i in labels:
                            if i[0]==key:
                                te_fold_lbls.append(i[1])
            tr_idx = np.random.permutation(len(tr_fold_lbls))
            tr_fold_lbls_sh=[]
            tr_fold_ftrs_sh=[]
            for tri in tr_idx:
                tr_fold_lbls_sh.append(tr_fold_lbls[tri])
                tr_fold_ftrs_sh.append(tr_fold_ftrs[tri])
            te_idx = np.random.permutation(len(te_fold_lbls))
            te_fold_lbls_sh=[]
            te_fold_ftrs_sh=[]
            for tei in te_idx:
                te_fold_lbls_sh.append(te_fold_lbls[tei])
                te_fold_ftrs_sh.append(te_fold_ftrs[tei])
            tr_ftrs.append(tr_fold_ftrs_sh)
            tr_lbls.append(tr_fold_lbls_sh)
            te_ftrs.append(te_fold_ftrs_sh)
            te_lbls.append(te_fold_lbls_sh)
            print(f'training ftrs have length {len(tr_fold_ftrs)}')
            print(f'training labels have length {len(tr_fold_lbls)}')
            print(f'testing ftrs have length {len(te_fold_ftrs)}')
            print(f'testing labels have length {len(te_fold_lbls)}')
    else:
        ftrs=[]
        lbls=[]
        folds=False
        n_folds=None
        for key in ftrs_dict.keys():
            #print(key)
            ftrs.append(ftrs_dict[key])
            for i in labels:
                if i[0]==key:
                    lbls.append(i[1])

    tr_lbls = encode_onehot(tr_lbls,folds,n_folds)
    te_lbls = encode_onehot(te_lbls,folds,n_folds)

    if len(set(l[1] for l in labels))==2:
        tr_labels=[]
        for l in tr_lbls:
            tr_labels.append(torch.LongTensor(np.where(l)[1]))
        trch_lbls=tr_labels
        te_labels=[]
        for l in te_lbls:
            te_labels.append(torch.LongTensor(np.where(l)[1]))
        te_trch_lbls=te_labels
    else:
        trch_lbls = torch.LongTensor(np.where(lbls)[1])
        te_trch_lbls = torch.LongTensor(np.where(lbls)[1])
    trch_ftrs=tr_ftrs
    te_trch_ftrs = te_ftrs
    for fld in range(len(trch_ftrs)):
        for acq in range(len(trch_ftrs[fld])):
            for ftrs in range(len(trch_ftrs[fld][acq])):
                trch_ftrs[fld][acq][ftrs] =
                        torch.FloatTensor(trch_ftrs[fld][acq][ftrs])
        for acq in range(len(te_trch_ftrs[fld])):
            for ftrs in range(len(te_trch_ftrs[fld][acq])):
                te_trch_ftrs[fld][acq][ftrs] =
                        torch.FloatTensor(te_trch_ftrs[fld][acq][ftrs])
    return adj, trch_ftrs, trch_lbls, tr_ftrs, tr_lbls, te_trch_ftrs, te_trch_lbls, te_ftrs, te_lbls # , np.array(te_ftrs), te_lbls #,idx_train, idx_val, idx_test

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

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = classes[unique_labels(y_true, y_pred).astype('int')]
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    return ax

def baselineML(tr_f_arr,tr_labels,te_f_arr,te_labels):
    tr_labels = np.array([l[1] for l in tr_labels])
    te_labels = np.array([l[1] for l in te_labels])
    fact=[]
    rk2=[]
    sl=[]
    tl=[]
    te_fact=[]
    te_rk2=[]
    te_sl=[]
    te_tl=[]
    l=[]
    for acq in tr_f_arr:
        fact.append([trks for vertex in acq[0] for trks in vertex])
        rk2. append([trks for vertex in acq[1] for trks in vertex])
        sl.  append([trks for vertex in acq[2] for trks in vertex])
        tl.  append([trks for vertex in acq[3] for trks in vertex])
    fact = np.array(fact)
    rk2  = np.array(rk2)
    sl   = np.array(sl)
    tl   = np.array(tl)
    train_data = [fact,rk2,sl,tl]
    for acq in te_f_arr:
        te_fact.append([trks for vertex in acq[0] for trks in vertex])
        te_rk2. append([trks for vertex in acq[1] for trks in vertex])
        te_sl.  append([trks for vertex in acq[2] for trks in vertex])
        te_tl.  append([trks for vertex in acq[3] for trks in vertex])
    te_fact = np.array(te_fact)
    te_rk2  = np.array(te_rk2)
    te_sl   = np.array(te_sl)
    te_tl   = np.array(te_tl)
    test_data = [te_fact,te_rk2,te_sl,te_tl]
    omit=[]
    """Train various classifiers to get a baseline."""
    clf, clfn, avg_train_accuracy, avg_test_accuracy, avg_train_f1,
        avg_test_f1, avg_exec_time, std_train_accuracy, std_test_accuracy,
        std_train_f1, std_test_f1, std_exec_time, avg_test_auc =
            [], [], [], {}, [], {}, [], [], [], [], [], [], {}
    clf.append(KNeighborsClassifier(n_neighbors=10))
    clfn.append('KNN')
    clf.append(LogisticRegression())
    clfn.append('LogReg')
    clf.append(BernoulliNB(alpha=.01))
    clfn.append('BernNB')
    clf.append(RandomForestClassifier())
    clfn.append('RF')
    clf.append(MultinomialNB(alpha=.01))
    clfn.append('MultNB')
    clf.append(RidgeClassifier())
    clfn.append('Ridge')
    clf.append(SVC(kernel='linear',probability=True))
    clfn.append('LinSVC')
    clf.append(SVC(kernel='poly',probability=True))
    clfn.append('SVC')
    for i,c in enumerate(clf):
        if i not in omit:
            train_accuracy, test_accuracy, train_f1,
                test_f1, exec_time, test_auc  =
                [], [], [], [], [], []
            for algo in range(4):
                print(clfn[i])
                print('algo ',algo)
                t_start = time.process_time()
                c.fit(train_data[algo], tr_labels)
                train_pred = c.predict(train_data[algo])
                test_pred = c.predict(test_data[algo])
                if clfn[i]!='Ridge': test_proba = c.predict_proba(test_data[algo])
                else: test_proba = [[0,0] for _ in test_pred]
                o_pos = [out[1] for out in test_proba]
                auc = sklearn.metrics.roc_auc_score(te_labels,o_pos)
                test_auc.append(auc)
                train_accuracy.append(100*sklearn.metrics.accuracy_score(
                    tr_labels, train_pred))
                test_accuracy.append(100*sklearn.metrics.accuracy_score(
                    te_labels, test_pred))
                train_f1.append(100*sklearn.metrics.f1_score(
                    tr_labels, train_pred, average='weighted'))
                test_f1.append(100*sklearn.metrics.f1_score(
                    te_labels, test_pred, average='weighted'))
                exec_time.append(time.process_time() - t_start)
                #plot_confusion_matrix(te_labels, test_pred, classes=len(set(te_labels)), title="Confusion Matrix")
            #avg_train_accuracy.append(np.mean(train_accuracy))
            avg_test_accuracy[clfn[i]]=np.mean(test_accuracy)
            #avg_train_f1.append(np.mean(train_f1))
            avg_test_f1[clfn[i]]=np.mean(test_f1)
            avg_test_auc[clfn[i]]=np.mean(test_auc)
            #avg_exec_time.append(np.mean(exec_time))
            #std_train_accuracy.append(np.std(train_accuracy))
            #std_test_accuracy.append(np.std(test_accuracy))
            #std_train_f1.append(np.std(train_f1))
            #std_test_f1.append(np.std(test_f1))
            #std_exec_time.append(np.std(exec_time))
    for i in clfn:
        print(i)
        #print('Train accuracy:      {}'.format(avg_train_accuracy[i]))
        print('Test accuracy:       {}'.format(avg_test_accuracy[i]))
        #print('Train F1 (weighted): {}'.format(avg_train_f1[i]))
        print('Test F1 (weighted):  {}'.format(avg_test_f1[i]))
        #print('Execution time:      {}'.format(avg_exec_time[i]))
        print('\n\n')
    return avg_test_accuracy, avg_test_f1, clfn, avg_test_auc

def fcNN(tr_f_arr,tr_labels,te_f_arr,te_labels,epochs=120,learning_rate=0.01):
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    net.train()
    for epoch in range(epochs):
        data, target = Variable(tr_f_arr), Variable(tr_labels)
        data = data.view(-1, 115*115)
        optimizer.zero_grad()
        net_out = net(data)
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()
    net_out = net(data)
    loss = criterion(net_out, target)
    loss.backward()
    optimizer.step()
    print('Train Epoch: {}\tLoss: {:.6f}'.format(
                        epoch, loss))
    test_loss = 0
    correct = 0
    divisor=0
    net.eval()
    with torch.no_grad():
        te_f1=[]
        te_loss=[]
        data, target = Variable(te_f_arr), Variable(te_labels)
        data = data.view(-1,115*115)
        net_out = net(data)
        test_loss += criterion(net_out, target)
        pred = net_out.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target).sum()
        f1=sklearn.metrics.f1_score(target,pred)
        o_pos = [out[1] for out in net_out.detach()]
        auc = sklearn.metrics.roc_auc_score(target,o_pos)
    test_loss /= len(te_f_arr) # divisor
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), F1: {}\n'.format(
            test_loss, correct, len(te_f_arr),
            100. * correct / len(te_f_arr), f1))

    return f1, float(correct) / float(len(te_f_arr)), auc

def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return(output)

#DataLoader takes in a dataset and a sampler for loading (num_workers deals with system level memory)
def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                        sampler=train_sampler, num_workers=2)
    return(train_loader)

def createLossAndOptimizer(net, learning_rate=0.001):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    return(loss, optimizer)

def CNN(net, batch_size, n_epochs, learning_rate,tr_f_arr,tr_lab3ls,te_f_arr,te_labels):

    print("===== CNN HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    loss, optimizer = createLossAndOptimizer(net, learning_rate)
    training_start_time = time.time()

    tr_features = tr_f_arr #[:int(0.9*len(tr_f_arr))]
    tr_labels = tr_lab3ls #[:int(0.9*len(tr_f_arr))]
    #val_features = tr_f_arr[int(0.9*len(tr_f_arr)):]
    #val_labels = tr_lab3ls[int(0.9*len(tr_f_arr)):]

    tr_f_arr = [torch.stack(acq[:]) for acq in tr_features]
    #val_features = [torch.stack(acq[:]) for acq in val_features]
    te_f_arr = [torch.stack(acq[:]) for acq in te_f_arr]
    tst_acc=[]
    tst_f1=[]
    net.train()
    for epoch in range(n_epochs):

        running_loss = 0.0
        start_time = time.time()
        total_train_loss = 0
        for i in range(0,len(tr_f_arr),batch_size):
            inputs = torch.stack(tr_f_arr[i:i+batch_size])
            labels = tr_labels[i:i+batch_size]
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            running_loss += loss_size
            total_train_loss += loss_size
            if i%10==0:
                print("Epoch {}, \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch+1, running_loss/10, time.time() - start_time))
            running_loss = 0.0
            start_time = time.time()

        #total_val_loss = 0
        #for inputs, labels in val_loader:
        #epoch_acc=[]
        #epoch_f1=[]
        #with torch.no_grad():
        #    #for i in range(0,len(val_features),batch_size):
        #    #Wrap tensors in Variables
        #    te_inputs = Variable(torch.stack(val_features))
        #    te_lab3ls = Variable(val_labels)

        #    #Forward pass
        #    te_outputs = net(te_inputs)
        #    te_loss_size = loss(te_outputs, te_lab3ls)
        #    total_val_loss = te_loss_size
        #    cnn_preds = te_outputs.max(1)[1].type_as(te_lab3ls)
        #    epoch_acc=sklearn.metrics.accuracy_score(te_lab3ls,cnn_preds)
        #    epoch_f1=sklearn.metrics.f1_score(te_lab3ls,cnn_preds)
        #    #epoch_acc.append(f_acc)
        #    #epoch_f1.append(f_f1)
        #print(f'CNN Test Acc: {epoch_acc}')
        #print(f'CNN Test F1:  {epoch_f1}')
        #tst_acc.append(np.mean(epoch_acc))
        #tst_f1.append(np.mean(epoch_f1))
        #print("Epoch {} Test loss = {:.2f}".format(epoch,total_val_loss))

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

    total_tst_loss = 0
    tst_acc=[]
    tst_f1=[]
    tst_auc=[]
    net.eval()
    with torch.no_grad():
        for i in range(0,len(te_f_arr),batch_size):
            te_inputs = Variable(torch.stack(te_f_arr[i:i+batch_size]))
            te_lab3ls = Variable(te_labels[i:i+batch_size])
            te_outputs = net(te_inputs)
            te_loss_size = loss(te_outputs, te_lab3ls)
            total_tst_loss += te_loss_size
            cnn_preds = te_outputs.max(1)[1].type_as(te_lab3ls)
            f_acc=sklearn.metrics.accuracy_score(te_lab3ls,cnn_preds)
            f_f1=sklearn.metrics.f1_score(te_lab3ls,cnn_preds)
            tst_acc.append(f_acc)
            tst_f1.append(f_f1)
            o_pos = [out[1] for out in te_outputs.detach()]
            auc = sklearn.metrics.roc_auc_score(te_lab3ls,o_pos)
            tst_auc.append(auc)
    print(f'Final CNN Test Acc: {np.mean(tst_acc)}')
    print(f'Final CNN Test F1:  {np.mean(tst_f1)}')

    return np.mean(tst_acc), np.mean(tst_f1), np.mean(tst_auc)

def analy(te_out_feats,te_labels):

    tsne = TSNE(n_components=2).fit_transform(te_out_feats)

    reduced_df = np.vstack((tsne.T, te_labels)).T
    reduced_df = pd.DataFrame(data=reduced_df, columns=["X", "Y", "label"])
    reduced_df.label = reduced_df.label.astype(np.int)
    g = sns.FacetGrid(reduced_df, hue='label', size=8).map(plt.scatter, 'X', 'Y').add_legend()
    return

def chi2(labels):
    exp = np.zeros(len(set(np.array(labels))))
    obs = np.zeros(len(set(np.array(labels))))
    for i in labels[idx_train]:
        exp[labels[idx_train][i]]+=1
    for i in labels[idx_test]:
        obs[labels[idx_train][i]]+=1
    print(exp)
    print(obs)
    exp = np.multiply(exp,len(labels[idx_test]))[[1,2,4,5,6]]
    obs = np.multiply(obs,len(labels[idx_train]))[[1,2,4,5,6]]
    print(exp)
    print(obs)
    print(X2(exp,obs))
    print(X2(obs,exp))
    return

if __name__=="__main__":

    #main_dir =
    #main_dir =
    #load_data(main_dir)
    Parallel(n_jobs=-2,verbose=50)(delayed(gen_feats)(os.path.join(main_dir,
        subdir,'dwi','tracks',run)) for subdir in os.listdir(main_dir) if
        'sub-' in subdir for run in os.listdir(os.path.join(main_dir,subdir,
        'dwi','tracks')) if 'run-' in run)
