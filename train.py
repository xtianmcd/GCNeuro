import numpy as np
from scipy.stats import chisquare as X2
import torch.nn.functional as F
from NN import *
from GCN import *
from utils import *
import math
import sys

def train(epoch, adj, all_feats, labels, model, optimizer,n):
    t = time.time()
    model.train()
    o=[]
    l=[]
    optimizer.zero_grad()
    for f in range(len(all_feats)):
        o.append(model(all_feats[f], adj,n))
        #    #if f+1%100==0:
        #    #    output=torch.stack(o)
        #    #    loss_train = F.nll_loss(output,labels[f-99:f+1])
        #    #    acc_train = accuracy(output, labels)
        #    #    loss_train.backward()
        #    #    optimizer.step()
        #    #    o=[]
        #    loss_train = F.nll_loss(op, labels[f])
        #    loss_train.backward()
        #    optimizer.step()
    output=torch.stack(o)
    loss_train = F.nll_loss(output, labels)
    acc_train = accuracy(output, labels)
    loss_train.backward()
    optimizer.step()
    # model.eval()
    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'time: {:.4f}s'.format(time.time() - t))
        #'loss_val: {:.4f}'.format(loss_val.item()),
        #'acc_val: {:.4f}'.format(acc_val.item()),
        #'time: {:.4f}s'.format(time.time() - t))
    return loss_train, acc_train #loss_val, acc_val, loss_train, acc_train


def test(adj, features, labels,model,n):
    model.eval()
    with torch.no_grad():
        o=[]
        for feats in features:
            output = model(feats, adj,n)
            o.append(output)
        output=torch.stack(o)
        loss_test = F.nll_loss(output, labels)
        acc_test = accuracy(output, labels)

        o_pos = [out[1] for out in output.detach()]
        auc = sklearn.metrics.roc_auc_score(labels,o_pos)

        print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return output, loss_test, acc_test, auc


if __name__=='__main__':
    # Training settings
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='Disables CUDA training.')
    # parser.add_argument('--fastmode', action='store_true', default=False,
    #                     help='Validate during training pass.')
    # parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    # parser.add_argument('--epochs', type=int, default=200,
    #                     help='Number of epochs to train.')
    # parser.add_argument('--lr', type=float, default=0.01,
    #                     help='Initial learning rate.')
    # parser.add_argument('--weight_decay', type=float, default=5e-4,
    #                     help='Weight decay (L2 loss on parameters).')
    # parser.add_argument('--hidden', type=int, default=16,
    #                     help='Number of hidden units.')
    # parser.add_argument('--dropout', type=float, default=0.5,
    #                     help='Dropout rate (1 - keep probability).')
    no_cuda=True

    # args = parser.parse_args()
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    use_cuda = not no_cuda and torch.cuda.is_available()

    main_dir = sys.argv[1]

    adj, pt_features, pt_labels, skl_features, skl_labels, pt_te_features,
            pt_te_labels, skl_te_features, skl_te_labels  = load_data(main_dir)

    print('SUMMARY')
    print(f'adj matrix shape: {adj.shape}')
    print(f'skl features[0][0]: {len(skl_features[0][0])}')
    print(f'skl features[0]:    {len(skl_features[0])}')
    print(f'skl features:       {len(skl_features)}')
    print(f'skl labels[0]: {skl_labels[0].shape}')
    print(f'skl labels:    {len(skl_labels)}')
    print(f'pt  features[0][0]: {len(pt_features[0][0])}')
    print(f'pt  features[0]     {len(pt_features[0])}')
    print(f'pt  features:       {len(pt_features)}')
    print(f'pt  labels[0]: {pt_labels[0].shape}')
    print(f'pt  labels:    {len(pt_labels)}')
    bl_te_acc={}
    bl_te_f1={}
    bl_te_auc={}

    seed=42 #42
    epochs= 100 #400
    lr=0.01
    weight_decay = 5e-4
    hidden = 8
    dropout= 0.5
    num_heads=8
    LRalpha=0.2

    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)

    flds=3
    te_losses=[]
    te_acc=[]
    te_auc=[]
    outputs=[]
    te_f1=[]
    cnn_acc=[]
    cnn_f1=[]
    fc_acc=[]
    fc_f1=[]
    fc_auc=[]
    cnn_auc=[]
    for f in range(len(pt_features)):
        te_features = pt_te_features[f]
        te_labels = pt_te_labels[f]
        tr_f3atures = pt_features[f]
        tr_lab3ls = pt_labels[f]

        fact=[]
        rk2=[]
        sl=[]
        tl=[]
        for acq in tr_f3atures:
            fact.append(acq[0])
            rk2.append(acq[1])
            sl.append(acq[2])
            tl.append(acq[3])
        fact=torch.stack(fact)
        rk2=torch.stack(rk2)
        sl=torch.stack(sl)
        tl=torch.stack(tl)
        te_fact=[]
        te_rk2=[]
        te_sl=[]
        te_tl=[]
        for acq in te_features:
            te_fact.append(acq[0])
            te_rk2.append(acq[1])
            te_sl.append(acq[2])
            te_tl.append(acq[3])
        te_fact=torch.stack(te_fact)
        te_rk2=torch.stack(te_rk2)
        te_sl=torch.stack(te_sl)
        te_tl=torch.stack(te_tl)
        tr_algo=[fact,rk2,sl,tl]
        te_algo=[te_fact,te_rk2,te_sl,te_tl]
        algo_acc=[]
        algo_f1=[]
        algo_auc=[]
        for algo in range(len(tr_algo)):
            fcf1a,fcacca,fcauca =
                fcNN(tr_algo[algo],tr_lab3ls,te_algo[algo],te_labels)
            algo_acc.append(fcacca)
            algo_f1.append(fcf1a)
            algo_auc.append(fcauca)
        fcacc = np.mean(algo_acc)
        fcf1 = np.mean(algo_f1)
        fcauc = np.mean(algo_auc)
        fc_acc.append(fcacc)
        fc_f1.append(fcf1)
        fc_auc.append(fcauc)
        print(f'Mean fcNN performce: Accuracy = {fcacc}, F1 = {fcf1:.0f}')

        del tr_algo,te_algo,fact,rk2,sl,tl,te_fact,te_rk2,te_sl,te_tl

        cnn = SimpleCNN()
        f_acc,f_f1,cnnauc =
            CNN(cnn, 32, 100, 0.01,tr_f3atures,tr_lab3ls,te_features,te_labels)
        print(f'CNN Test Acc: {f_acc}')
        print(f'CNN Test F1:  {f_f1}')
        cnn_acc.append(f_acc)
        cnn_f1.append(f_f1)
        cnn_auc.append(cnnauc)

        model = GCNetwork(nfeat=tr_f3atures[0][0].shape[1],
                        nhid=hidden,
                        nclass=tr_lab3ls.max().item() + 1,
                        dropout=dropout,
                        nheads=num_heads,
                          alpha=LRalpha)
        optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)
        v_losses = [0,0,0,0,0]
        v_accs   = []
        tr_losses = []
        tr_accs = []
        tr_losses5=[0,0,0,0,0]

        tr_features = tr_f3atures #[:int(0.9*len(tr_f3atures))]
        tr_labels = tr_lab3ls #[:int(0.9*len(tr_f3atures))]
        #val_features = tr_f3atures[int(0.9*len(tr_f3atures)):]
        #val_labels = tr_lab3ls[int(0.9*len(tr_f3atures)):]

        #Train model
        t_total = time.time()
        for epoch in range(epochs):
                with open('attentions.txt','a') as attn:
                    attn.write(str(epoch))
                tr_idx = np.random.permutation(len(tr_features))
                tr_fold_lbls_sh=[]
                tr_fold_ftrs_sh=[]
                for tri in tr_idx:
                    tr_fold_ftrs_sh.append(tr_features[tri])
                    tr_fold_lbls_sh.append(tr_labels[tri])
                tr_features = tr_fold_ftrs_sh
                tr_labels = torch.stack(tr_fold_lbls_sh)
                tr_loss=[]
                tracc=[]
                for batch in range(0,len(tr_features),32):
                    batch_features = tr_features[batch:batch+32]
                    batch_labels = tr_labels[batch:batch+32]
                    tloss,tacc =
                        train(epoch, adj, tr_features,
                        tr_labels, model, optimizer,f)
                    tr_loss.append(tloss.detach())
                    tracc.append(tacc.detach())
                tr_losses.append(torch.mean(torch.stack(tr_loss)))
                tr_accs.append(torch.mean(torch.stack(tracc))*100)
                """ if using validation set, uncomment the lines below """
                #if epoch%5==0:
                #    tr_losses5[4] = tr_losses5[3]
                #    tr_losses5[3] = tr_losses5[2]
                #    tr_losses5[2] = tr_losses5[1]
                #    tr_losses5[1] = tr_losses5[0]
                #    tr_losses5[0] = tloss#

                #    f_output,f_l,f_a=test(adj,val_features,val_labels,model)
                #    print(tloss.detach(),f_l)
                #    #v_losses[4] = v_losses[3]
                #    v_losses[3] = v_losses[2]
                #    v_losses[2] = v_losses[1]
                #    v_losses[1] = v_losses[0]
                #    v_losses[0] = f_l
                #    if v_losses[3] <= v_losses[2] and v_losses[2] <= v_losses[1] and v_losses[1] <= v_losses[0] and epoch>20:
                #        print(f'Ending optimization at epoch {epoch}')
                #        break
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        tst_fld_acc=[]
        tst_fld_f1=[]
        tst_fld_losses=[]
        f_output,f_l,f_a,f_auc=test(adj,te_features,te_labels,model,f)
        f_output = f_output.detach()
        preds = f_output.max(1)[1].type_as(te_labels)
        f1 = sklearn.metrics.f1_score(te_labels,preds)
        te_auc.append(f_auc)
        te_f1.append(f1)
        te_acc.append(f_a.detach())
        te_losses.append(f_l.detach())
    #print('Test Losses: ',torch.stack(fld_te_losses).mean(),torch.stack(fld_te_losses).std())
    #print('Test Accuracy :',torch.stack(fld_te_acc).mean(),torch.stack(fld_te_acc).std())
    bl_te_acc['GCN']=te_acc
    bl_te_f1['GCN']=te_f1
    bl_te_auc['GCN']=te_auc
    print('Test Losses:   ',torch.stack(te_losses).mean(),torch.stack(te_losses).std())
    print('Test Accuracy: ',torch.stack(te_acc).mean(),torch.stack(te_acc).std())
    print('TesT F1:       ',np.mean(te_f1),np.std(te_f1))

    bl_te_acc['CNN']=cnn_acc
    bl_te_f1['CNN']=cnn_f1
    bl_te_acc['fcNN']=fc_acc
    bl_te_f1['fcNN']=fc_f1
    bl_te_auc['CNN']=cnn_auc
    bl_te_auc['fcNN']=fc_auc


    for f in range(len(skl_features)):
        tr_features = skl_features[f]
        tr_labels = skl_labels[f]
        te_features = skl_te_features[f]
        te_labels = [ls for ls in skl_te_labels[f]]


        tst_acc, tst_f1, clf, tst_auc = baselineML(tr_features,tr_labels,
                                                        te_features,te_labels)
        for i,c in enumerate(clf):
            if c not in bl_te_acc.keys(): bl_te_acc[c]=[]
            if c not in bl_te_f1.keys():  bl_te_f1[c]=[]
            if c not in bl_te_auc.keys(): bl_te_auc[c]=[]
            bl_te_acc[c].append(tst_acc[c])
            bl_te_f1[c].append(tst_f1[c])
            bl_te_auc[c].append(tst_auc[c])
    with open('baselines.txt','a') as bl:
        bl.write('------NEW RESULTS----')
    for c in bl_te_acc.keys():
        print(c)
        print(f'Acc: {np.mean(bl_te_acc[c])}, +/-{np.std(bl_te_acc[c])}')
        print(f'F1 : {np.mean(bl_te_f1[c])}, +/-{np.std(bl_te_f1[c])}')
        print(f'AUC : {np.mean(bl_te_auc[c])}, +/-{np.std(bl_te_f1[c])}')
        with open('baselines.txt','a') as bl:
            bl.write(c)
            bl.write(f'\tAcc: {bl_te_acc[c]}\t')
            bl.write(f'F1:  {bl_te_f1[c]}\t')
            bl.write(f'AUC: {bl_te_auc[c]}')
            bl.write('\n')
