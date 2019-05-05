import numpy as np
from scipy.stats import chisquare as X2
import torch.nn.functional as F
from GCN import *
from utils import *

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

def train(epoch, adj, all_feats, labels, model, optimizer):
    t = time.time()
    model.train()
    o=[]
    for feats in all_feats:
        optimizer.zero_grad()
        o.append(model(feats, adj))
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
    return #loss_val, acc_val, loss_train, acc_train


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return output, loss_test, acc_test


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

    # args = parser.parse_args()
    # args.cuda = not args.no_cuda and torch.cuda.is_available()

    #main_dir = '/Volumes/ElementsExternal/mridti_test2/'
    main_dir = '/data/brain/mridti_small'
    # load_data(main_dir,os.path.join(main_dir,'sub-01','dwi','tracks','sub-01_run-01'))

    # Load data
    adj, features, labels  = load_data(main_dir)

    seed=42 #42
    epochs=10 #200
    lr=0.01 #0.01
    weight_decay=False #5e-4
    hidden = 8 #16
    dropout=False #0.5
    diffpriv=False
    ns=4.0
    C=1.0

    num_heads=2
    LRalpha=0.2

    np.random.seed(seed)
    torch.manual_seed(seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(args.seed)

    # Model and optimizer
    model = GCNetwork(nfeat=features[0][0].shape[1],
                        nhid=hidden,
                        nclass=labels.max().item() + 1,
                        dropout=dropout,
                        nheads=num_heads,
                        alpha=LRalpha)

    # if diffpriv: optimizer = DPSGD(model.parameters(),
    #                        lr=lr, weight_decay=False,
    #                         noise_scale=ns,gclip=C)
    # else:
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)


    v_losses = []
    v_accs   = []
    tr_losses = []
    tr_accs = []

    #Train model
    t_total = time.time()
    for epoch in range(epochs):
        train(epoch, adj, features, labels, model, optimizer)
        # v_losses.append(vloss)
        # v_accs.append(vacc*100)
        # tr_losses.append(tloss)
        # tr_accs.append(tacc*100)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    """
    # Testing
    output,l,a=test()

    #Plot
    plot_training(tr_accs,v_accs,tr_losses,v_losses,dp=diffpriv, wt_decay=weight_decay, dropout=dropout)

    preds = output.max(1)[1].type_as(labels)

    exp = np.zeros(len(set(np.array(labels))))
    obs = np.zeros(len(set(np.array(labels))))
    for i in preds[idx_train]:
        exp[i]+=1
    for i in preds[idx_test]:
        obs[i]+=1
    print(exp)
    print(obs)
    exp = np.multiply(exp,len(preds[idx_test]))[[1,3]]
    obs = np.multiply(obs,len(preds[idx_train]))[[1,3]]
    # print(exp)
    # print(obs)
    print(X2(exp,obs))
    print(X2(obs,exp))
    """
    # wt_decays=[0,0.5, 0.05, 0.005, 0.0005]
    # dropouts = [0.1,0.2, 0.5,0.8]
    #
    # WD_accs=[]
    # # WD_losses=[]
    # D_accs=[]
    # # D_losses=[]
    #
    # for wd in wt_decays:
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     # if args.cuda:
    #     #     torch.cuda.manual_seed(args.seed)
    #
    #     # Load data
    #     adj, features, labels, idx_train, idx_val, idx_test = load_data()
    #
    #     # Model and optimizer
    #     model = GCN(nfeat=features.shape[1],
    #                 nhid=hidden,
    #                 nclass=labels.max().item() + 1,
    #                 dropout=dropout)
    #     if diffpriv: optimizer = DPSGD(model.parameters(),
    #                            lr=lr, weight_decay=False,
    #                             noise_scale=ns,gclip=C)
    #     else: optimizer = optim.SGD(model.parameters(),
    #                            lr=lr, weight_decay=weight_decay)
    #     seed=42 #42
    #     epochs=140 #200
    #     lr=0.01 #0.01
    #     weight_decay=wd #5e-4
    #     hidden = 32 #16
    #     dropout=False #0.5
    #     diffpriv=False
    #     ns=3.0
    #     C=3.0
    #
    #     v_losses = []
    #     v_accs   = []
    #     tr_losses = []
    #     tr_accs = []
    #
    #     #Train model
    #     t_total = time.time()
    #     for epoch in range(epochs):
    #         vloss, vacc, tloss, tacc = train(epoch)
    #     #     v_losses.append(vloss)
    #     #     v_accs.append(vacc*100)
    #     #     tr_losses.append(tloss)
    #     #     tr_accs.append(tacc*100)
    #     print("Optimization Finished!")
    #     print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    #
    #     # Testing
    #     output,tst_loss, tst_acc=test()
    #     WD_accs.append(tst_acc*100)
    # #     WD_losses.append(tst_loss)
    #
    # for d in dropouts:
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     # if args.cuda:
    #     #     torch.cuda.manual_seed(args.seed)
    #
    #     # Load data
    #     adj, features, labels, idx_train, idx_val, idx_test = load_data()
    #
    #     # Model and optimizer
    #     model = GCN(nfeat=features.shape[1],
    #                 nhid=hidden,
    #                 nclass=labels.max().item() + 1,
    #                 dropout=dropout)
    #     if diffpriv: optimizer = DPSGD(model.parameters(),
    #                            lr=lr, weight_decay=False,
    #                             noise_scale=ns,gclip=C)
    #     else: optimizer = optim.SGD(model.parameters(),
    #                            lr=lr, weight_decay=weight_decay)
    #     seed=42 #42
    #     epochs=140 #200
    #     lr=0.01 #0.01
    #     weight_decay=False #5e-4
    #     hidden = 32 #16
    #     dropout=d #0.5
    #     diffpriv=False
    #     ns=3.0
    #     C=3.0
    #
    #     v_losses = []
    #     v_accs   = []
    #     tr_losses = []
    #     tr_accs = []
    #
    #     #Train model
    #     t_total = time.time()
    #     for epoch in range(epochs):
    #         vloss, vacc, tloss, tacc = train(epoch)
    #     #     v_losses.append(vloss)
    #     #     v_accs.append(vacc*100)
    #     #     tr_losses.append(tloss)
    #     #     tr_accs.append(tacc*100)
    #     print("Optimization Finished!")
    #     print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    #
    #     # Testing
    #     output,tst_loss, tst_acc=test()
    #     D_accs.append(tst_acc*100)
    # #     D_losses.append(tst_loss)
    #
    # #Plot
    # plot_hps(WD_accs,D_accs)
