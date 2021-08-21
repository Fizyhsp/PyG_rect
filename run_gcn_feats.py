import argparse
import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from rect_model import remove_unseen_classes_from_training
from rect import evaluate_embeds
# 注意，这里，在github project里面，上面和rect相关的路径，你得改下

class GCN(torch.nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super(GCN, self).__init__()      
        self.gcn_1 = GCNConv(in_feats, n_hidden, cached=True,
                             normalize=False)
        self.gcn_2 = GCNConv(n_hidden, n_classes, cached=True,
                             normalize=False)

    def forward(self, inputs, edge_index, edge_attr):
        self.edge_index, self.edge_weight = edge_index, edge_attr
        h = F.relu(self.gcn_1(inputs, self.edge_index, self.edge_weight))
        h = F.dropout(h, training=self.training)
        preds = self.gcn_2(h, self.edge_index, self.edge_weight)
        return F.log_softmax(preds,dim=1)

    # Detach the return variables
    def embed(self, inputs):
        h_1 = self.gcn_1(inputs, self.edge_index, self.edge_weight)
        return h_1.detach()

def process_classids(labels_zs):
    ''' Reorder the remaining classes with unseen classes removed.
        Input: the label only removing unseen classes
        Output: the label with reordered classes
    '''
    labeldict = {}
    num=0
    for i in labels_zs:
        labeldict[int(i)]=1
    labellist=sorted(labeldict)
    for label in labellist:
        labeldict[int(label)]=num
        num=num+1
    for i in range(labels_zs.numel()):
        labels_zs[i]=labeldict[int(labels_zs[i])]
    return labels_zs

def run(args, data):   
    # remove these unseen classes from the training set, to construct the zero-shot label setting
    train_mask_zs = remove_unseen_classes_from_training(train_mask=data.train_mask, labels=data.y, removed_classes=args.removed_class)
    print('after removing the unseen classes, seen class labeled node num:', sum(train_mask_zs).item())    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    print('Training GCN:')
    model, features = GCN(in_feats=dataset.num_features, n_hidden=args.n_hidden, n_classes=dataset.num_classes-len(args.removed_class)).to(device), data.x
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        labels_train = process_classids(labels_zs=data.y[train_mask_zs])
        loss_train=F.nll_loss(model(features, data.edge_index, data.edge_attr)[train_mask_zs], labels_train)
        print('Epoch {:d} | Train Loss {:.5f}'.format(epoch + 1, loss_train.item()))
        loss_train.backward()
        optimizer.step()

    model.eval()
    embeds_gcn = model.embed(features)
    
    # evaluate the quality of embedding results with the original balanced labels, to assess the model performance (as suggested in the paper)
    res_gcn = evaluate_embeds(embeddings=embeds_gcn, labels=data.y, train_mask=data.train_mask, test_mask=data.test_mask)
    print("Test Accuracy of {:s}: {:.4f}".format('GCN', res_gcn))

    res_feats = evaluate_embeds(embeddings=features, labels=data.y, train_mask=data.train_mask, test_mask=data.test_mask)    
    print("Test Accuracy of {:s}: {:.4f}".format('NodeFeats', res_feats))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MODEL')
    parser.add_argument("--dataset", type=str, default='cora', choices=['cora', 'citeseer', 'pubmed'], help="dataset")
    parser.add_argument("--removed-class", type=int, nargs='*', default={1,2,3}, help="remove the unseen classes")
    parser.add_argument("--n-hidden", type=int, default=200, help="number of hidden gcn units")
    args = parser.parse_args()
    
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    data = T.GDC()(dataset[0])
    run(args, data)