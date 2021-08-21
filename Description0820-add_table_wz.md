### **Add RECT from TKDE2020 **

A proposal of a new GNN model coming from TKDE 2020: https://ieeexplore.ieee.org/document/8979355
The original implementation is: https://github.com/zhengwang100/RECT

## **Dataset and experimental setting**

Three DGL's build-in datasets (Cora, Citeseer, and Pubmed) with their default train/val/test settings are used in this example. In addition, as this paper considers the zero-shot (i.e., completely-imbalanced) label setting, those "unseen" classes should be removed from the training set, as suggested in the paper. In this example, in each dataset, we simply remove the 2-3 classes (i.e., these 2-3 classes are unseen classes) from the labeled training set. Then, we obtain graph embedding results by different models. Finally, with the obtained embedding results and the original balanced labels, we train a logistic regression classifier to evaluate the model performance.

## **Usage** 

`python rect.py --dataset cora --removed-class 1 2 3` #reproducing the RECT-L on "cora" datasets in the zero-shot label setting

`python run_gcn_feats.py --dataset cora --removed-class 1 2 3` #reproducing the GCN on "cora" datasets in the zero-shot label setting and evaluating the original node features

## **Performance**

The performance results are are as follows:

<center><B>Table 1: Node classification results with some classes as "unseen"</B></center>

<table>
        <tr>
                <td align="center"><B>Datasets</B></td>
                <td align="center" colspan="2"><B>Citeseer</B></td>
                </td>
                <td align="center" colspan="2"><B>Cora</B></td>
                </td>
                <td align="center"><B>Pubmed</B></td>
        </tr>
        <tr>
                <td align="center"><B>Unseen Classes</B></td>
                <td align="center">{1, 2, 5}</td>
                <td align="center">{3, 4}</td>
                <td align="center">{1, 2, 3}</td>
                <td align="center">{3, 4, 6}</td>
                <td align="center">{2}</td>
        </tr>
        <tr>
                <td align="center"><B>RECT-L</B></td>
            <td align="center"><B>66.30</B></td>
                <td align="center"><B>68.20</B></td>
                <td align="center"><B>74.60</B></td>
                <td align="center"><B>71.20</B></td>
                <td align="center"><B>75.30</B></td>
        </tr>
        <tr>
                <td align="center"><B>GCN</B></td>
                <td align="center">51.80</td>
                <td align="center">55.70</td>
                <td align="center">55.80</td>
                <td align="center">57.10</td>
                <td align="center">59.80</td>
        </tr>
        <tr>
            <td align="center"><B>NodeFeats</B></td>
                <td align="center">61.40</td>
                <td align="center">61.40</td>
                <td align="center">57.50</td>
                <td align="center">57.50</td>
                <td align="center">73.10</td>
        </tr>
</table>
We use three PyG's build-in datasets (Citeseer, Cora and Pubmed) with their default train/val/test settings. As shown above, RECT significantly outperforms GCN by 20-35%, relatively. The comparison implementations of RECT, GCN and NodeFeats can be found in our local [project](https://github.com/Fizyhsp/xxx/compare_feats_gcn.py).

We think RECT should be part of PyG, as the zero-shot label setting (when some classes do not have any labels) is very common in practical applications. On the other hand, existing methods (like GCN) perform very badly in this case; for example, GCN performs much worse than using raw node features. 

We are very happy to answer any questions and discuss how to integrate our method in PyG in an optimal way. Please feel free to contact us. 

Author: [Tingzhang Zhao](https://github.com/Fizyhsp) (undergraduate, @USTB) , and Zheng Wang (supervisor, the original author of this paper, AP@USTB).

