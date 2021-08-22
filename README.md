# **PyG Implementation of RECT (TKDE20)**

This PyG example implements the GNN model **RECT** (or more specifically its supervised part **RECT-L**) proposed in the paper [Network Embedding with Completely-imbalanced Labels](https://ieeexplore.ieee.org/document/8979355). The authors' original implementation can be found [here](https://github.com/zhengwang100/RECT).

## **Dataset and experimental setting**

Three PyG's build-in datasets (Cora, Citeseer, and Pubmed) with their default train/val/test settings are used in this example. In addition, as this paper considers the zero-shot (i.e., completely-imbalanced) label setting, those "unseen" classes should be removed from the training set, as suggested in the paper. In this example, in each dataset, we simply remove the 1-3 classes (i.e., these 1-3 classes are unseen classes) from the labeled training set. Then, we obtain graph embedding results by different models. Finally, with the obtained embedding results and the original balanced labels, we train a logistic regression classifier to evaluate the model performance.

## **Usage** 

`python rect.py --dataset citeseer --removed-classes 1 2 5` #reproducing the RECT-L on "citeseer" datasets in the zero-shot label setting

`python run_gcn_feats.py --dataset citeseer --removed-classes 1 2 5` #reproducing the GCN on "citeseer" datasets in the zero-shot label setting and evaluating the original node features

## **Performance**

The performance results are are as follows:
<center><B>Table 1: Node classification results with some classes as "unseen"</B></center>
<br/><br/>
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
