### **Add RECT from TKDE2020 **

We added RECT from the TKDE2020 paper "[Network Embedding with Completely-imbalanced Labels](https://ieeexplore.ieee.org/document/8979355).", which fixes the problem of zero-shot (i.e., completely-imbalanced) label setting in graph embedding tasks. 

1. Add a model file `rect_model.py` in torch_geometric/nn/models.

   -`rect_model.py` implements RECT (or more specifically its supervised part RECT-L) and some related functions like label processing and class semantic knowledge generation.

2. Add an example file `rect.py` in pytorch_geometric/examples/.

​       -`rect.py` shows how to train and evaluate the model.



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

