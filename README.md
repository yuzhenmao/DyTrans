# DyTrans

This is the pytorch implementation of _**Dynamic Transfer Learning across Graphs**_


### Datasets
Please download the datasets from the original paper listed in our paper. And put them under ''./data'' folder

### Implementation Details
_**DyTrans**_ is firstly pre-trained on the source dataset for 2000 epochs; then it is fine-tuned on the target dataset for 600 epochs using limited labeled data in each class. We use Adam optimizer with learning rate 3e-3. AUC is used as the evaluation metric.

### Demo case: Benchamrk 1 (D5 -> D3)
```
 python dytrans.py --datasets='D3+D5' --finetune_epoch=600  --mu=1e-2 --gnn='gcn' --few_shot=5  --epoch=2000  --heads=4  --m_dim=128  --feat_num=128  --batch_size=-1   --finetune_lr=0.01   --ratio 0.7  --_alpha=0.01  --_alpha=0.01  --only True
```
### Demo case: Benchamrk 5 (D3 -> Br)
```
 python dytrans.py --datasets='Br+D3' --finetune_epoch=600  --mu=1e-2 --gnn='gcn' --few_shot=5  --epoch=2000  --heads=4  --m_dim=64  --feat_num=128  --batch_size=-1   --finetune_lr=0.01   --ratio 0.7  --_alpha=0.05  --_alpha=0.05  --only True
```
