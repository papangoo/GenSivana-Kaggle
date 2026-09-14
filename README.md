# GenSivana Real or Fake: results and code

Kaggle account: papangoo.

| Competition | Rank | Final score | Metric |
|---|---|---|---|
| Level 1, Detection | 2 of 19 | 1.00000 | F1, positive = manipulated |
| Level 2, Classification | 3 of 15 | 1.00000 | macro F1 |
| Level 3, Localization | 2 of 10 | 0.89455 | mean IoU over edited images |


## Level 1, `level-1-detection/`

`gensivana-l1-detection.ipynb` produced the scoring submission on its own: a 5-fold CV
ensemble of tf_efficientnetv2_s at native 640px, 3 epochs per fold, RGB, no TTA, with
the decision threshold fitted on the pooled 8835-image out-of-fold set. OOF F1 0.99932,
661 of 993 test images called positive.

## Level 2, `level-2-classification/`

`gensivana-l2-classification.ipynb` produced the scoring submission on its own, and was
the only submission made to that competition. Same backbone at 640px with the SRM
high-pass residual as a 6th to 8th channel, 5 folds, 3 epochs. OOF macro F1 0.99875,
test split 332 real / 330 generated / 331 local_edit.

Its `predictions.npz` is also an input to Level 3: the argmax of those probabilities is
the gate deciding which test images are local edits and therefore get a polygon.

## Level 3, `level-3-localization/`

`gensivana-l3-effnetv2l/` holds the strongest single model: tf_efficientnetv2_l with a
U-Net decoder at 640px, 16 epochs, SRM input, D4 flip TTA, trained on Cat-2 images only.
OOF mIoU 0.8417.

**The submitted 0.89455 was not this notebook's own submission.csv.** It was a weighted
blend of the probability maps from three runs, with the weights fitted on the pooled
out-of-fold set:

| run | weight | OOF mIoU |
|---|---|---|
| effnetv2_l, 16 epochs (this notebook) | 0.4 | 0.8417 |
| effnetv2_m, 14 epochs | 0.4 | 0.8410 |
| effnetv2_m, 14 epochs, seed 1337 | 0.2 | 0.8422 |

Blend OOF 0.8500 at threshold 0.55, public 0.89534, private 0.89455. The other two
notebooks were variants of the same file differing only in encoder, epochs and seed;
they have been removed, and the blend script itself was run locally and is not here.

