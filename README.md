# [TF 2.x] PaDiM - Anomaly Detection Localization

## Dependencies
* Windows 10 | Python 3.8.8 | Tensorflow 2.4.1 GPU
* Scikit-learn | Scikit-image | Matplotlib

## Run
```
# options: seed, rd, target, batch_size, is_plot, net
python main.py
```

## Datasets
[MVTecAD dataset](https://mvtec.com/company/research/datasets/mvtec-ad/)

## Results (AU ROC)
### Implementation results on MVTec
* I failed to retrieve same results from the 
[original github](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master).

* For the full reproducibility, use PyTorch version.  

* Network Type:
    - org.: WideResNet50, Rd 550 (from original github)
    - Net 1: EfficientNetB7 \[layer5, 6, 7], Rd 1000
    - Net 2: EfficientNetB7 \[layer4, 6, 7], Rd 1000

MvTec       | org. (Img)| org. (Patch)  | Net 1 (Img)   | Net 2 (Img)   | Net 1 (Patch) | Net 2 (Patch) |
-----       | -----     | -----         | -----         | -----         | -----         | -----         |
carpet      |   0.999   | 0.990         | 0.9498        | 0.9823        | 0.9727        | 0.8536        |
grid        |   0.957   | 0.965         | 0.9357        | 0.9708        | 0.9581        | 0.7504        |
leather     |   1.000   | 0.989         | 0.9993        | 1.0000        | 0.9861        | 0.9016        |
tile        |   0.974   | 0.939         | 0.9567        | 0.9838        | 0.9049        | 0.7293        |
wood        |   0.988   | 0.941         | 0.9482        | 0.9535        | 0.9460        | 0.8305        |
bottle      |   0.998   | 0.982         | 0.9833        | 0.9960        | 0.9712        | 0.8613        |
cable       |   0.922   | 0.968         | 0.9091        | 0.9185        | 0.9628        | 0.8146        |
capsule     |   0.915   | 0.986         | 0.9462        | 0.9529        | 0.9774        | 0.9401        |
hazelnut    |   0.933   | 0.979         | 0.9825        | 0.9732        | 0.9651        | 0.8763        |
metal_nut   |   0.992   | 0.971         | 0.8685        | 0.9296        | 0.9856        | 0.9255        |
pill        |   0.944   | 0.961         | 0.8822        | 0.8794        | 0.9548        | 0.8929        |
screw       |   0.844	| 0.983         | 0.6319        | 0.7670        | 0.9862        | 0.9413        |
toothbrush  |   0.972   | 0.983         | 0.7667        | 0.9722        | 0.9786        | 0.9367        |
transistor  |   0.978   | 0.987         | 0.9304        | 0.9492        | 0.9772        | 0.9583        |
zipper      |   0.909   | 0.975         | 0.9800        | 0.9858        | 0.9649        | 0.8399        |
Avg. (tex.) |   0.984   | 0.965         | 0.9579        | 0.9781        | 0.9536        | 0.8131        |
Avg. (obj.) |   0.941   | 0.978         | 0.8881        | 0.9323        | 0.9724        | 0.8987        |
Avg. (all)  |   0.955   | 0.973         | 0.9114        | 0.9476        | 0.9961        | 0.8702        |

#### ROC Curve (Net 1) Bottle
![bottle_auroc](assets/AUROC-bottle.png)

#### PR Curve (Net 1) Bottle
![bottle_pr](assets/PR-bottle.png)

#### Localization examples (Net 1) (cherry-picked)
![carpet_ex](assets/carpet_66.png)
![grid_ex](assets/grid_66.png)
![leather_ex](assets/leather_66.png)
![tile_ex](assets/tile_51.png)
![wood_ex](assets/wood_66.png)
![bottle_ex](assets/bottle_66.png)
![cable_ex](assets/cable_66.png)
![capsule_ex](assets/capsule_66.png)
![hazelnut_ex](assets/hazelnut_66.png)
![metalnut_ex](assets/metal_nut_54.png)
![pill_ex](assets/pill_66.png)
![screw_ex](assets/screw_66.png)
![toothbrush_ex](assets/toothbrush_35.png)
![transistor_ex](assets/transistor_86.png)
![zipper_ex](assets/zipper_63.png)
