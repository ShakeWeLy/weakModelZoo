
ssh l0009@gpu01
source ~/envs/train_/bin/activate

python experiments/segment/project1_2610/train.py -m unet --epochs 150
python experiments/detection/project2_2611/train.py -m sgc_yolov5s

python experiments/track/project2_2611/train.py -m siamrpn_pp



### 表3-5 与其他算法对比实验
### Tab.3-5 Comparison experiment with other algorithms

| 算法类型 | 算法名称 | P | 参数量 | 模型大小<br>(MB) | FPS<br>(f/s) | GFLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 通用目标检测<br>网络 | Faster R-CNN | 0.708 | 54525952 | 221 | 8.03 | 129.8 |
| | YOLOv3 | 0.725 | 66060288 | 236.1 | 40.65 | 66.9 |
| 轻量级网络 | YOLOv3-Tiny | 0.648 | 8666692 | 17.4 | 120.68 | 12.9 |
| | YOLOv5s-SFN | 0.679 | 9034984 | 18.0 | 132.71 | 32.2 |
| | YOLOv7-Tiny | 0.726 | 6014988 | 12.3 | 138.65 | 13.2 |
| | YOLOv8s | 0.729 | 11125971 | 21.3 | 130.32 | 28.4 |
| | YOLOv5s | 0.731 | 8375766 | 16.3 | 143.25 | 21.2 |
| | SAG-YOLOv5s | 0.742 | 5064622 | 10.3 | 136.6 | 11.5 |
| | YOLOv5s-CEM | 0.685 | **2889205** | **5.67** | **172.71** | **6.8** |
| | SGC-YOLOv5 | **0.749** | 4313695 | 8.8 | 169.42 | 9.0 |


已用 `best`（epoch 40）在更新后的 test 集上重新跑完。

| 项目 | 结果 |
|---|---|
| 图片数 | 1501 |
| mAP50 | 0.7582 |
| mAP50-95 | 0.4435 |
| P | 0.9167 |
| R | 0.7115 |
