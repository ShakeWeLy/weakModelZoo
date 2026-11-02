
ssh l0009@gpu01
source ~/envs/train_/bin/activate

python experiments/segment/project1_2610/train.py -m unet --epochs 150
python experiments/detection/project2_2611/train.py -m sgc_yolov5s