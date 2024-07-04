# name="vit_s_1k_dual_run_sam_test3_concat"
# root_dir="/home/abbas/AUT/Dataset/DenseUAV/google2"
# val_dir="/home/abbas/AUT/Dataset/DenseUAV/google2/validation"
# data_dir=$root_dir/train
# seg_dir=$root_dir/train_segment
# seg_val_dir=$root_dir/seg_val
# seg_test_dir=$root_dir/test_segment
# test_dir=$root_dir/test
# gpu_ids=0
# num_worker=8
# lr=0.01
# batchsize=16
# sample_num=1
# block=1
# num_bottleneck=512
# backbone="ViTS-224" # resnet50 ViTS-224 senet
# head="SingleBranch"
# head_pool="avg" # global avg max avg+max
# cls_loss="CELoss" # CELoss FocalLoss
# feature_loss="WeightedSoftTripletLoss" # TripletLoss HardMiningTripletLoss WeightedSoftTripletLoss ContrastiveLoss
# kl_loss="KLLoss" # KLLoss
# h=224   
# w=224
# load_from="no"
# ra="satellite"  # random affine
# re="satellite"  # random erasing
# cj="no"  # color jitter
# rr="uav"  # random rotate
# num_epochs=120

# lr_step1=70
# lr_step2=110
# lr_step3=2000
# lr_step4=4000
# val_step=2
# seg_backbone="ViTS-224"
# segmentaion=True


# python3 dual_train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
#                 --block $block --lr $lr --num_worker $num_worker --head $head  --head_pool $head_pool \
#                 --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
#                 --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
#                 --num_epochs $num_epochs --lr_step1 $lr_step1 --lr_step2 $lr_step2 --lr_step3 $lr_step3 --lr_step4 $lr_step4 --val_dir $val_dir --val_step $val_step \
#                 --segmentaion $segmentaion --seg_dir $seg_dir --seg_val_dir $seg_val_dir

# cd checkpoints/$name
# python3 test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker --seg_test_dir $seg_test_dir
# python3 evaluate_gpu.py
# python3 evaluateDistance.py --root_dir $root_dir
# cd ../../

name="mvit_s_1k_dual_run_sam_elementvisemax"
root_dir="/home/abbas/AUT/Dataset/DenseUAV/google2"
val_dir="/home/abbas/AUT/Dataset/DenseUAV/google2/validation"
data_dir=$root_dir/train
seg_dir=$root_dir/train_segment
seg_val_dir=$root_dir/seg_val
seg_test_dir=$root_dir/test_segment
test_dir=$root_dir/test
gpu_ids=0
num_worker=8
lr=0.015
batchsize=16
sample_num=1
block=1
num_bottleneck=512
backbone="eva02_small_patch14_224.mim_in22k" # resnet50 ViTS-224 senet
head="SingleBranch"
head_pool="avg" # global avg max avg+max
cls_loss="CELoss" # CELoss FocalLoss
feature_loss="WeightedSoftTripletLoss" # TripletLoss HardMiningTripletLoss WeightedSoftTripletLoss ContrastiveLoss
kl_loss="KLLoss" # KLLoss
h=224   
w=224
load_from="no"
ra="satellite"  # random affine
re="satellite"  # random erasing
cj="no"  # color jitter
rr="uav"  # random rotate
num_epochs=150

lr_step1=760
lr_step2=100
lr_step3=2000
lr_step4=4000
val_step=2
seg_backbone="ViTS-224"
segmentaion=True


python3 dual_train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
                --block $block --lr $lr --num_worker $num_worker --head $head  --head_pool $head_pool \
                --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
                --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
                --num_epochs $num_epochs --lr_step1 $lr_step1 --lr_step2 $lr_step2 --lr_step3 $lr_step3 --lr_step4 $lr_step4 --val_dir $val_dir --val_step $val_step \
                --segmentaion $segmentaion --seg_dir $seg_dir --seg_val_dir $seg_val_dir

cd checkpoints/$name
python3 test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker --seg_test_dir $seg_test_dir
python3 evaluate_gpu.py
python3 evaluateDistance.py --root_dir $root_dir
cd ../../

