
# name="caf_s36_22k__0015__250"
# root_dir="/home/abbas/AUT/Dataset/DenseUAV/google2"
# val_dir="/home/abbas/AUT/Dataset/DenseUAV/google2/validation"
# data_dir=$root_dir/train
# test_dir=$root_dir/test
# gpu_ids=0
# num_worker=8
# lr=0.0015
# batchsize=16
# sample_num=1
# block=1
# num_bottleneck=512
# backbone="caformer_s36.sail_in22k" # resnet50 ViTS-224 senet
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
# num_epochs=250

# lr_step1=180
# lr_step2=2200
# lr_step3=2000
# lr_step4=4000
# val_step=3

# python3 train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
#                 --block $block --lr $lr --num_worker $num_worker --head $head  --head_pool $head_pool \
#                 --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
#                 --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
#                 --num_epochs $num_epochs --lr_step1 $lr_step1 --lr_step2 $lr_step2 --lr_step3 $lr_step3 --lr_step4 $lr_step4 --val_dir $val_dir --val_step $val_step

# cd checkpoints/$name
# python3 test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
# python3 evaluate_gpu.py
# python3 evaluateDistance.py --root_dir $root_dir
# cd ../../


# name="caf_s36_22k__002__250"
# root_dir="/home/abbas/AUT/Dataset/DenseUAV/google2"
# val_dir="/home/abbas/AUT/Dataset/DenseUAV/google2/validation"
# data_dir=$root_dir/train
# test_dir=$root_dir/test
# gpu_ids=0
# num_worker=8
# lr=0.002
# batchsize=16
# sample_num=1
# block=1
# num_bottleneck=512
# backbone="caformer_s36.sail_in22k" # resnet50 ViTS-224 senet
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
# num_epochs=250

# lr_step1=150
# lr_step2=220
# lr_step3=2000
# lr_step4=4000
# val_step=3

# python3 train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
#                 --block $block --lr $lr --num_worker $num_worker --head $head  --head_pool $head_pool \
#                 --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
#                 --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
#                 --num_epochs $num_epochs --lr_step1 $lr_step1 --lr_step2 $lr_step2 --lr_step3 $lr_step3 --lr_step4 $lr_step4 --val_dir $val_dir --val_step $val_step

# cd checkpoints/$name
# python3 test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
# python3 evaluate_gpu.py
# python3 evaluateDistance.py --root_dir $root_dir
# cd ../../



# name="caf_s36_22k_1k__001__300"
# root_dir="/home/abbas/AUT/Dataset/DenseUAV/google2"
# val_dir="/home/abbas/AUT/Dataset/DenseUAV/google2/validation"
# data_dir=$root_dir/train
# test_dir=$root_dir/test
# gpu_ids=0
# num_worker=8
# lr=0.001
# batchsize=16
# sample_num=1
# block=1
# num_bottleneck=512
# backbone="caformer_s36.sail_in22k_ft_in1k" # resnet50 ViTS-224 senet
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
# num_epochs=300

# lr_step1=200
# lr_step2=2200
# lr_step3=2000
# lr_step4=4000
# val_step=3

# python3 train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
#                 --block $block --lr $lr --num_worker $num_worker --head $head  --head_pool $head_pool \
#                 --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
#                 --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
#                 --num_epochs $num_epochs --lr_step1 $lr_step1 --lr_step2 $lr_step2 --lr_step3 $lr_step3 --lr_step4 $lr_step4 --val_dir $val_dir --val_step $val_step

# cd checkpoints/$name
# python3 test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
# python3 evaluate_gpu.py
# python3 evaluateDistance.py --root_dir $root_dir
# cd ../../



# name="caf_s36_22k_1k__0012__200"
# root_dir="/home/abbas/AUT/Dataset/DenseUAV/google2"
# val_dir="/home/abbas/AUT/Dataset/DenseUAV/google2/validation"
# data_dir=$root_dir/train
# test_dir=$root_dir/test
# gpu_ids=0
# num_worker=8
# lr=0.0012
# batchsize=16
# sample_num=1
# block=1
# num_bottleneck=512
# backbone="caformer_s36.sail_in22k_ft_in1k" # resnet50 ViTS-224 senet
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
# num_epochs=200

# lr_step1=80
# lr_step2=150
# lr_step3=2000
# lr_step4=4000
# val_step=2

# python3 train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
#                 --block $block --lr $lr --num_worker $num_worker --head $head  --head_pool $head_pool \
#                 --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
#                 --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
#                 --num_epochs $num_epochs --lr_step1 $lr_step1 --lr_step2 $lr_step2 --lr_step3 $lr_step3 --lr_step4 $lr_step4 --val_dir $val_dir --val_step $val_step

# cd checkpoints/$name
# python3 test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
# python3 evaluate_gpu.py
# python3 evaluateDistance.py --root_dir $root_dir
# cd ../../


# name="mvit_b_21k__015__150_AD"
# root_dir="/home/abbas/AUT/Dataset/DenseUAV/DenseUAV"
# val_dir="/home/abbas/AUT/Dataset/DenseUAV/google2/validation"
# data_dir=$root_dir/train
# test_dir=$root_dir/test
# gpu_ids=0
# num_worker=8
# lr=0.015
# batchsize=16
# sample_num=1
# block=1
# num_bottleneck=512
# backbone="mvitv2_base_cls.fb_inw21k" # resnet50 ViTS-224 senet
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
# num_epochs=150
# lr_step1=60
# lr_step2=100
# lr_step3=200
# lr_step4=4000
# val_step=3

# python3 train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
#                 --block $block --lr $lr --num_worker $num_worker --head $head  --head_pool $head_pool \
#                 --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
#                 --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
#                 --num_epochs $num_epochs --lr_step1 $lr_step1 --lr_step2 $lr_step2 --lr_step3 $lr_step3 --lr_step4 $lr_step4 --val_dir $val_dir --val_step $val_step

# cd checkpoints/$name
# python3 test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
# python3 evaluate_gpu.py
# python3 evaluateDistance.py --root_dir $root_dir
# cd ../../


# name="mvit_b_21k__015__150_v2"
# root_dir="/home/abbas/AUT/Dataset/DenseUAV/google2"
# val_dir="/home/abbas/AUT/Dataset/DenseUAV/google2/validation"
# data_dir=$root_dir/train
# test_dir=$root_dir/test
# gpu_ids=0
# num_worker=8
# lr=0.015
# batchsize=16
# sample_num=1
# block=1
# num_bottleneck=512
# backbone="maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k" # resnet50 ViTS-224 senet
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
# num_epochs=150
# lr_step1=75
# lr_step2=120
# lr_step3=200
# lr_step4=4000
# val_step=3

# python3 train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
#                 --block $block --lr $lr --num_worker $num_worker --head $head  --head_pool $head_pool \
#                 --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
#                 --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
#                 --num_epochs $num_epochs --lr_step1 $lr_step1 --lr_step2 $lr_step2 --lr_step3 $lr_step3 --lr_step4 $lr_step4 --val_dir $val_dir --val_step $val_step

# cd checkpoints/$name
# python3 test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
# python3 evaluate_gpu.py
# python3 evaluateDistance.py --root_dir $root_dir
# cd ../../


name="eva_s_22k__00125__175__alaki"
root_dir="/home/abbas/AUT/Dataset/DenseUAV/google2"
val_dir="/home/abbas/AUT/Dataset/DenseUAV/google2/validation"
data_dir=$root_dir/train
test_dir=$root_dir/test
gpu_ids=0
num_worker=8
lr=0.00125
batchsize=16
sample_num=1
block=1
num_bottleneck=512
backbone="mvitv2_base_cls.fb_inw21k" # resnet50 ViTS-224 senet
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
num_epochs=175
lr_step1=80
lr_step2=130
lr_step3=200
lr_step4=4000
val_step=2

# python3 train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
#                 --block $block --lr $lr --num_worker $num_worker --head $head  --head_pool $head_pool \
#                 --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
#                 --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
#                 --num_epochs $num_epochs --lr_step1 $lr_step1 --lr_step2 $lr_step2 --lr_step3 $lr_step3 --lr_step4 $lr_step4 --val_dir $val_dir --val_step $val_step

cd checkpoints/$name
python3 test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
python3 evaluate_gpu.py
python3 evaluateDistance.py --root_dir $root_dir
cd ../../