

name="vit_s_p16_21k_1k__01__150__75_120"
root_dir="/home/abbas/AUT/Dataset/DenseUAV/google2"
val_dir="/home/abbas/AUT/Dataset/DenseUAV/google2/validation"
data_dir=$root_dir/train
test_dir=$root_dir/test
gpu_ids=0
num_worker=8
lr=0.01
batchsize=16
sample_num=1
block=1
num_bottleneck=512
backbone="vit_small_patch16_224.augreg_in21k_ft_in1k" # resnet50 ViTS-224 senet
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

lr_step1=75
lr_step2=120
lr_step3=2000
lr_step4=4000
val_step=2

python3 train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
                --block $block --lr $lr --num_worker $num_worker --head $head  --head_pool $head_pool \
                --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
                --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
                --num_epochs $num_epochs --lr_step1 $lr_step1 --lr_step2 $lr_step2 --lr_step3 $lr_step3 --lr_step4 $lr_step4 --val_dir $val_dir --val_step $val_step

cd checkpoints/$name
python3 test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
python3 evaluate_gpu.py
python3 evaluateDistance.py --root_dir $root_dir
cd ../../



name="vit_s_r26_s32__21k_1k__01__180__60_120"
root_dir="/home/abbas/AUT/Dataset/DenseUAV/google2"
val_dir="/home/abbas/AUT/Dataset/DenseUAV/google2/validation"
data_dir=$root_dir/train
test_dir=$root_dir/test
gpu_ids=0
num_worker=8
lr=0.01
batchsize=16
sample_num=1
block=1
num_bottleneck=512
backbone="vit_small_r26_s32_224.augreg_in21k_ft_in1k" # resnet50 ViTS-224 senet
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
num_epochs=180

lr_step1=60
lr_step2=120
lr_step3=2000
lr_step4=4000
val_step=2

python3 train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
                --block $block --lr $lr --num_worker $num_worker --head $head  --head_pool $head_pool \
                --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
                --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
                --num_epochs $num_epochs --lr_step1 $lr_step1 --lr_step2 $lr_step2 --lr_step3 $lr_step3 --lr_step4 $lr_step4 --val_dir $val_dir --val_step $val_step

cd checkpoints/$name
python3 test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
python3 evaluate_gpu.py
python3 evaluateDistance.py --root_dir $root_dir
cd ../../


name="repvit_m2_3_450e_1k__0005__180__65_130"
root_dir="/home/abbas/AUT/Dataset/DenseUAV/google2"
val_dir="/home/abbas/AUT/Dataset/DenseUAV/google2/validation"
data_dir=$root_dir/train
test_dir=$root_dir/test
gpu_ids=0
num_worker=8
lr=0.0005
batchsize=16
sample_num=1
block=1
num_bottleneck=512
backbone="repvit_m2_3.dist_450e_in1k" # resnet50 ViTS-224 senet
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
num_epochs=180

lr_step1=65
lr_step2=130
lr_step3=2000
lr_step4=4000
val_step=2

python3 train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
                --block $block --lr $lr --num_worker $num_worker --head $head  --head_pool $head_pool \
                --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
                --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
                --num_epochs $num_epochs --lr_step1 $lr_step1 --lr_step2 $lr_step2 --lr_step3 $lr_step3 --lr_step4 $lr_step4 --val_dir $val_dir --val_step $val_step

cd checkpoints/$name
python3 test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
python3 evaluate_gpu.py
python3 evaluateDistance.py --root_dir $root_dir
cd ../../



name="xcit_s_12_p8_1k__008__250__80_140_200"
root_dir="/home/abbas/AUT/Dataset/DenseUAV/google2"
val_dir="/home/abbas/AUT/Dataset/DenseUAV/google2/validation"
data_dir=$root_dir/train
test_dir=$root_dir/test
gpu_ids=0
num_worker=8
lr=0.008
batchsize=16
sample_num=1
block=1
num_bottleneck=512
backbone="xcit_small_12_p8_224.fb_dist_in1k" # resnet50 ViTS-224 senet
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
num_epochs=250

lr_step1=80
lr_step2=140
lr_step3=200
lr_step4=4000
val_step=2

python3 train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
                --block $block --lr $lr --num_worker $num_worker --head $head  --head_pool $head_pool \
                --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
                --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
                --num_epochs $num_epochs --lr_step1 $lr_step1 --lr_step2 $lr_step2 --lr_step3 $lr_step3 --lr_step4 $lr_step4 --val_dir $val_dir --val_step $val_step

cd checkpoints/$name
python3 test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
python3 evaluate_gpu.py
python3 evaluateDistance.py --root_dir $root_dir
cd ../../



name="eva2_b_p14__22k_1k__0075__200__85_150"
root_dir="/home/abbas/AUT/Dataset/DenseUAV/google2"
val_dir="/home/abbas/AUT/Dataset/DenseUAV/google2/validation"
data_dir=$root_dir/train
test_dir=$root_dir/test
gpu_ids=0
num_worker=8
lr=0.0075
batchsize=16
sample_num=1
block=1
num_bottleneck=512
backbone="eva02_base_patch14_224.mim_in22k" # resnet50 ViTS-224 senet
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
num_epochs=200

lr_step1=85
lr_step2=150
lr_step3=2000
lr_step4=4000
val_step=2

python3 train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
                --block $block --lr $lr --num_worker $num_worker --head $head  --head_pool $head_pool \
                --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
                --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
                --num_epochs $num_epochs --lr_step1 $lr_step1 --lr_step2 $lr_step2 --lr_step3 $lr_step3 --lr_step4 $lr_step4 --val_dir $val_dir --val_step $val_step

cd checkpoints/$name
python3 test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
python3 evaluate_gpu.py
python3 evaluateDistance.py --root_dir $root_dir
cd ../../


name="mvit_b_21k__014__250__65_120_200"
root_dir="/home/abbas/AUT/Dataset/DenseUAV/google2"
val_dir="/home/abbas/AUT/Dataset/DenseUAV/google2/validation"
data_dir=$root_dir/train
test_dir=$root_dir/test
gpu_ids=0
num_worker=8
lr=0.014
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
num_epochs=250

lr_step1=65
lr_step2=120
lr_step3=200
lr_step4=4000
val_step=2

python3 train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
                --block $block --lr $lr --num_worker $num_worker --head $head  --head_pool $head_pool \
                --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
                --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
                --num_epochs $num_epochs --lr_step1 $lr_step1 --lr_step2 $lr_step2 --lr_step3 $lr_step3 --lr_step4 $lr_step4 --val_dir $val_dir --val_step $val_step

cd checkpoints/$name
python3 test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
python3 evaluate_gpu.py
python3 evaluateDistance.py --root_dir $root_dir
cd ../../