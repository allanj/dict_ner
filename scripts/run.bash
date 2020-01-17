


device=cuda:0
dataset=conll2003
embedding_file=data/glove.6B.100d.txt
extraction=1
target_type=ORG
dict_ratio=0.1
inst_ratio=0.01
use_unlabeled_insts=0

python3 cvt_dict.py --device ${device} --dataset ${dataset} --embedding_file ${embedding_file} --extraction ${extraction} \
                    --target_type ${target_type} --dict_ratio 0.1 --inst_ratio 0.01 \
                    --use_unlabeled_insts 1 > logs/dict_0.1_insts_0.01_unlabeled_1.log 2>&1

python3 cvt_dict.py --device ${device} --dataset ${dataset} --embedding_file ${embedding_file} --extraction ${extraction} \
                    --target_type ${target_type} --dict_ratio 0.1 --inst_ratio 0.01 \
                    --use_unlabeled_insts 0 > logs/dict_0.1_insts_0.01_unlabeled_0.log 2>&1

python3 cvt_dict.py --device ${device} --dataset ${dataset} --embedding_file ${embedding_file} --extraction ${extraction} \
                    --target_type ${target_type} --dict_ratio 0 --inst_ratio 0.01 \
                    --use_unlabeled_insts 0 > logs/dict_0_insts_0.01_unlabeled_0.log 2>&1

