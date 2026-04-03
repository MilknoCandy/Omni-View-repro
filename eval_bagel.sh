# PYTHONPATH=$(pwd):$PYTHONPATH torchrun --nproc_per_node=4 \
#     --master_port=12345 \
#     eval/vlm/eval/vqa/evaluate_vsibench.py \
#     -m eval.vlm.eval.vqa.evaluate_vsibench \
#     --model-path ./pretrained_model/umms/BAGEL-7B-MoT \
#     --safetensor-path ema.safetensors \
#     --dataset vsibench
torchrun --nproc_per_node=4 --master_port=12345 -m eval.vlm.eval.vqa.evaluate_vsibench --model-path ./pretrained_model/umms/BAGEL-7B-MoT/ --safetensor-path model.safetensors --dataset vsibench