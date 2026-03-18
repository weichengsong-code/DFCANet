EXPID=Z
NUM_GPU=1

python test.py \
  --config '/bvg/code/MultiModal-DeepFake/configs/test.yaml' \
  --output_dir '/bvg/code/MultiModal-DeepFake/results' \
  --launcher pytorch \
  --rank 0 \
  --log_num ${EXPID} \
  --token_momentum \
  --world_size ${NUM_GPU} \
  --test_epoch best \
  --vis_heatmap \
  --vis_num 100
