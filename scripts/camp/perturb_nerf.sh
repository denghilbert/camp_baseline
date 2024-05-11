# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
# Script for training CamP on the 360 dataset.

export CUDA_VISIBLE_DEVICES=0
#export JAX_TRACEBACK_FILTERING=off
#export XLA_FLAGS=/usr/local/cuda-11.2

DATA_DIR=/home/yd428/pose/dataset/nerf_synthetic
CHECKPOINT_DIR=/home/yd428/blender_perturb

# Outdoor scenes.
#for SCENE in bicycle flowerbed gardenvase stump treehill
for SCENE in chair drums ficus hotdog lego materials mic ship
do
  echo python -m camp_zipnerf.train \
    --gin_configs=configs/zipnerf/blender.gin \
    --gin_configs=configs/camp/camera_optim_perturbed_blender.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
    --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}/${SCENE}'"
  python -m camp_zipnerf.train \
    --gin_configs=configs/zipnerf/blender.gin \
    --gin_configs=configs/camp/camera_optim_perturbed_blender.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
    --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}/${SCENE}'"
done
