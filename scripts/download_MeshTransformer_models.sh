#!/bin/bash

cd MeshTransformer
mkdir -p models
bash scripts/download_models.sh
cd ..

cp scripts/inference_bodymesh.py MeshTransformer/metro/tools/