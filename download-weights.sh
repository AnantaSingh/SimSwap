#!/bin/sh
wget -P ./arcface_model https://github.com/neuralchen/SimSwap/releases/download/1.0/arcface_checkpoint.tar
wget https://github.com/neuralchen/SimSwap/releases/download/1.0/checkpoints.zip
unzip ./checkpoints.zip  -d ./checkpoints
rm checkpoints.zip
wget https://github.com/deepinsight/insightface/releases/download/v0.2/antelope.zip
mkdir -p insightface_func/models
unzip ./antelope.zip -d ./insightface_func/models/
rm antelope.zip
