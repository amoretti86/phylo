#!/bin/bash
# Automate multiple experiments on GCP
#for k in {1024,1024,1024,2048,2048,2048}
for k in {32,32,32,64,64,64}
do
echo "Running Primates K =" $k
python runner.py --dataset=primate_data --n_particles=$k --batch_size=256 --learning_rate=0.001 --num_epoch=100 --nested=true
#python runner.py --dataset=DS1 --n_particles=$k --batch_size=512 --learning_rate=0.001 --num_epoch=100 --M=$k --nested=true
#python runner.py --dataset=DS2 --n_particles=$k --batch_size=512 --learning_rate=0.001 --num_epoch=100 --M=$k --nested=true
#python runner.py --dataset=DS3 --n_particles=$k --batch_size=512 --learning_rate=0.001 --num_epoch=100 --M=$k --nested=true
#python runner.py --dataset=DS4 --n_particles=$k --batch_size=512 --learning_rate=0.001 --num_epoch=100 --M=$k --nested=true
done
