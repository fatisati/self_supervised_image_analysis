#!/bin/bash

mkdir -p data
cd data

if [ ! -f infimnist.tar.gz ]; then 
    wget http://leon.bottou.org/_media/projects/infimnist.tar.gz
else
    rm -rf infimnist
fi

tar xzvf infimnist.tar.gz
cd infimnist
make
./infimnist lab 0 9999 > test10k-labels
./infimnist pat 0 9999 > test10k-patterns