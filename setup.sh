#!/bin/sh

#### set up logging file 
if [ -d "./logs/" ]; then
	echo "Logging directory exists"	
else
	mkdir ./logs/
fi
touch ./logs/setup.log

### set up directories for data
if [ -d "./data/" ]; then
	echo "Data directory exists"	
else
    mkdir ./data/
fi

if [ -d "./data/synthetic" ]; then
	echo "Synthetic data directory exists"	
else
    mkdir ./data/synthetic/
fi

if [ -d "./data/real" ]; then
	echo "Real data directory exists"	
else
    mkdir ./data/synthetic/
fi

(
#### generate bivariate gaussian synthetic data 
echo ~~~~~~~~ Generating bivariate gaussian data ~~~~~~~~~~~~~~~
python src/models/bivariate_gaussian/data_synth.py --path-to-data="data/synthetic/"/
python src/models/bivariate_gaussian/toy_data_synth.py --path-to-data="data/synthetic/"

#### generate logistic regression synthetic data
echo ~~~~~~~~ Generating logistic regression data ~~~~~~~~~~~~~
python src/models/logistic_regression/toy_data_synth_balanced.py --path-to-data="data/synthetic/"
python src/models/logistic_regression/data_synth_balanced.py --path-to-data="data/synthetic/"
python src/models/logistic_regression/toy_data_synth_imbalanced.py --path-to-data="data/synthetic/"

####download real data
echo ~~~~ Downloading real datasets ~~~~~~~~~~~~~~~~
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2 -O data/real/covtype.libsvm.binary.bz2
bzip2 -d data/real/covtype.libsvm.binary.bz2
python src/models/logistic_regression/covtype.py --path-to-data="data/real/"
rm data/real/covtype.libsvm.binary

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv -O data/real/CASP.csv
python src/models/linear_regression/casp.py --path-to-data="data/real/"

#### save stderr and stdout 
) 2>&1 | tee ./logs/setup.log

