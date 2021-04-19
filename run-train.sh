#
# TRAINING LAUNCHER
# Train each of the appliances and models analyzed in the project and described in settings.yaml
# See documentation describing each of the appliance analyzed 
# See documentation describing each of the model architectures evaluated

mkdir output-train

################ DISHWASHER 

# Experiment 1
mkdir output-train/dishwasher
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance dishwasher --path output-train/dishwasher --train --epochs 5 --disable-random > output-train/dishwasher/results-train.log

# Experiment 2
mkdir output-train/dishwasher-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance dishwasher-norm --path output-train/dishwasher-norm --train --epochs 5 --disable-random > output-train/dishwasher-norm/results-train.log

# Experiment 4
mkdir output-train/dishwasher-onlyregression
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance dishwasher-onlyregression --path output-train/dishwasher-onlyregression --train --epochs 5 --disable-random > output-train/dishwasher-onlyregression/results-train.log

# Experiment 5
mkdir output-train/dishwasher-onlyregression-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance dishwasher-onlyregression-norm --path output-train/dishwasher-onlyregression-norm --train --epochs 5 --disable-random > output-train/dishwasher-onlyregression-norm/results-train.log

# Experiment 7
mkdir output-train/dishwasher-classattention
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance dishwasher-classattention --path output-train/dishwasher-classattention --train --epochs 5 --disable-random > output-train/dishwasher-classattention/results-train.log

################ FRIDGE 

# Experiment 1
mkdir output-train/fridge
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance fridge --path output-train/fridge --train --epochs 5 --disable-random > output-train/fridge/results-train.log

# Experiment 2
mkdir output-train/fridge-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance fridge-norm --path output-train/fridge-norm --train --epochs 5 --disable-random > output-train/fridge-norm/results-train.log

# Experiment 4
mkdir output-train/fridge-onlyregression
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance fridge-onlyregression --path output-train/fridge-onlyregression --train --epochs 5 --disable-random > output-train/fridge-onlyregression/results-train.log

# Experiment 5
mkdir output-train/fridge-onlyregression-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance fridge-onlyregression-norm --path output-train/fridge-onlyregression-norm --train --epochs 5 --disable-random > output-train/fridge-onlyregression-norm/results-train.log

# Experiment 7
mkdir output-train/fridge-classattention
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance fridge-classattention --path output-train/fridge-classattention --train --epochs 5 --disable-random > output-train/fridge-classattention/results-train.log

################ MICROWAVE

# Experiment 1
mkdir output-train/microwave
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance microwave --path output-train/microwave --train --epochs 5 --disable-random > output-train/microwave/results-train.log

# Experiment 2
mkdir output-train/microwave-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance microwave-norm --path output-train/microwave-norm --train --epochs 5 --disable-random > output-train/microwave-norm/results-train.log

# Experiment 4
mkdir output-train/microwave-onlyregression
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance microwave-onlyregression --path output-train/microwave-onlyregression --train --epochs 5 --disable-random > output-train/microwave-onlyregression/results-train.log

# Experiment 5
mkdir output-train/microwave-onlyregression-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance microwave-onlyregression-norm --path output-train/microwave-onlyregression-norm --train --epochs 5 --disable-random > output-train/microwave-onlyregression-norm/results-train.log

# Experiment 7
mkdir output-train/microwave-classattention
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance microwave-classattention --path output-train/microwave-classattention --train --epochs 5 --disable-random > output-train/microwave-classattention/results-train.log

