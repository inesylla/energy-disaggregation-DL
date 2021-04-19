#
# TESTING LAUNCHER
# Test each of the appliances and models analyzed in the project and described in settings.yaml
# See documentation describing each of the appliance analyzed 
# See documentation describing each of the model architectures evaluated

mkdir output-test

############### DISHWASHER

# Experiment 1
mkdir output-test/dishwasher
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance dishwasher --path output-test/dishwasher  --epochs 1 --disable-random > output-test/dishwasher/results-test.log

# Experiment 2
mkdir output-test/dishwasher-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance dishwasher-norm --path output-test/dishwasher-norm  --epochs 1 --disable-random > output-test/dishwasher-norm/results-test.log

# Experiment 3
mkdir output-test/dishwasher-norm-trainnorm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance dishwasher-norm --path output-test/dishwasher-norm-trainnorm  --epochs 1 --disable-random > output-test/dishwasher-norm-trainnorm/results-test.log

# Experiment 4
mkdir output-test/dishwasher-onlyregression
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance dishwasher-onlyregression --path output-test/dishwasher-onlyregression  --epochs 1 --disable-random > output-test/dishwasher-onlyregression/results-test.log

# Experiment 5
mkdir output-test/dishwasher-onlyregression-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance dishwasher-onlyregression-norm --path output-test/dishwasher-onlyregression-norm  --epochs 1 --disable-random > output-test/dishwasher-onlyregression-norm/results-test.log

# Experiment 5
mkdir output-test/dishwasher-onlyregression-norm-trainnorm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance dishwasher-onlyregression-norm --path output-test/dishwasher-onlyregression-norm-trainnorm  --epochs 1 --disable-random > output-test/dishwasher-onlyregression-norm-trainnorm/results-test.log

# Experiment 7
mkdir output-test/dishwasher-classattention
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance dishwasher-classattention --path output-test/dishwasher-classattention  --epochs 1 --disable-random > output-test/dishwasher-classattention/results-test.log

################ FRIDGE

# Experiment 1
mkdir output-test/fridge
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance fridge --path output-test/fridge  --epochs 1 --disable-random > output-test/fridge/results-test.log

# Experiment 2
mkdir output-test/fridge-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance fridge-norm --path output-test/fridge-norm  --epochs 1 --disable-random > output-test/fridge-norm/results-test.log

# Experiment 3
mkdir output-test/fridge-norm-trainnorm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance fridge-norm --path output-test/fridge-norm-trainnorm  --epochs 1 --disable-random > output-test/fridge-norm-trainnorm/results-test.log

# Experiment 4
mkdir output-test/fridge-onlyregression
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance fridge-onlyregression --path output-test/fridge-onlyregression  --epochs 1 --disable-random > output-test/fridge-onlyregression/results-test.log

# Experiment 5
mkdir output-test/fridge-onlyregression-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance fridge-onlyregression-norm --path output-test/fridge-onlyregression-norm  --epochs 1 --disable-random > output-test/fridge-onlyregression-norm/results-test.log

# Experiment 6
mkdir output-test/fridge-onlyregression-norm-trainnorm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance fridge-onlyregression-norm --path output-test/fridge-onlyregression-norm-trainnorm  --epochs 1 --disable-random > output-test/fridge-onlyregression-norm-trainnorm/results-test.log

# Experiment 7
mkdir output-test/fridge-classattention
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance fridge-classattention --path output-test/fridge-classattention  --epochs 1 --disable-random > output-test/fridge-classattention/results-test.log

################# MICROWAVE

# Experiment 1
mkdir output-test/microwave
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance microwave --path output-test/microwave  --epochs 1 --disable-random > output-test/microwave/results-test.log

# Experiment 2
mkdir output-test/microwave-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance microwave-norm --path output-test/microwave-norm  --epochs 1 --disable-random > output-test/microwave-norm/results-test.log

# Experiment 3
mkdir output-test/microwave-norm-trainnorm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance microwave-norm --path output-test/microwave-norm-trainnorm  --epochs 1 --disable-random > output-test/microwave-norm-trainnorm/results-test.log

# Experiment 4
mkdir output-test/microwave-onlyregression
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance microwave-onlyregression --path output-test/microwave-onlyregression  --epochs 1 --disable-random > output-test/microwave-onlyregression/results-test.log

# Experiment 5
mkdir output-test/microwave-onlyregression-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance microwave-onlyregression-norm --path output-test/microwave-onlyregression-norm  --epochs 1 --disable-random > output-test/microwave-onlyregression-norm/results-test.log

# Experiment 6
mkdir output-test/microwave-onlyregression-norm-trainnorm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance microwave-onlyregression-norm --path output-test/microwave-onlyregression-norm-trainnorm  --epochs 1 --disable-random > output-test/microwave-onlyregression-norm-trainnorm/results-test.log

# Experiment 7
mkdir output-test/microwave-classattention
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings settings.yaml --appliance microwave-classattention --path output-test/microwave-classattention  --epochs 1 --disable-random > output-test/microwave-classattention/results-test.log
