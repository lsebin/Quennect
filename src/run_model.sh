today=$(date '+%Y-%m-%d')
now=${today}

seed=42
MODEL=All
run=pytorch

mkdir result/${now}
mkdir result/${now}/${run}
mkdir result/${now}/${run}/seed_${seed}

# 'RidgeRegression', 'LassoRegression', 'Perceptron', 'LinearSVM', 'RBFSVM', 'RandomForest', 'GradientBoostedTree', 'AdaBoost', 'NeuralNet'

nohup python -u src/modelComparison_${run}.py --model ${MODEL} > result/${now}/${run}/seed_${seed}/${MODEL}.log &