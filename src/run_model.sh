today=$(date '+%Y-%m-%d')
now=${today}

seed=42
MODEL=All

mkdir result/${now}
mkdir result/${now}/seed_${seed}

# 'RidgeRegression', 'LassoRegression', 'Perceptron', 'LinearSVM', 'RBFSVM', 'RandomForest', 'GradientBoostedTree', 'AdaBoost', 'NeuralNet'

nohup python -u src/modelComparison_undersample.py --model ${MODEL} > result/${now}/seed_${seed}/${MODEL}.log &