today=$(date '+%Y-%m-%d')
now=${today}

seed=42
MODEL=All
hidden=256
hidden1=128
hidden2=64
lr=0.001
run=pytorch
epoch=50

mkdir result/${now}
mkdir result/${now}/${run}
mkdir result/${now}/${run}/seed_${seed}

# 'RidgeRegression', 'LassoRegression', 'Perceptron', 'LinearSVM', 'RBFSVM', 'RandomForest', 'GradientBoostedTree', 'AdaBoost', 'NeuralNet'

#nohup python -u src/modelComparison_${run}.py --model ${MODEL} > result/${now}/${run}/seed_${seed}/${MODEL}.log &

for hidden in 256
do
    for hidden1 in 192
    do 
        for hidden2 in 128
        do
            for epoch in 50 75 100
            do
                nohup python -u src/modelComparison_${run}.py --hidden ${hidden} --hidden1 ${hidden1} --hidden2 ${hidden2} --lr ${lr} --epoch ${epoch} > result/${now}/${run}/seed_${seed}/${run}_${hidden}.${hidden1}.${hidden2}_${lr}_epoch${epoch}-4_b200.log &
            done
        done
    done
done