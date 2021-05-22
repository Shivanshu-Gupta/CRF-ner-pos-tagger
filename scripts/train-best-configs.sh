mkdir -p logs/best
for ds in ner pos
do
    for model in simple crf
    do
        echo "python driver.py -o --dataset $ds --model $model --enc 0 --gpu_idx=0 --test --silent 2>&1 | tee logs/best/${ds}-${model}.txt"
        python driver.py -o --dataset $ds --model $model --enc 0 --gpu_idx=0 --test --silent 2>&1 | tee logs/best/${ds}-${model}.txt
        echo "python driver.py -o --dataset $ds --model $model --emb --enc 0 --gpu_idx=0 --test --silent 2>&1 | tee logs/best/${ds}-${model}-emb.txt"
        python driver.py -o --dataset $ds --model $model --emb --enc 0 --gpu_idx=0 --test --silent 2>&1 | tee logs/best/${ds}-${model}-emb.txt
        for enc in 1
        do
            echo "python driver.py -o --dataset $ds --model $model --enc ${enc} --gpu_idx=0 --test --silent 2>&1 | tee logs/best/${ds}-${model}-enc${enc}.txt"
            python driver.py -o --dataset $ds --model $model --enc ${enc} --gpu_idx=0 --test --silent 2>&1 | tee logs/best/${ds}-${model}-enc${enc}.txt
            echo "python driver.py -o --dataset $ds --model $model --emb --enc ${enc} --gpu_idx=0 --test --silent 2>&1 | tee logs/best/${ds}-${model}-emb-enc${enc}.txt"
            python driver.py -o --dataset $ds --model $model --emb --enc ${enc} --gpu_idx=0 --test --silent 2>&1 | tee logs/best/${ds}-${model}-emb-enc${enc}.txt
        done
    done
done