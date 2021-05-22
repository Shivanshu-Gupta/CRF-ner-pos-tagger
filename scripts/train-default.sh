mkdir -p logs/default
for ds in ner pos
do
    for model in simple crf
    do
        echo "python driver.py -o --dataset $ds --model $model --default --gpu_idx=0 --test --silent 2>&1 | tee logs/default/${ds}-${model}.txt"
        python driver.py -o --dataset $ds --model $model --default --gpu_idx=0 --test --silent 2>&1 | tee logs/default/${ds}-${model}.txt
    done
done