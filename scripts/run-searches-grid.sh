mkdir -p logs/grid
for ds in ner pos
do
    for model in simple crf
    do
        echo "python search.py --dataset $ds --model $model --enc 0 --gpu_idx=0 --gpus_per_trial=1 --num_samples=1 2>&1 | tee logs/grid/${ds}-${model}.txt"
        python search.py --dataset $ds --model $model --enc 0 --gpu_idx=0 --gpus_per_trial=1 --num_samples=1 2>&1 | tee logs/grid/${ds}-${model}.txt
        echo "python search.py --dataset $ds --model $model --emb --enc 0 --gpu_idx=0 --gpus_per_trial=1 --num_samples=1 2>&1 | tee logs/grid/${ds}-${model}-emb.txt"
        python search.py --dataset $ds --model $model --emb --enc 0 --gpu_idx=0 --gpus_per_trial=1 --num_samples=1 2>&1 | tee logs/grid/${ds}-${model}-emb.txt
        for enc in 1
        do
            echo "python search.py --dataset $ds --model $model --enc ${enc} --gpu_idx=0 --gpus_per_trial=1 --num_samples=1 2>&1 | tee logs/grid/${ds}-${model}-enc${enc}.txt"
            python search.py --dataset $ds --model $model --enc ${enc} --gpu_idx=0 --gpus_per_trial=1 --num_samples=1 2>&1 | tee logs/grid/${ds}-${model}-enc${enc}.txt
            echo "python search.py --dataset $ds --model $model --emb --enc ${enc} --gpu_idx=0 --gpus_per_trial=1 --num_samples=1 2>&1 | tee logs/grid/${ds}-${model}-emb-enc${enc}.txt"
            python search.py --dataset $ds --model $model --emb --enc ${enc} --gpu_idx=0 --gpus_per_trial=1 --num_samples=1 2>&1 | tee logs/grid/${ds}-${model}-emb-enc${enc}.txt
        done
    done
done