python train.py --gradient_clip_val 1.0 \
                --train_file data/my_train.tsv \
                --test_file data/my_test.tsv \
                --max_epochs 5 \
                --default_root_dir logs \
                --batch_size 10 \
                --num_workers 4 \
                --gpus 2