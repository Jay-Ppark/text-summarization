python train.py  --gradient_clip_val 1.0 --amp_level O3 --precision 16 --max_epochs 5 --train_file data/my_train.tsv --default_root_dir logs  --batch_size 1 --num_workers 4 --max_len 1024 --gpus 1
