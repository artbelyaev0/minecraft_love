# This will resume training if data/training_checkpoint is not empty. If you want to start from scratch, delete the directory.

accelerate launch --num_processes 4 --mixed_precision fp16 --multi_gpu steve1/training/train.py \
--in_model data/weights/vpt/2x.model \
--in_weights data/weights/vpt/rl-from-foundation-2x.weights \
--out_weights data/weights/steve1/trained_with_script.weights \
--trunc_t 64 \
--T 640 \
--batch_size 6 \
--gradient_accumulation_steps 2 \
--num_workers 4 \
--weight_decay 0.039428 \
--n_frames 8_400_000 \
--learning_rate 4e-5 \
--warmup_frames 525_000 \
--p_uncond 0.1 \
--min_btwn_goals 15 \
--max_btwn_goals 200 \
--checkpoint_dir data/training_checkpoint \
--val_freq 1000 \
--val_freq_begin 100 \
--val_freq_switch_steps 500 \
--val_every_nth 10 \
--save_each_val False \
--sampling neurips \
--sampling_dir data/samplings/ \
--snapshot_every_n_frames 1_500_000