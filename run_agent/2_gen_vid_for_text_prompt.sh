# Change custom_text_prompt to whatever text prompt you want to generate a video for
xvfb-run --auto-servernum python steve1/run_agent/run_agent.py \
--in_model data/weights/vpt/2x.model \
--in_weights data/training_checkpoint/pytorch_model.bin \
--prior_weights data/weights/steve1/steve1_prior.pt \
--text_cond_scale 6.0 \
--visual_cond_scale 7.0 \
--gameplay_length 2880 \
--save_dirpath data/generated_videos/our_pipeline \
--custom_text_prompt "chop down the tree, gather wood, pick up wood, chop it down, break tree"