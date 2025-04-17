python -m scripts.eval_policy_with_websocket --host localhost --port 8000 --plot \
    --modality_keys arm_joints \
    --steps 300 \
    --trajs 2 \
    --action_horizon 16 \
    --video_backend decord \
    --dataset_path demo_data/aloha_put_cube/ \
    --embodiment_tag new_embodiment \
    --data_config aloha_put_cube