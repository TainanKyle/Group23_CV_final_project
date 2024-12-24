stamp=$(date "+%Y-%m-%d_%H-%M-%S")

log_dir="outputs/" # TODO
prompt="a bedroom"
scene_id="room_1_1_preprocessed"
img_path="/home/gpl_homee/indoor_scene/SceneTex/data/scenes/room_1_1_preprocessed/modern.jpg"
python scripts/train_texture.py --config config/template_learn.yaml --stamp $stamp --log_dir $log_dir --prompt "$prompt" --scene_id "$scene_id" --img_path "$img_path"