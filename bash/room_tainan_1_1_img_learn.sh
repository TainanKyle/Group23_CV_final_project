stamp=$(date "+%Y-%m-%d_%H-%M-%S")

log_dir="outputs/" # TODO
prompt="a Japanese style bedroom"
scene_id="room_tainan_1_1"
img_path="/home/gpl_homee/indoor_scene/SceneTex/data/scenes/room_tainan_1_1/japanese.jpg"
python scripts/train_texture.py --config config/template_learn.yaml --stamp $stamp --log_dir $log_dir --prompt "$prompt" --scene_id "$scene_id" --img_path "$img_path"