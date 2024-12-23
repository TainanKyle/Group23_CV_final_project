stamp=$(date "+%Y-%m-%d_%H-%M-%S")

log_dir="outputs/" # TODO
prompt="a bedroom"
scene_id="room_21"
img_path="data/scenes/room_tainan_1_1/style.jpg"
python scripts/train_texture.py --config config/template.yaml --stamp $stamp --log_dir $log_dir --prompt "$prompt" --scene_id "$scene_id" --img_path "$img_path"