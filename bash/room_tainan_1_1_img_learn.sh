stamp=$(date "+%Y-%m-%d_%H-%M-%S")

log_dir="outputs/" # TODO
prompt="a bedroom"
scene_id="room_tainan_1_1"
img_path="/home/ado/storage/CV_final_project/data/scenes/modern.jpg"
python scripts/train_texture.py --config config/template_learn.yaml --stamp $stamp --log_dir $log_dir --prompt "$prompt" --scene_id "$scene_id" --img_path "$img_path"