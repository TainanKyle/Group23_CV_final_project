stamp=$(date "+%Y-%m-%d_%H-%M-%S")

log_dir="outputs/" # TODO
prompt="a Scandinavian style office room"
scene_id="room_27"
python scripts/train_texture.py --config config/template.yaml --stamp $stamp --log_dir $log_dir --prompt "$prompt" --scene_id "$scene_id"