stamp=$(date "+%Y-%m-%d_%H-%M-%S")

log_dir="outputs/" # TODO
prompt="a Bohemian style bedroom"
scene_id="room_1_1_preprocessed"
img_path="/home/gpl_homee/indoor_scene/SceneTex/clip_testdata/a_Bohemian_style_bedroom.jpg"
python scripts/train_texture.py --config config/template_learn.yaml --stamp $stamp --log_dir $log_dir --prompt "$prompt" --scene_id "$scene_id" --img_path "$img_path"