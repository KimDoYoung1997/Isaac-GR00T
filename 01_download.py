from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="learner1119/merge_ffw_sh5_rev1_20260504_27dof_pick_up_a_red_cylinder_and_place_it_on_the_basket",
  repo_type="dataset",  # 안 되면 "model"로 바꿔 보기
  local_dir="/home/work/ku_doyoung/Isaac-GR00T/demo_data/merge_ffw_sh5_rev1_20260504_27dof_pick_up_a_red_cylinder_and_place_it_on_the_basket",
  local_dir_use_symlinks=False,
)

