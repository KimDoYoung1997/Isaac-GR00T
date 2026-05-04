from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="learner1119/ffw_sh5_rev1_ffw_sh5_rev1_20260503_trial345_27dof_merge",
  repo_type="dataset",  # 안 되면 "model"로 바꿔 보기
  local_dir="/workspace/dataset/ffw_sh5_rev1_ffw_sh5_rev1_20260503_trial345_27dof_merge",
  local_dir_use_symlinks=False,
)
