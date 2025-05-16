from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="fkand/vae-gan-test",
    local_dir="/home/ubuntu/vae-gan-mark/checkpoints_vaegan_wandb",  
    token="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"  
)