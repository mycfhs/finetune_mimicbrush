    
import os   

# 设置 HTTP 代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:3333"
# 设置 HTTPS 代理
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:3333"

from huggingface_hub import snapshot_download
snapshot_download(repo_id="xichenhku/cleansd", local_dir="./cleansd")
print('=== Pretrained SD weights downloaded ===')
# snapshot_download(repo_id="xichenhku/MimicBrush", local_dir="./MimicBrush")
# print('=== MimicBrush weights downloaded ===')