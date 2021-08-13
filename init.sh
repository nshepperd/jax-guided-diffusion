pip install dm-haiku
git clone https://github.com/kingoflolz/CLIP_JAX

# Download checkpoint
curl -OL 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'
python convert_checkpoint.py