pip install dm-haiku
git clone https://github.com/kingoflolz/CLIP_JAX

# Download checkpoint
test -a 256x256_diffusion_uncond.pt || curl -OL 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'
test -a 256x256_diffusion_uncond.cbor || python convert_checkpoint.py