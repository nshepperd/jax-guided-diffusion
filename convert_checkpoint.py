import torch
import jaxtorch

with torch.no_grad():
    print('Loading state dict...')
    diffusion_state_dict = torch.load('256x256_diffusion_uncond.pt', map_location=torch.device('cpu'))
    jax_state_dict = {name : par.cpu().numpy() for (name, par) in diffusion_state_dict.items()}

with open('256x256_diffusion_uncond.cbor', 'wb') as fp:
    jaxtorch.cbor.dump(jax_state_dict, fp)