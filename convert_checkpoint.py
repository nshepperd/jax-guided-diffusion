import sys
import torch
import jaxtorch

model = sys.argv[1]

with torch.no_grad():
    print('Loading state dict...')
    diffusion_state_dict = torch.load(f'{model}.pt', map_location=torch.device('cpu'))
    jax_state_dict = {name : par.cpu().numpy() for (name, par) in diffusion_state_dict.items()}

with open(f'{model}.cbor', 'wb') as fp:
    jaxtorch.cbor.dump(jax_state_dict, fp)