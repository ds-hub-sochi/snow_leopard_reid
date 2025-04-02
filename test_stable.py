import torch
from diffusers import StableDiffusion3Pipeline


'''
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    prompt="a photorealistic cameratrap shot of an Amur forest cat runnig at the forest",
    negative_prompt="good quality",
    num_inference_steps=30,
    guidance_scale=10.0,
).images[0]
image.save("capybara.png")
'''

import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# prompt = "wildlife image of a wolverine captured by a camera trap in the nature"
prompt = "cameratrap photo of a European badger in the wildlife from the side"
image = pipe(
    prompt,
    # prompt_2='winter',
    negative_prompt="fox",
    # negative_prompt_2="short distance photo",
    guidance_scale=6.5,
    num_inference_steps=30,
    # max_sequence_length=512,
    # generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")
