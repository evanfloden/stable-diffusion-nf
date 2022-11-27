#!/usr/bin/env python3
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import sys

prompt = str(sys.argv[1]) 
h = int(sys.argv[2])
w = int(sys.argv[3])

print("Prompt: ${prompt}")
print("Size: ${h} x ${w}")

model_id = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16) 
pipe = pipe.to("cuda")

image = pipe(prompt, height=h, width=w).images[0]
image.save("image.png")