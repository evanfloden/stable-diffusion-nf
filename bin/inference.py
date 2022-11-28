#!/usr/bin/env python3
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import sys
from datetime import datetime
import string

prompt = str(sys.argv[1]) 
h = int(sys.argv[2])
w = int(sys.argv[3])

print("Prompt: ${prompt}")
print("Size: ${h} x ${w}")

model_id = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16) 
pipe = pipe.to("cuda")

clean_sample_str = prompt.translate(str.maketrans('', '', string.punctuation))
first_chars = clean_sample_str[0:29].replace(" ", "_")

dt_now = datetime.now()
time_stamp = datetime.timestamp(dt_now)
date_time = datetime.fromtimestamp(time_stamp)
str_date_time = date_time.strftime("%Y%m%d%H%M")

image_name = str_date_time + "_" + first_chars + ".png"

image = pipe(prompt, height=h, width=w).images[0]
image.save(image_name)
