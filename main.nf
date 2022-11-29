nextflow.enable.dsl=2

params.prompt = "A portrait of Barak Obama wearing a bow tie in a scenic environment by Adi Granov"
params.height = 760
params.width = 760
params.images = 5
params.outdir = 'results'

process INFERENCE {

    secret 'HUGGINGFACE_HUB_TOKEN'
    container 'evanfloden/stable-diffusion-nf:v0.1'
    publishDir "$params.outdir"

    input:
    tuple val(prompt), val(height), val(width), val(seed)

    output:
    path("*.png")

    script:
    """
    #!/usr/bin/env python3

    # Import
    import torch
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
    import string
    from huggingface_hub import HfFolder

    # Save Huggingface API token 
    HfFolder.save_token("\$HUGGINGFACE_HUB_TOKEN") 

    # Stable Diffusion parameters
    model_id   = "stabilityai/stable-diffusion-2"
    scheduler  = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe       = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16, safety_checker=None) 
    pipe       = pipe.to("cuda")

    # Define image filename
    prompt      = "$prompt"
    prompt_str  = prompt.translate(str.maketrans('', '', string.punctuation))
    first_chars = prompt_str[0:29].replace(" ", "_")
    seed        = "$seed"
    image_name  = seed + "_" + first_chars + ".png"

    # Generate image
    image       = pipe(prompt, height=$height, width=$width).images[0]
    image.save(image_name)
    """
}

workflow {

    // Create a channel containing N random seeds
    Channel
        .of(1..100000)
        .randomSample(params.images.toInteger())
        .set{images_ch}

    // Combine prompt with N random seeds
    Channel
        .of([params.prompt,params.height,params.width])
        .combine(images_ch)
        .set{stable_diffusion_ch}

    INFERENCE(stable_diffusion_ch)

}
