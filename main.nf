nextflow.enable.dsl=2

params.prompt = "3D DNA exiting cell"
params.height = 512
params.width = 512

process INFERENCE {

    secret 'HUGGINGFACE_HUB_TOKEN'
    container 'evanfloden/stable-diffusion-nf:v0.1'

    input:
    tuple val(prompt), val(height), val(width)

    script:
    """
    python3 -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('\$HUGGINGFACE_HUB_TOKEN')"
    inference.py "$prompt" "$height" "$width"
    """
}

workflow {

    INFERENCE(Channel.of([params.prompt,params.height,params.width]))

}
