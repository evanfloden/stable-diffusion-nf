nextflow.enable.dsl=2

params.prompt = "Picture of Darth Vader eating broccoli pizza"
params.height = 760
params.width = 760
params.outdir = 'results'

process INFERENCE {

    secret 'HUGGINGFACE_HUB_TOKEN'
    container 'evanfloden/stable-diffusion-nf:v0.1'
    publishDir "$params.outdir"

    input:
    tuple val(prompt), val(height), val(width)

    output:
    path("image.png")

    script:
    """
    python3 -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('\$HUGGINGFACE_HUB_TOKEN')"
    inference.py "$prompt" "$height" "$width"
    """
}

workflow {

    INFERENCE(Channel.of([params.prompt,params.height,params.width]))

}
