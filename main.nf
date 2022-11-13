
process INFERENCE {

    secret 'HUGGINGFACE_HUB_TOKEN'
    container 'evanfloden/stable-diffusion-nf:v0.1'

    script:
    """
    python3 -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('\$HUGGINGFACE_HUB_TOKEN')"
    inference.py
    """
}

workflow {
    INFERENCE()
}
