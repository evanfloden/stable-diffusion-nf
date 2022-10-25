
process INFERENCE {

    secret 'HUGGINGFACE_HUB_TOKEN'
    container 'evanfloden/stable-diffusion-nf'

    script:
    """
    python3 -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('\$HUGGINGFACE_HUB_TOKEN')"
    inference.py
    """
}

workflow {
    INFERENCE()
}
