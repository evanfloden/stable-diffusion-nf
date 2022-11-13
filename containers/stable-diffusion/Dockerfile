FROM continuumio/miniconda3:4.10.3

COPY conda.yml .
RUN \
   pip install transformers diffusers lpips \
&& conda env update -n root -f conda.yml \
&& conda clean -a

RUN apt-get install -y procps