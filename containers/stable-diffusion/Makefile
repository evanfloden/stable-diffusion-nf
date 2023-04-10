version ?= v0.1

all: build push

build:
	docker build --tag storytek/stable-diffusion-nf:${version} .
	
push:
	docker push storytek/stable-diffusion-nf:${version}
