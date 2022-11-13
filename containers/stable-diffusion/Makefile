version ?= v0.1

all: build push

build:
	docker build --tag evanfloden/stable-diffusion-nf:${version} .
	
push:
	docker push evanfloden/stable-diffusion-nf:${version}