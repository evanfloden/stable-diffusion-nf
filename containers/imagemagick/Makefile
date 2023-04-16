version ?= v0.1

all: build push

build:
	docker build --tag storytek/imagemagick:${version} .
	
push:
	docker push storytek/imagemagick:${version}
