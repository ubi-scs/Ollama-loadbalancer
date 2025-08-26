image_name := ollama_proxy

build:
	docker build . -t ${image_name}


# we need to use port mapping
run:
	docker stop ${image_name} || true
	docker rm ${image_name} || true
	docker run --name ${image_name} -p 8000:8000 ${image_name}
