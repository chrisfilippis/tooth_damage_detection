#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

git pull

docker stop tooth_maskrcnn

docker rm tooth_maskrcnn

docker build --tag=tooth_maskrcnn:latest .

docker run -it --name tooth_maskrcnn --gpus all -v ~/project/logs:/var/project/logs tooth_maskrcnn:latest bash