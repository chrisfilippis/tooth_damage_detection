FROM tensorflow/tensorflow:1.13.2-gpu-py3

WORKDIR /var/project/data/annotator
COPY ./data/annotator /var/project/data/annotator

WORKDIR /var/project/src
COPY ./src /var/project/src

COPY requirements.txt /var/project/

RUN apt-get install -y git

RUN pip install Cython
RUN pip install --requirement /var/project/requirements.txt

# CMD python tooth.py --data_dir=/var/project/data/ --model_dir=/var/project/src/logs/ --init_with=coco
# sudo docker run -it --rm --gpus all -v /project/tooth_damage_detection/logs:/var/project/logs toot_damage:latest bash
# sudo docker build --tag=tooth_damage:latest .