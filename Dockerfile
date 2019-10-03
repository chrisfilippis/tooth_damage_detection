FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /var/project/data/annotator
COPY ./data/annotator /var/project/data/annotator

WORKDIR /var/project/src
COPY ./src /var/project/src

COPY requirements.txt /var/project/

RUN apt-get install -y git

RUN pip install Cython
RUN pip install --requirement /var/project/requirements.txt
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# CMD python tooth.py --data_dir=/var/project/data/ --model_dir=/var/project/src/logs/ --init_with=coco