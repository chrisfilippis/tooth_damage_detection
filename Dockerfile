FROM tensorflow/tensorflow:1.13.1-gpu-py3

WORKDIR /var/project/data/annotator
COPY ./data/annotator /var/project/data/annotator

WORKDIR /var/project/src
COPY ./src /var/project/src

COPY requirements.txt /var/project/

WORKDIR /var/project
RUN apt-get install -y git

RUN pip install Cython
RUN pip install --requirement requirements.txt

CMD python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"

# CMD python tooth.py --data_dir=/var/project/data/ --model_dir=/var/project/src/logs/ --init_with=coco
# sudo docker run -it --rm --gpus all -v /project/tooth_damage_detection/logs:/var/project/logs tooth_damage:latest bash
# sudo docker build --tag=tooth_damage:latest .