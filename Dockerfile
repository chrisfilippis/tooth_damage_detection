FROM tensorflow/tensorflow:1.13.1-gpu-py3

WORKDIR /var/project/data/annotator
COPY ./data/annotator /var/project/data/annotator

WORKDIR /var/project/src
COPY ./src /var/project/src

WORKDIR /var/project/data/output/training
WORKDIR /var/project/data/output/validation
WORKDIR /var/project/data/output/unknown

COPY requirements.txt /var/project/

WORKDIR /var/project

RUN apt-get install -y git

RUN pip install Cython
RUN pip install --requirement requirements.txt
RUN apt-get install -y libsm6 libxext6 libxrender-dev

CMD python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"
# CMD python src/tooth.py --data_dir=/var/project/data/ --model_dir=/var/project/logs/ --init_with=coco > logs/output.txt

# sudo docker build --tag=tooth_damage:latest .
# sudo docker create --gpus all -v ~/project/tooth_damage_detection/logs/test_2:/var/project/logs tooth_damage:latest


# sudo docker run -it --rm --gpus all -v ~/project/tooth_damage_detection/logs/test_2:/var/project/logs tooth_damage:latest bash