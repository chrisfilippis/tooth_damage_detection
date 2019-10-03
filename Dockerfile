FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /var/project/data
COPY ./data /var/project/data

WORKDIR /var/project/src
COPY ./src /var/project/src

COPY requirements.txt /var/project/

RUN pip install --requirement /var/project/requirements.txt

CMD python tooth.py --data_dir=/var/project/data