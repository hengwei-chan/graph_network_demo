FROM centos:7

ENV HOME /home/
WORKDIR /home/

RUN yum -y update && \
    yum clean all && \
    rm -rf /var/cache/yum && \
    yum install -y python3 

COPY ./reports/model_weights/ /home/app/reports/model_weights/
COPY ./reports/configs/ /home/app/reports/configs/
COPY ./reports/predictions/ /home/app/reports/predictions/
COPY ./reports/stats/ /home/app/reports/stats/
COPY ./reports/tensorboard/ /home/app/reports/tensorboard/
COPY ./data/ /data/
COPY ./requirements/ /home/app/requirements/

RUN yes | bash /home/app/requirements/external_libs/CDPKit-1.0.0-Linux-x86_64.sh && \
    rm /home/app/requirements/external_libs/CDPKit-1.0.0-Linux-x86_64.sh

ENV PYTHONPATH=${PYTHONPATH}:/home/CDPKit/Python:${Pythonpath}
RUN python3 -m pip install -U pip && \
    pip3 install -r /home/app/requirements/base.txt

COPY ./graph_networks/ /home/app/graph_networks/
COPY ./scripts/ /home/app/scripts/
RUN sed -i 's/\r$//g' /home/app/scripts/*.py && \
    chmod +x /home/app/scripts/*.py

WORKDIR /home/app/

ENTRYPOINT ["python3","./scripts/run_gnn.py"]