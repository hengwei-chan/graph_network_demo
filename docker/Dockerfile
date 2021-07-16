FROM centos:7

ENV HOME /home/
WORKDIR /home/

RUN yum -y update && \
    yum clean all && \
    rm -rf /var/cache/yum && \
    yum install -y python3 

COPY ./requirements /home/app/requirements
COPY ./reports/model_weights/dgin_m2_6_4/epoch_890/* /home/app/reports/model_weights/dgin_m2_6_4/epoch_890/
COPY ./data/try_2.xls /data/try_2.xls

RUN yes | bash /home/app/requirements/external_libs/CDPKit-1.0.0-Linux-x86_64.sh && \
    rm /home/app/requirements/external_libs/CDPKit-1.0.0-Linux-x86_64.sh

ENV PYTHONPATH=${PYTHONPATH}:/home/CDPKit/Python:${Pythonpath}
RUN python3 -m pip install -U pip && \
    pip3 install -r /home/app/requirements/base.txt

COPY ./graphnets/ /home/app/graphnets/
COPY ./scripts/ /home/app/scripts/
RUN sed -i 's/\r$//g' /home/app/scripts/*.py && \
    chmod +x /home/app/scripts/*.py

WORKDIR /home/app/

ENTRYPOINT ["python3","./scripts/run_logd.py"]