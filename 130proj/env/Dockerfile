FROM dorowu/ubuntu-desktop-lxde-vnc:bionic
ENV PYTHONUNBUFFERED 1
RUN add-apt-repository universe && \
    apt update && \
    apt install freeglut3-dev -y -qq && \
    # apt upgrade -y -qq && \
    apt-get update -y -qq && \
    # apt-get upgrade -y -qq && \
    apt-get install vim python python3 python3-pip -y -qq
RUN mkdir /prjroot
COPY requirements.txt /prjroot/
WORKDIR /prjroot
RUN pip3 install -r requirements.txt
COPY . /prjroot
#CMD ["nginx", "-g", "daemon off;"]
