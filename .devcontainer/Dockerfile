FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq update && apt-get -qqy upgrade

# set timezone
ENV TZ=Asia/Tokyo
RUN apt-get -qqy install tzdata

# Set locale
RUN apt-get -qqy install locales && \
  sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
  locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8   

# Create non-root user
RUN apt-get -qqy install sudo
ARG USERNAME=gh-runner
ARG UID=1000
ARG GID=$UID
RUN groupadd --gid $GID $USERNAME \
	&& useradd --uid $UID --gid $GID -m $USERNAME -s /bin/bash \
	&& echo "$USERNAME ALL=(root) NOPASSWD: ALL" > /etc/sudoers.d/$USERNAME \
	&& chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get -qqy install --no-install-recommends \
	curl \
  wget \
  unzip \
  uidmap \
  debootstrap \
  ca-certificates


WORKDIR /opt
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
    rm -r Miniconda3-latest-Linux-x86_64.sh

ENV PATH /opt/miniconda3/bin:$PATH

USER $USERNAME
WORKDIR /home/dev/workspace/notebooks

CMD ["/bin/bash"]