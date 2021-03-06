FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu14.04

MAINTAINER TrsNium

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

COPY requirements.txt /root/
COPY data /root/

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install git libssl-dev build-essential

RUN apt-get -y install software-properties-common
RUN add-apt-repository -y ppa:neovim-ppa/unstable
RUN apt-get -y update 
RUN apt-get -y install neovim

#setup nevoim
RUN git clone https://github.com/TrsNium/nvim_config.git ~/nvim_config
RUN mkdir ~/.config
RUN mv ~/nvim_config/nvim $HOME/.config/ && mv ~/nvim_config/dein ~/.config/
RUN rm -r ~/nvim_config

#setup pyenv
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
RUN echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
RUN echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc

RUN git clone https://github.com/TrsNium/Conditional_C_RNN_GAN.git ~/Conditional_C_RNN_GAN
RUN rm ~/scraip.py
RUN mv ~/genre-* ~/Conditional_C_RNN_GAN/data/
