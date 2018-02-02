FROM nvidia/cuda:8.0-cudnn5-runtime

MAINTAINER TrsNium
                                             
COPY requirements.txt $HOME/
COPY data/ $HOME/

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install git libssl-dev build-essentiald

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
RUN echo 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc
