#!/bin/bash

#
# To use, replace the INSERT_GUROBI_KEY_HERE with your Gurobi key. This is a string
# of alphanumerics and dashes that the Gurobi documentation says to use with grbgetkey.
# The installation will pause at one point, just hit enter.
#
#   bash install.sh INSERT_GUROBI_KEY_HERE
#   source ~/.bashrc
#
# You can then go into the deepkidney directory and run the examples. Remember to always
# run Python using the python3 command. On a fresh VM, you must install the following
# packages as root.
#
#   sudo apt-get install libblas-dev liblapack-dev libopenmpi-dev python3-dev git
#

USER=$(whoami)
cd ~
wget http://packages.gurobi.com/7.0/gurobi7.0.2_linux64.tar.gz
tar xvfz gurobi7.0.2_linux64.tar.gz
cd gurobi702
echo 'export GUROBI_HOME="/home/'$USER'/gurobi702/linux64"' >> ~/.bashrc
echo 'export PATH="${PATH}:${GUROBI_HOME}/bin"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"' >> ~/.bashrc
echo 'export GRB_LICENSE_FILE=/home/'$USER'/gurobi.lic' >> ~/.bashrc
~/gurobi702/linux64/bin/grbgetkey $1
wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py --user
cd linux64
python3 setup.py install --user
pip install --user numpy
pip install --user scipy
pip install --user tensorflow
pip install --user networkx
pip install --user gym
pip install --user baselines
pip install --user mpi4py
cd ~
mkdir deepkidney
cd ~/deepkidney
git clone https://github.com/camoy/gym-kidney
cd gym-kidney
pip install --user -e .
cd ~
wget https://gforge.inria.fr/frs/download.php/file/36849/spams-python3-v2.6-2017-06-06.tar.gz
tar xvfz spams-python3-v2.6-2017-06-06.tar.gz
cd spams-python3
python3 setup.py install --user
