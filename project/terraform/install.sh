sudo apt-get update -y
sudo apt-get dist-upgrade -y
sudo apt install unzip -y
sudo apt install python3-pip -y

#sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
#sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/'
#sudo apt update
#sudo apt install r-base -y

sudo mkdir /mnt/data-disk
sudo mount -o discard,defaults /dev/disk/by-id/google-data-disk /mnt/data-disk

pip3 install tqdm
pip3 install IPython
pip3 install pandas
pip3 install numpy
pip3 install scipy
pip3 install matplotlib
