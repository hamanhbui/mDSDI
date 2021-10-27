#Download the datasets
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1eObdrCIawCltN_PRUdJfGH1Rj6g0A1ii' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1eObdrCIawCltN_PRUdJfGH1Rj6g0A1ii" -O data.zip && rm -rf /tmp/cookies.txt
unzip data.zip
rm -rf data.zip
cd data/DomainNet/Raw\ images/

#Download DomainNet dataset
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip
unzip clipart.zip
rm -rf clipart.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip
unzip infograph.zip
rm -rf infograph.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip
unzip sketch.zip
rm -rf sketch.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip
unzip real.zip
rm -rf real.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip
unzip quickdraw.zip
rm -rf quickdraw.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip
unzip painting.zip
rm -rf painting.zip