wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZGyOpyN0ho64pEUZxAsMskK3zprNOAFg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZGyOpyN0ho64pEUZxAsMskK3zprNOAFg" -O resnet18_pascal.pth && rm -rf /tmp/cookies.txt
