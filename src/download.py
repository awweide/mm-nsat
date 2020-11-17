import urllib
import tarfile
import os

testfile = urllib.URLopener()
print "Downloading 177MB Inception Network for FID from tensorflow.org"
if not os.path.exists("data/fid_inception_model"):
    os.makedirs("data/fid_inception_model")
testfile.retrieve("http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz", "data/fid_inception_model/frozen_inception_v1_2015_12_05.tar.gz")

print "Downloading 163MB CIFAR-10 dataset from cs.toronto.edu"
testfile.retrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "data/cifar-10/cifar-10-python.tar.gz")

tar = tarfile.open("data/cifar-10/cifar-10-python.tar.gz", "r:gz")

for member in tar.getmembers():
    if member.isreg():  # skip if the TarInfo is not files
        member.name = os.path.basename(member.name) # remove the path by reset it
        tar.extract(member,"data/cifar-10/") # extract 
tar.close()

print "Downloads finished"



