import sys
import os

path = sys.argv[1]

fids = []
jss = []
mixes = []

for i in xrange(0,100):
    fpath = path + str(i) + '/fid_0.txt'
    if not os.path.exists(fpath): break
    fid = []
    js = []
    for j in xrange(0,20000):
        if os.path.exists(path + str(i) + '/fid_' + str(j) + '.txt'):
            with open(path + str(i) + '/fid_' + str(j) + '.txt') as f:
                _fid = (f.readline().strip('\n'))
                if _fid == 'nan': _fid = 500.0
                else: _fid = float(_fid)
                fid.append(_fid)
            with open(path + str(i) + '/cls_' + str(j) + '.txt') as f:
                lines = f.readlines()
                elem = lines[-1]
                _js = float(elem)
                js.append(_js)
    fids.append(fid)
    jss.append(js)
    mixes.append(zip(fid,js))

strng = '[\n'
for m in mixes: strng += str(m) + ',\n'
strng += ']'
print '(FID,DJSCD) during training for each run:'
print strng
print ''

a = [f[-1] for f in fids]
print 'Final FID: min, mean, max:'
print  min(a), sum(a) / len(a), max(a)
