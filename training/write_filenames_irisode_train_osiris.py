from __future__ import division
import numpy as np
import math
import os
import random
import collections
import sys
import os.path

SRC_FOLDER1='/home/IrisCode/NormalizedImages_jpeg'
SRC_FOLDER2='/home/IrisCode/NormalizedMasks_jpeg'
SRC_FOLDER3='/home/IrisCode/IrisCodes_jpeg'

FileDest_FOLDER='/home/IrisCode/osiris'

FileDest_FOLDER_train='/home/IrisCode/osiris/train_2splits'
FileDest_FOLDER_test1='/home/IrisCode/osiris/test1_2splits'


if not os.path.exists(FileDest_FOLDER):
    os.makedirs(FileDest_FOLDER)

files1=os.listdir(SRC_FOLDER1)
files2=os.listdir(SRC_FOLDER2)
files3=os.listdir(SRC_FOLDER3)




listID=dict()


for f in files1:
    id=f.split('_')[0]
    if id in listID:
        listID[id].append(f)
    else:
        listID[id]=[f]



ID=[]
for f in files1:
    ID.append(f.split('_')[0])

ID=list(set(ID))

le=len(ID)
# a=np.int(np.floor(le/2.0))


trainlist=ID[0:np.floor(le*.8).astype(np.int)]
test1list=ID[np.floor(le*.8).astype(np.int):]




file1 = open(FileDest_FOLDER_train, 'w')
c=0
for id in trainlist:
    for file in listID[id]:

        t1 = file+'++'

        li = file.rsplit('imno', 1)
        a = 'mano'.join(li)
        t1=t1+a+'++'

        li = file.rsplit('imno', 1)
        a = 'code'.join(li)
        t1 = t1 + a

        file1.write(t1 + '\n')
        c+=1

file1.close()
print (c)

file1 = open(FileDest_FOLDER_test1, 'w')
c=0
for id in test1list:
    for file in listID[id]:
        t1 = file + '++'

        li = file.rsplit('imno', 1)
        a = 'mano'.join(li)
        t1 = t1 + a + '++'

        li = file.rsplit('imno', 1)
        a = 'code'.join(li)
        t1 = t1 + a

        file1.write(t1 + '\n')
        c+=1

file1.close()
print (c)

