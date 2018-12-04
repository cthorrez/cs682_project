import os
import shutil


cats = os.listdir('data/train')
for cat in cats:
    dirname = 'data/train/'+cat
    if os.path.exists(dirname+'/labels'):
        shutil.rmtree(dirname+'/labels')