import shutil
import os
import random

i=0
path =r'E:\TE\test'
files = os.listdir(path)
while(i<100):
    index = random.randrange(0, len(files))
    print(files[index])
    shutil.move(os.path.join(path,files[index]), r'E:\Third\test2')
    files.remove(files[index])
    i=i+1
#os.rename(source, destination)
