import pandas as pd
import os
import re
import shutil

src_pne = 'pne'
src_nonpne = 'nonpne'
data_dict=[]
data = pd.read_csv('ALL.csv')
df = pd.DataFrame(data)
class_id = list(df['filename'])
class_value = list(df['class'])
count=0
temp={}
def find(lst, key, value):
    for i, dic in enumerate(lst):
        if dic[key] == value:
            return i
    return -1
for i,j in zip(class_id,class_value):
    temp['name']=i
    temp['value']=j
    data_dict.append(temp)
    temp={}



filename = list(os.listdir('ALL'))
print(len(filename))
for i in range(len(filename)):
    temp_file = re.sub('.jpeg','',filename[i])
    index_file = find(data_dict,'name',temp_file)
    if data_dict[index_file]['value'] == 'pne':
        shutil.copy(os.path.join('ALL', filename[i]), src_pne)
        print(os.path.join('ALL', filename[i]))
    else:
        shutil.copy(os.path.join('ALL', filename[i]), src_nonpne)
    count+=1
    if count == 10:
        break
    print(count)
    
