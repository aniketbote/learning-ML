import pandas as pd
x = pd.read_csv('Dataset/train.csv')
df = pd.DataFrame(x)
test_list = df['target'].tolist()
z=0
print(test_list.count(z))
