import load_data
import preprocess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

'''
Index(['TPSA', 'Saacc', 'H-050', 'MLOGP', 'RDCHI', 'GATS1p', 'nN', 'C-040',
       'Target'],
      dtype='object')

'''
rel_attributes = ['x','hue','size','style','row','col','col_wrap','row_order','col_order','hue_order','hue_norm','sizes','size_order','size_norm','markers','dashes','style_order','style_order']
n = len(rel_attributes)

def relplot(cols,t,d):
    s=""
    m = len(cols)
    if m > n:
        print('Too many Columns to represent')
    for i in range(m):
        temp = str(rel_attributes[i]) + '=' + "'" + str(cols[i]) +"'" + ","
        s = s + temp
    s = s + "y =" + "'" + str(t) + "'" + ","
    s = s + "data = " + d
    print(s)
    plt.show(sns.relplot())
    
def relplot1(c1, t, d):
    plt.show(sns.relplot(x = c1, y = t, data = df))


if __name__ == '__main__':
    filepath = 'C:/Users/Aniket/Desktop/Aniket/data/qsar_aquatic_toxicity.csv'
    data = pd.read_csv(filepath)
    df = pd.DataFrame(data)
    cols = ['TPSA', 'Saacc', 'H-050']
    relplot1('TPSA','Target',df)
    #print()
