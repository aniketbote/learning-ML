from load_data import *
from sklearn import preprocessing as pre
from sklearn.model_selection import train_test_split


def train_test(X, y,ratio):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ratio, random_state=42)
    return X_train, X_test, y_train, y_test


'''
ALL SCALERS COMAPRISON
https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
'''
#Zero mean and unit variance
def standardize(X):  
    scaler = pre.StandardScaler().fit(X)
    X = scaler.transform(X)
    return scaler,X


#range 0 - 1 (sparse data)
def min_max(X):
    scaler = pre.MinMaxScaler().fit(X)
    X = scaler.transform(X)
    return scaler,X

# range -1  - 1 (sparse data)
def max_abs(X): 
    scaler = pre.MaxAbsScaler().fit(X)
    X = scaler.transform(X)
    return scaler,X

#for ouliers
def robust(X):
    scaler = pre.RobustScaler().fit(X)
    X = scaler.transform(X)
    return scaler,X

def normalize(X):
    scaler = pre.Normalizer().fit(X)
    X = scaler.transform(X)
    return scaler,X

def quantile(X):
    scaler = pre.QuantileTransformer().fit(X)
    X = scaler.transform(X)
    return scaler,X

def power(X, m):
    try:
        scaler = pre.PowerTransformer(method = m).fit(X,m)
        X = scaler.transform(X)
    except ValueError:
        if m == 'box-cox':
            print('box-cox doesnt handle negative values')
        return 0,0
    return scaler,X

    
if __name__ == "__main__" :
    filepath = 'C:/Users/Aniket/Desktop/Aniket/data/qsar_aquatic_toxicity.csv'
    imagedir = 'C:/Users/Aniket/Desktop/Aniket/data/chest_xray/val'
    start_col = 0
    end_col = 7
    target_col = 8

    X,y = csv_to_numpy(filepath,end_col, target_col)
    X_train, X_test, y_train, y_test = train_test(X,y,0.3)
    scaler, lol0 = standardize(X_train) 
    scaler, lol1 =  min_max(X_train)
    scaler, lol2 = robust(X_train)
    scaler, lol3 = quantile(X_train) 
    scaler, lol4 =  power(X_train,'yeo-johnson')
    scaler, lol5 = power(X_train,'box-cox')


    print(lol0.mean(axis = 0))
    print()
    print(lol1.mean(axis = 0))
    print()
    print(lol2.mean(axis = 0))
    print()
    print(lol3.mean(axis = 0))
    print()
    print(lol4.mean(axis = 0))
    print()
    print(lol5.mean(axis = 0))

    
    
