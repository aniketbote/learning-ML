import load_data
import preprocess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

filepath = 'C:/Users/Aniket/Desktop/Aniket/data/qsar_aquatic_toxicity.csv'

data = pd.read_csv(filepath)
df = pd.DataFrame(data)


def relplot():
    

print(df.columns)
