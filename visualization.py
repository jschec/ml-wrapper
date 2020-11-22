import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(dataframe, heatmap_figsize=(50,50), heatmap_clr_scheme=plt.cm.Blues, heatmap_file_name='./corr_matrix.png'):
    dataframe = dataframe.apply(pd.to_numeric, errors='coerce') 
    cor = dataframe.corr()
    plt.figure(figsize=heatmap_figsize)
    sns.heatmap(cor, annot=True, cmap=heatmap_clr_scheme)
    plt.tight_layout()
    plt.savefig(heatmap_file_name)