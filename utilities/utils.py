import os
import matplotlib.pyplot as plt
from IPython.display import clear_output



def live_plot(data_dict, figsize=(15,5)):

    clear_output(wait=True)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    
    for label, data in data_dict.items():
        ax.plot(data, label=label)
    
    ax.legend(loc='lower left')
    
    return ax



def mkdir(base, name):

    path = os.path.join(base, name)
  
    if not os.path.exists(path):
        os.makedirs(path)
  
    return path


        