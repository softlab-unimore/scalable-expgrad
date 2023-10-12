import os

import send2trash

base_plot_dir = os.path.join('..', 'results', 'plots')


for f in os.scandir(base_plot_dir):
    if not f.is_dir():
        continue
    for f2 in os.scandir(f):
        if not f2.is_dir():
            continue
        for f3 in os.scandir(f2):
            if f3.is_dir():
                send2trash.send2trash(f3)


    pass
