import os
from numpy.random import default_rng

old_path = 'archive/'
new_path = "African_Wildlife_with_val/"
classes = {c: len(os.listdir(f'{old_path}{c}/'))//2 for c in os.listdir('archive/')}
train_size = int(0.6 * 376)
test_val_size = (376 - train_size) // 2
rng = default_rng(seed=103)

for idx, c in enumerate(classes.keys()):
    fnames = os.listdir(f'archive/{c}/')
    fnum = set(f.split('.')[0] for f in fnames)

    # generating random splits
    test_split = rng.choice([x for x in fnum], size=test_val_size, replace=False)
    val_split = rng.choice([x for x in fnum if x not in test_split], size=test_val_size, replace=False)

    # saving
    for i,f in enumerate(fnum):
        name = str(i + 376 * idx)
        if f in test_split:
            os.rename(f'{old_path}{c}/{f}.jpg', f'{new_path}test/{name}.jpg')
        elif f in val_split:
            os.rename(f'{old_path}{c}/{f}.jpg', f'{new_path}val/{name}.jpg')
        else:
            os.rename(f'{old_path}{c}/{f}.jpg', f'{new_path}train/{name}.jpg')
        os.rename(f'{old_path}{c}/{f}.txt', f'{new_path}annotations/{name}.txt')