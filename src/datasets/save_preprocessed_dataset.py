import numpy as np

from datasets.cifar10_count_dataset import CIFAR10CountDataset
from utils.system import create_dirs

image_grid_distribution = (3, 3)

train_set = CIFAR10CountDataset('./data/CIFAR10', image_grid_distribution, template_view='raw', train=True)
test_set = CIFAR10CountDataset('./data/CIFAR10', image_grid_distribution, template_view='raw', train=False)

create_dirs('./data/CIFAR10Count')
create_dirs('./data/CIFAR10Count/templates')
image_grid, templates, counts = train_set[0]
for i, template in enumerate(templates):
    template.save('./data/CIFAR10Count/templates/' + train_set.class_names[i] + '.jpg', 'JPEG')

create_dirs('./data/CIFAR10Count/train')
create_dirs('./data/CIFAR10Count/train/images')
create_dirs('./data/CIFAR10Count/train/counts')
with open('./data/CIFAR10Count/train/counts/counts.txt', 'w') as counts_file:
    for i, data in enumerate(train_set):
        image_grid, _, counts = data
        counts = [count[0] for count in counts.astype(np.int32)]
        image_grid.save(f'./data/CIFAR10Count/train/images/{i}.jpg', 'JPEG')
        counts_file.write(f'{" ".join(map(str, counts))}\n')

create_dirs('./data/CIFAR10Count/test')
create_dirs('./data/CIFAR10Count/test/images')
create_dirs('./data/CIFAR10Count/test/counts')
with open('./data/CIFAR10Count/test/counts/counts.txt', 'w') as counts_file:
    for i, data in enumerate(train_set):
        image_grid, _, counts = data
        counts = [count[0] for count in counts.astype(np.int32)]
        image_grid.save(f'./data/CIFAR10Count/test/images/{i}.jpg', 'JPEG')
        counts_file.write(f'{" ".join(map(str, counts))}\n')
