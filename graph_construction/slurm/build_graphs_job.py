import os
import sys
sys.path.append('../')
sys.path.append('../../')
import itertools
import build_graphs
import measure_particle_properties

# grab job ID
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
print(f'job {idx}')

# hyperparameters
n_tasks = 100
train = [1]
tasks = range(n_tasks)
params = list(itertools.product(train, tasks))
p = params[idx]

args = ['../configs/build_graphs.yaml',
        '--n-tasks', f'{n_tasks}', 
        '--task', f'{p[1]}',
        '--n-workers', '4']

build_graphs.main(args)
