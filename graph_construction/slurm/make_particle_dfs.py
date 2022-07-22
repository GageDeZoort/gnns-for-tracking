import os
import sys

sys.path.append("../")
sys.path.append("../../")
import itertools
import measure_particle_properties

# grab job ID
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
print(f"job {idx}")

# hyperparameters
n_tasks = 50
train = [1]
tasks = range(n_tasks)
params = list(itertools.product(train, tasks))
params = params[idx]

args = [
    "--input-dir",
    f"/tigress/jdezoort/codalab/train_{params[0]}",
    "--output-dir",
    "../particle_properties",
    "--n-workers",
    "4",
    "--n-tasks",
    f"{n_tasks}",
    "--task",
    f"{params[1]}",
]

measure_particle_properties.main(args)
