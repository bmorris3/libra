
import os

cfg_1 = """
# you can change this so condor will email you under certain circumstances.
Notification = never

# very necessary. This loads your .cshrc file and makes Condor aware of your home directory/the network before the job starts.
# the job will fail without this line.
getenv = true

# what file condor will run
Executable = /astro/users/bmmorris/git/libra/condor_run.sh

# which directory your code starts in (e.g. calls to ./ mean this directory)
Initialdir = /astro/users/bmmorris/git/libra

# read the documentation before changing this
Universe   = vanilla

# Condor log information
"""

cfg_2 = """
Log        = {log_path}
Output     = {output_path}
Error = {error_path}
Arguments  = {arg}
Queue
"""

output_dir = '/astro/store/scratch/tmp/bmmorris/libra/outputs'

from libra import ObservationArchive

with ObservationArchive('trappist1_bright2_b', 'r') as obs:
    n_transits = len(obs.b)

cfg = cfg_1

python_path = "/astro/users/bmmorris/premap/miniconda3/bin/python"

for i in range(n_transits):
    cfg += cfg_2.format(log_path=os.path.join(output_dir,
                                              'log_{0:03d}.txt'.format(i)),
                        output_path=os.path.join(output_dir,
                                                 'out_{0:03d}.txt'.format(i)),
                        error_path=os.path.join(output_dir,
                                                'err_{0:03d}.txt'.format(i)),
                        arg=i)

with open('condor.cfg', 'w') as w:
    w.write(cfg)
