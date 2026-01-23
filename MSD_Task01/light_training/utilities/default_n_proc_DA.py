import subprocess
import os


def get_allowed_n_proc_DA():
    """"""

    if 'nnUNet_n_proc_DA' in os.environ.keys():
        use_this = int(os.environ['nnUNet_n_proc_DA'])
    else:
        hostname = subprocess.getoutput(['hostname'])
        if hostname in ['Fabian', ]:
            use_this = 12
        elif hostname in ['hdf19-gpu16', 'hdf19-gpu17', 'hdf19-gpu18', 'hdf19-gpu19', 'e230-AMDworkstation']:
            use_this = 16
        elif hostname.startswith('e230-dgx1'):
            use_this = 10
        elif hostname.startswith('hdf18-gpu') or hostname.startswith('e132-comp'):
            use_this = 16
        elif hostname.startswith('e230-dgx2'):
            use_this = 6
        elif hostname.startswith('e230-dgxa100-'):
            use_this = 28
        elif hostname.startswith('lsf22-gpu'):
            use_this = 28
        elif hostname.startswith('hdf19-gpu') or hostname.startswith('e071-gpu'):
            use_this = 12
        else:
            use_this = 12                 

    use_this = min(use_this, os.cpu_count())
    return use_this
