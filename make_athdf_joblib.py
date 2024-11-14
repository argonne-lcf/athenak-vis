# A simple script for converting a collection of .bin files to .athdf/.xdmf files using
# bin_convert_sc24

# Python modules
import os
import argparse
import glob
from multiprocessing import cpu_count
from joblib import Parallel, delayed

# AthenaK modules
import bin_convert_sc24 as bin_convert


# Polaris compute node: 32 core CPU w/ HT
nprocs = 16   # 32 = int(cpu_count())/2
# 512 GB RAM
# 4 procs ---> ~3.0 %CPU in top output

# Main function
def main(**kwargs):
    # Get the root name for the file.
    files = glob.glob(kwargs['file_stem'] + '*.bin')
    if len(files) < 1:
        print(f"No files found with stem {kwargs['file_stem']}")
        quit()

    # total = len(files)
    # count = 1
    Parallel(n_jobs=int(nprocs))(delayed(convert_single_file)(
        fname, kwargs['verbose'], kwargs['overwrite']) for fname in files)


def convert_single_file(fname, verbose=False, overwrite=False):
    athdf_name = fname.replace(".bin", ".athdf")
    xdmf_name = athdf_name + ".xdmf"
    # if only .athdf OR .athdf.xdmf exists, script will reconvert both in all cases
    if (os.path.isfile(athdf_name) and os.path.isfile(xdmf_name)) and not overwrite:
        if verbose:
            print(f'Skipping source {fname} since overwrite={overwrite}')
            print(f'and both {athdf_name} and {xdmf_name} already exist...')
    else:
        filedata = bin_convert.read_binary(fname)
        bin_convert.write_athdf(athdf_name, filedata)
        bin_convert.write_xdmf_for(xdmf_name, os.path.basename(athdf_name), filedata)
        if verbose:
            print(f'Converting {fname}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_stem', help='path to files, excluding .#.bin')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='print file conversion progress')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='repeat conversion even if either destination file exists')
    args = parser.parse_args()
    print("Number of logical cores: ", int(cpu_count()))
    print("Number of processes: ", nprocs)
    main(**vars(args))
