# athenak-vis
ALCF Visualization team scripts for AthenaK simulation outputs, originally for generating a video for the Department of Energy's SC24 Exhibitor booth. 

Slightly modified versions of scripts from original repo: https://github.com/IAS-Astrophysics/athenak/tree/main/vis/python

Convert full, multiresolution (SMR/AMR) 3D binary `.bin` restart dumps to HDF5 + XML pairs for ParaView visualization, and make the following changes:
- Cut out unused output variables to save disk space
- Convert observer-frame 3-vectors to relativistic velocity and magnetic field 4-vectors (contravariant Cartesian Kerr-Schild coordinate frame)

Example usage:
```bash
❯ python make_athdf.py -v /grand/RadBlackHoleAcc/vis/edd_survey/16_8_d30_s3/edd_survey.full
```

Converts 600 `.bin` files, 6.6 GB each, to `.athdf` files (2.1 GB each) and `.athdf.xdmf` files (5.0 MB each).  This simulation used SMR with 8 levels of refinement, so all source `.bin` files have the same number of MeshBlocks and occupy the same amount of storage space (and take roughly 1.5 minutes to convert). Approx. 4.0 TB source files reduced to about 2.1 TB in about 15 hours, if serial.

Cannot run on a Polaris UAN without getting killed due to memory limits (at least for 6.6 GB source file). Example usage on a compute node:
```bash
❯ qsub -A RadBlackHoleAcc -q debug -I -l select=1,walltime=0:60:00,filesystems=swift:grand
❯ module use /soft/modulefiles/; module load conda/2024-04-29; conda activate
❯ cd ~/athenak-vis/
❯ python make_athdf.py -v /grand/RadBlackHoleAcc/vis/edd_survey/16_8_d30_s3/edd_survey.full
```

### Original `.bin` complete list of `filedata["var_names"]`
```
# ['dens', 'velx', 'vely', 'velz', 'eint', 's_00', 'bcc1', 'bcc2', 'bcc3', 'r00', 'r01', 'r02', 'r03', 'r11', 'r12', 'r13', 'r22', 'r23', 'r33', 'r00_ff', 'r01_ff', 'r02_ff', 'r03_ff', 'r11_ff', 'r12_ff', 'r13_ff', 'r22_ff', 'r23_ff', 'r33_ff']
```

### Reduced HDF5 variable set with relativistic vector quantities

```
❯ h5dump -a VariableNames edd_survey.full.00228.athdf
HDF5 "edd_survey.full.00228.athdf" {
ATTRIBUTE "VariableNames" {
   DATATYPE  H5T_STRING {
      STRSIZE 21;
      STRPAD H5T_STR_NULLPAD;
      CSET H5T_CSET_ASCII;
      CTYPE H5T_C_S1;
   }
   DATASPACE  SIMPLE { ( 9 ) / ( 9 ) }
   DATA {
   (0): "dens\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000",
   (1): "ux\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000",
   (2): "uy\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000",
   (3): "uz\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000",
   (4): "r00\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000",
   (5): "r00_ff\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000",
   (6): "bx\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000",
   (7): "by\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000",
   (8): "bz\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
   }
}
}
```
