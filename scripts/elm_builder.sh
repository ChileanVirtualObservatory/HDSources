#!/bin/bash
#PBS -l cput=150:00:00
#PBS -l walltime=150:00:00

/user/m/marvill/anaconda2/bin/python2 /user/m/marvill/VarClump/scripts/elm_builder.py $1 $2 $3
