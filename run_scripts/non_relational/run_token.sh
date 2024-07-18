#!/bin/bash
args=(
 "--random-seed 42 open-ea-dataset --graph-pair D_W --size 15K --version V1 token-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_W --size 15K --version V2 token-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_Y --size 15K --version V1 token-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_Y --size 15K --version V2 token-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_DE --size 15K --version V1 token-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_DE --size 15K --version V2 token-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_FR --size 15K --version V1 token-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_FR --size 15K --version V2 token-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_W --size 100K --version V1 token-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_W --size 100K --version V2 token-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_Y --size 100K --version V1 token-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_Y --size 100K --version V2 token-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_DE --size 100K --version V1 token-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_DE --size 100K --version V2 token-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_FR --size 100K --version V1 token-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_FR --size 100K --version V2 token-blocker"
)

curr_param=$(echo ${args[$1]})
echo $curr_param

micromamba run -n klinker-conda -r y experiment.py $curr_param
