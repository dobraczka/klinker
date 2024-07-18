#!/bin/bash
myargs=(
 "open-ea-dataset --graph-pair D_W --size 15K --version V1 relational-token-blocker --top-n-r 5"
 "open-ea-dataset --graph-pair D_W --size 15K --version V2 relational-token-blocker --top-n-r 7"
 "open-ea-dataset --graph-pair D_Y --size 15K --version V1 relational-token-blocker --top-n-r 3"
 "open-ea-dataset --graph-pair D_Y --size 15K --version V2 relational-token-blocker --top-n-r 4"
 "open-ea-dataset --graph-pair EN_DE --size 15K --version V1 relational-token-blocker --top-n-r 4"
 "open-ea-dataset --graph-pair EN_DE --size 15K --version V2 relational-token-blocker --top-n-r 6"
 "open-ea-dataset --graph-pair EN_FR --size 15K --version V1 relational-token-blocker --top-n-r 5"
 "open-ea-dataset --graph-pair EN_FR --size 15K --version V2 relational-token-blocker --top-n-r 8"
 "open-ea-dataset --graph-pair D_W --size 100K --version V1 relational-token-blocker --top-n-r 5"
 "open-ea-dataset --graph-pair D_W --size 100K --version V2 relational-token-blocker --top-n-r 7"
 "open-ea-dataset --graph-pair D_Y --size 100K --version V1 relational-token-blocker --top-n-r 3"
 "open-ea-dataset --graph-pair D_Y --size 100K --version V2 relational-token-blocker --top-n-r 4"
 "open-ea-dataset --graph-pair EN_DE --size 100K --version V1 relational-token-blocker --top-n-r 5"
 "open-ea-dataset --graph-pair EN_DE --size 100K --version V2 relational-token-blocker --top-n-r 6"
 "open-ea-dataset --graph-pair EN_FR --size 100K --version V1 relational-token-blocker --top-n-r 5"
 "open-ea-dataset --graph-pair EN_FR --size 100K --version V2 relational-token-blocker --top-n-r 8"
)

curr_param=$(echo ${args[$SLURM_ARRAY_TASK_ID]})
echo $curr_param

micromamba run -n klinker-conda -r y experiment.py $curr_param
