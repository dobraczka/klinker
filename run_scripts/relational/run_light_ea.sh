#!/bin/bash
iebsize="512"
depth="2"
small_nneighbors=500
large_nneighbors=1000

myargs=(
 "--wandb --random-seed 42 open-ea-dataset --graph-pair D_W --size 15K --version V1 light-ea-blocker"
 "--wandb --random-seed 42 open-ea-dataset --graph-pair D_W --size 15K --version V2 light-ea-blocker"
 "--wandb --random-seed 42 open-ea-dataset --graph-pair D_Y --size 15K --version V1 light-ea-blocker"
 "--wandb --random-seed 42 open-ea-dataset --graph-pair D_Y --size 15K --version V2 light-ea-blocker"
 "--wandb --random-seed 42 open-ea-dataset --graph-pair EN_DE --size 15K --version V1 light-ea-blocker"
 "--wandb --random-seed 42 open-ea-dataset --graph-pair EN_DE --size 15K --version V2 light-ea-blocker"
 "--wandb --random-seed 42 open-ea-dataset --graph-pair EN_FR --size 15K --version V1 light-ea-blocker"
 "--wandb --random-seed 42 open-ea-dataset --graph-pair EN_FR --size 15K --version V2 light-ea-blocker"
 "--wandb --random-seed 42 open-ea-dataset --graph-pair D_W --size 100K --version V1 light-ea-blocker"
 "--wandb --random-seed 42 open-ea-dataset --graph-pair D_W --size 100K --version V2 light-ea-blocker"
 "--wandb --random-seed 42 open-ea-dataset --graph-pair D_Y --size 100K --version V1 light-ea-blocker"
 "--wandb --random-seed 42 open-ea-dataset --graph-pair D_Y --size 100K --version V2 light-ea-blocker"
 "--wandb --random-seed 42 open-ea-dataset --graph-pair EN_DE --size 100K --version V1 light-ea-blocker"
 "--wandb --random-seed 42 open-ea-dataset --graph-pair EN_DE --size 100K --version V2 light-ea-blocker"
 "--wandb --random-seed 42 open-ea-dataset --graph-pair EN_FR --size 100K --version V1 light-ea-blocker"
 "--wandb --random-seed 42 open-ea-dataset --graph-pair EN_FR --size 100K --version V2 light-ea-blocker"
)

nnargs=()
for base in "${myargs[@]}"
do
	if [[ $base =~ .*15K.* ]]
	then
		nnargs+=("$base --n-neighbors $small_nneighbors")
	else
		nnargs+=("$base --n-neighbors $large_nneighbors")
	fi
done

sifembeddings="fasttext"
embeddings="gtr-t5-base"
multi_embeddings="LaBSE"
other_args=" --depth $depth --force True"
args=()
for base in "${nnargs[@]}"
do
	if [[ $base =~ .*D_Y.* ]] || [[ $base =~ .*D_W.* ]]
	then
		args+=("$base $other_args --embeddings $embeddings --inner-encoder sentencetransformertokenized --inner-encoder-batch-size $iebsize")
	else
		args+=("$base $other_args --embeddings $multi_embeddings --inner-encoder sentencetransformertokenized --inner-encoder-batch-size $iebsize")
	fi
done
for base in "${nnargs[@]}"
do
    args+=("$base $other_args --embeddings $sifembeddings --inner-encoder sifembeddingtokenized")
done

curr_param=$(echo ${args[$1]})
echo $curr_param

micromamba run -n klinker-conda -r y python experiment.py $curr_param
