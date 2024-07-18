#!/bin/bash
small_nneighbors=500
large_nneighbors=1000
myargs=(
 "--random-seed 42 open-ea-dataset --graph-pair D_W --size 15K --version V1 only-embeddings-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_W --size 15K --version V2 only-embeddings-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_Y --size 15K --version V1 only-embeddings-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_Y --size 15K --version V2 only-embeddings-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_DE --size 15K --version V1 only-embeddings-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_DE --size 15K --version V2 only-embeddings-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_FR --size 15K --version V1 only-embeddings-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_FR --size 15K --version V2 only-embeddings-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_W --size 100K --version V1 only-embeddings-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_W --size 100K --version V2 only-embeddings-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_Y --size 100K --version V1 only-embeddings-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_Y --size 100K --version V2 only-embeddings-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_DE --size 100K --version V1 only-embeddings-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_DE --size 100K --version V2 only-embeddings-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_FR --size 100K --version V1 only-embeddings-blocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_FR --size 100K --version V2 only-embeddings-blocker"
)

nnargs=()
for base in "${myargs[@]}"
do
	if [[ $base =~ .*15K.* ]]
	then
		nnargs+=("$base --n-neighbors $small_nneighbors")
	else
		nnargs+=("$base --n-neighbors $large_nneighbors --block-builder-kwargs faisshnsw")
	fi
done

sifembeddings="fasttext"
embeddings="gtr-t5-base"
multi_embeddings="LaBSE"
args=()
for base in "${nnargs[@]}"
do
	if [[ $base =~ .*D_Y.* ]] || [[ $base =~ .*D_W.* ]]
	then
		args+=("$base --embeddings $embeddings --encoder sentencetransformertokenized --inner-encoder-batch-size $iebsize")
	else
		args+=("$base --embeddings $multi_embeddings --encoder sentencetransformertokenized --inner-encoder-batch-size $iebsize")
	fi
done
for base in "${nnargs[@]}"
do
	args+=("$base --encoder sifembeddingtokenized --embeddings $sifembeddings")
done

curr_param=$(echo ${args[$1]})
echo $curr_param

micromamba run -n klinker-conda -r y python experiment.py $curr_param
