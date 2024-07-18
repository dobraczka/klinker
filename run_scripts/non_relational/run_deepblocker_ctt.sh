#!/bin/bash
small_nneighbors=500
large_nneighbors=1000
iebsize="512"
hidden_dim="384"
learning_rate="0.0030405"
max_perturbation="0.408395"
pos_to_neg_ratio="1.55515"
synth_tuples_per_tuple="5"

myargs=(
 "--random-seed 42 open-ea-dataset --graph-pair D_W --size 15K --version V1 deepblocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_W --size 15K --version V2 deepblocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_Y --size 15K --version V1 deepblocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_Y --size 15K --version V2 deepblocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_DE --size 15K --version V1 deepblocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_DE --size 15K --version V2 deepblocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_FR --size 15K --version V1 deepblocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_FR --size 15K --version V2 deepblocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_W --size 100K --version V1 deepblocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_W --size 100K --version V2 deepblocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_Y --size 100K --version V1 deepblocker"
 "--random-seed 42 open-ea-dataset --graph-pair D_Y --size 100K --version V2 deepblocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_DE --size 100K --version V1 deepblocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_DE --size 100K --version V2 deepblocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_FR --size 100K --version V1 deepblocker"
 "--random-seed 42 open-ea-dataset --graph-pair EN_FR --size 100K --version V2 deepblocker"
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

embeddings="gtr-t5-base"
multi_embeddings="LaBSE"
st_other_args="--encoder crosstupletraining --inner-encoder sentencetransformertokenized --inner-encoder-batch-size $iebsize --batch-size 512 --hidden-dimension $hidden_dim --learning-rate $learning_rate --force True --max-perturbation=$max_perturbation --pos-to-neg-ratio=$pos_to_neg_ratio --synth-tuples-per-tuple=$synth_tuples_per_tuple --block-builder-kwargs faisshnsw"
args=()
for base in "${nnargs[@]}"
do
	if [[ $base =~ .*D_Y.* ]] || [[ $base =~ .*D_W.* ]]
	then
		args+=("$base $st_other_args --embeddings $embeddings")
	else
		args+=("$base $st_other_args --embeddings $multi_embeddings")
	fi
done

sifembeddings="fasttext"
sif_other_args="--encoder crosstupletraining --inner-encoder sifembeddingtokenized --batch-size 512 --hidden-dimension $hidden_dim --learning-rate $learning_rate --force True --max-perturbation=$max_perturbation --pos-to-neg-ratio=$pos_to_neg_ratio --synth-tuples-per-tuple=$synth_tuples_per_tuple"
for base in "${nnargs[@]}"
do
	args+=("$base $other_args --embeddings $sifembeddings")
done

curr_param=$(echo ${args[$1]})
echo $curr_param

micromamba run -n klinker-conda -r y python experiment.py $curr_param
