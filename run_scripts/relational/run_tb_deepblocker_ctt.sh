#!/bin/bash
myargs=()
small_nneighbors=250
large_nneighbors=500
iebsize="512"
hidden_dim="384"
learning_rate="0.0030405"
max_perturbation="0.408395"
pos_to_neg_ratio="1.55515"
synth_tuples_per_tuple="5"

myargs=(
 "--wandb open-ea-dataset --graph-pair D_W --size 15K --version V1 composite-relational-deepblocker --top-n-r 5"
 "--wandb open-ea-dataset --graph-pair D_W --size 15K --version V2 composite-relational-deepblocker --top-n-r 7"
 "--wandb open-ea-dataset --graph-pair D_Y --size 15K --version V1 composite-relational-deepblocker --top-n-r 3"
 "--wandb open-ea-dataset --graph-pair D_Y --size 15K --version V2 composite-relational-deepblocker --top-n-r 4"
 "--wandb open-ea-dataset --graph-pair EN_DE --size 15K --version V1 composite-relational-deepblocker --top-n-r 4"
 "--wandb open-ea-dataset --graph-pair EN_DE --size 15K --version V2 composite-relational-deepblocker --top-n-r 6"
 "--wandb open-ea-dataset --graph-pair EN_FR --size 15K --version V1 composite-relational-deepblocker --top-n-r 5"
 "--wandb open-ea-dataset --graph-pair EN_FR --size 15K --version V2 composite-relational-deepblocker --top-n-r 8"
 "--wandb open-ea-dataset --graph-pair D_W --size 100K --version V1 composite-relational-deepblocker --top-n-r 5"
 "--wandb open-ea-dataset --graph-pair D_W --size 100K --version V2 composite-relational-deepblocker --top-n-r 7"
 "--wandb open-ea-dataset --graph-pair D_Y --size 100K --version V1 composite-relational-deepblocker --top-n-r 3"
 "--wandb open-ea-dataset --graph-pair D_Y --size 100K --version V2 composite-relational-deepblocker --top-n-r 4"
 "--wandb open-ea-dataset --graph-pair EN_DE --size 100K --version V1 composite-relational-deepblocker --top-n-r 5"
 "--wandb open-ea-dataset --graph-pair EN_DE --size 100K --version V2 composite-relational-deepblocker --top-n-r 6"
 "--wandb open-ea-dataset --graph-pair EN_FR --size 100K --version V1 composite-relational-deepblocker --top-n-r 5"
 "--wandb open-ea-dataset --graph-pair EN_FR --size 100K --version V2 composite-relational-deepblocker --top-n-r 8"
)

other_args="--encoder crosstupletraining --batch-size 512 --hidden-dimension $hidden_dim --learning-rate $learning_rate --force True --max-perturbation=$max_perturbation --pos-to-neg-ratio=$pos_to_neg_ratio --synth-tuples-per-tuple=$synth_tuples_per_tuple --block-builder-kwargs faisshnsw"
args=()
st_args="--inner-encoder sentencetransformertokenized --inner-encoder-batch-size $iebsize"
embeddings="gtr-t5-base"
multi_embeddings="LaBSE"
for base in "${myargs[@]}"
do
	if [[ $base =~ .*D_Y.* ]] || [[ $base =~ .*D_W.* ]]
	then
		args+=("$base $other_args $st_other_args --embeddings $embeddings")
	else
		args+=("$base $other_args $st_other_args --embeddings $multi_embeddings")
	fi
	if [[ $base =~ .*15K.* ]]
	then
		nnargs+=("$base --n-neighbors $small_nneighbors --rel-n-neighbors $small_nneighbors")
	else
		nnargs+=("$base --n-neighbors $large_nneighbors --rel-n-neighbors $large_nneighbors --block-builder-kwargs faisshnsw --reduce-dim-to $reduce_dim_to --reduce-sample-perc $reduce_sample_perc")
	fi
done
sif_other_args="--inner-encoder sifembeddingtokenized"
for base in "${nnargs[@]}"
do
	if [[ $base =~ .*15K.* ]]
	then
		nnargs+=("$base $other_args $sif_other_args --embeddings fasttext --n-neighbors $small_nneighbors")
	else
		nnargs+=("$base $other_args $sif_other_args --embeddings 100wiki.en.bin --n-neighbors $small_nneighbors --block-builder-kwargs faisshnsw")
	fi
done

curr_param=$(echo ${args[$1]})
echo $curr_param

micromamba run -n klinker-conda -r y python experiment.py $curr_param
