#!/bin/bash
iebsize="512"
embeddings="gtr-t5-base"
multi_embeddings="LaBSE"
reduce_transformer_dim_to="32"
reduce_sample_perc="0.3"
noise_cluster_handling="token"
reduce_dim_to=25
umap_n_neighbors=500
umap_min_dist=0.1

myargs=(
 "open-ea-dataset --graph-pair D_W --size 15K --version V1 composite-relational-token-clustering-blocker --top-n-r 5"
 "open-ea-dataset --graph-pair D_W --size 15K --version V2 composite-relational-token-clustering-blocker --top-n-r 7"
 "open-ea-dataset --graph-pair D_Y --size 15K --version V1 composite-relational-token-clustering-blocker --top-n-r 3"
 "open-ea-dataset --graph-pair D_Y --size 15K --version V2 composite-relational-token-clustering-blocker --top-n-r 4"
 "open-ea-dataset --graph-pair EN_DE --size 15K --version V1 composite-relational-token-clustering-blocker --top-n-r 4"
 "open-ea-dataset --graph-pair EN_DE --size 15K --version V2 composite-relational-token-clustering-blocker --top-n-r 6"
 "open-ea-dataset --graph-pair EN_FR --size 15K --version V1 composite-relational-token-clustering-blocker --top-n-r 5"
 "open-ea-dataset --graph-pair EN_FR --size 15K --version V2 composite-relational-token-clustering-blocker --top-n-r 8"
 "open-ea-dataset --graph-pair D_W --size 100K --version V1 composite-relational-token-clustering-blocker --top-n-r 5"
 "open-ea-dataset --graph-pair D_W --size 100K --version V2 composite-relational-token-clustering-blocker --top-n-r 7"
 "open-ea-dataset --graph-pair D_Y --size 100K --version V1 composite-relational-token-clustering-blocker --top-n-r 3"
 "open-ea-dataset --graph-pair D_Y --size 100K --version V2 composite-relational-token-clustering-blocker --top-n-r 4"
 "open-ea-dataset --graph-pair EN_DE --size 100K --version V1 composite-relational-token-clustering-blocker --top-n-r 5"
 "open-ea-dataset --graph-pair EN_DE --size 100K --version V2 composite-relational-token-clustering-blocker --top-n-r 6"
 "open-ea-dataset --graph-pair EN_FR --size 100K --version V1 composite-relational-token-clustering-blocker --top-n-r 5"
 "open-ea-dataset --graph-pair EN_FR --size 100K --version V2 composite-relational-token-clustering-blocker --top-n-r 8"
)

other_args="--noise-cluster-handling $noise_cluster_handling --use-unique-name False --save True"
st_args="--inner-encoder sentencetransformertokenized --inner-encoder-batch-size $iebsize --reduce-dim-to $reduce_transformer_dim_to --reduce-sample-perc $reduce_sample_perc"
args=()
for base in "${myargs[@]}"
do
	if [[ $base =~ .*D_Y.* ]] || [[ $base =~ .*D_W.* ]]
	then
		args+=("$base $other_args $st_args --embeddings $embeddings")
	else
		args+=("$base $other_args $st_args --embeddings $multi_embeddings")
	fi
done
sif_args="--inner-encoder sifembeddingtokenized --reduce-dim-to $reduce_dim_to --umap-n-neighbors $umap_n_neighbors --umap-min-dist $umap_min_dist"
for base in "${myargs[@]}"
do
    args+=("$base $other_args $sif_args --embeddings $embeddings")
done

curr_param=$(echo ${args[$1]})
echo $curr_param

micromamba run -n klinker-conda -r y python experiment.py $curr_param
