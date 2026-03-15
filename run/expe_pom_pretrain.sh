#!/usr/bin/bash
source /your/path/to/bin/activate;
conda activate pom;
SECONDS=0;
(OMP_NUM_THREADS=2 torchrun --nproc_per_node=2 --nnodes=1 /your/path/to/speechbrain/recipes/LibriSpeech/self-supervised-learning/BEST-RQ/train.py \
 	/your/path/to/pom/configs/BEST-RQ_pom_example.yaml \
	--data_folder /your/path/to/LibriSpeech \
	--logs_folder /your/path/to/pom/logs/)
duration=$SECONDS;
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed.";
conda deactivate;
exit 0