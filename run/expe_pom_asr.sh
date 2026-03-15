#!/usr/bin/bash
source /your/path/to/bin/activate;
conda activate pom;
SECONDS=0;
(OMP_NUM_THREADS=2 torchrun --nproc_per_node=2 /your/path/to/speechbrain/recipes/LibriSpeech/ASR/CTC/train_with_bestrq.py \
        /your/path/to/pom/configs/train_sb_BEST-RQ_pom_example.yaml \
        --pt_model_path /your/path/to/pom/logs/results/1000/save/CKPT+id.pt)
duration=$SECONDS;
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed.";
conda deactivate
echo "Finished experiment"
exit 0