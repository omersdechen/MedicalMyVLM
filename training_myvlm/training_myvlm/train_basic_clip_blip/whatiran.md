
conda deactivate
conda deactivate
conda activate myvlm_fix


export PYTHONPATH=$PYTHONPATH:/home/user_7734/MyVLM
cd training_myvlm/

#dont run the below command if the concept head trainning works!!!
#rm -rf ~/.cache/huggingface/
pip install --no-cache-dir tokenizers==0.15.2
pip install --upgrade transformers


python /home/user_7734/MyVLM/concept_heads/clip/concept_head_training/train.py \
--config_path /home/user_7734/training_myvlm/train_basic_clip_blip/config/concept_head_training.yaml

cp /home/user_7734/training_myvlm/train_basic_clip_blip/output_concept_head_training/my_cat/DFN5B-CLIP-ViT-H-14-384-my_cat-step-500.pt \
/home/user_7734/training_myvlm/train_basic_clip_blip/trained_concept_head/my_cat/seed_42

python /home/user_7734/MyVLM/concept_embedding_training/train.py \
--config_path /home/user_7734/training_myvlm/train_basic_clip_blip/config/concept_embedding_training_captioning.yaml


python /home/user_7734/MyVLM/inference/run_myvlm_inference.py \
--config_path /home/user_7734/training_myvlm/train_basic_clip_blip/config/run_myvlm_inference.yaml





## hyoid bone conecpt embedding training
```bash
python concept_embedding_training/train.py \
--config_path /home/user_7734/omer/try_train_hyoid_bone_test/config/concept_embedding_training_captioning.yaml
```

## hyoid bone conecpt embedding training with LLaVa
```bash
python concept_embedding_training/train.py \
--config_path /home/user_7734/omer/try_train_hyoid_bone_test/config/LLAVA_concept_embedding_training_captioning.yaml
```

## MyVLM Inference
```bash
python inference/run_myvlm_inference.py \
--config_path /home/user_7734/omer/try_train_my_cat/config/run_myvlm_inference.yaml
```