
conda deactivate
conda deactivate
conda activate medicalmyvlm
#new env!

export PYTHONPATH=$PYTHONPATH:/home/user_7734/MedicalMyVLM
cd ~/training_myvlm/

#dont run the below command if the concept head trainning works!!!
#rm -rf ~/.cache/huggingface/
pip install --no-cache-dir tokenizers==0.15.2
pip install --upgrade transformers


python /home/user_7734/MedicalMyVLM/concept_heads/clip/concept_head_training/train.py \
--config_path /home/user_7734/training_myvlm/train_medical_clip_basic_blip/config/concept_head_training.yaml

mkdir -p /home/user_7734/training_myvlm/train_medical_clip_basic_blip/trained_concept_head/hyoidbone/seed_42

cp /home/user_7734/training_myvlm/train_medical_clip_basic_blip/output_concept_head_training/hyoidbone/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-hyoidbone-step-500.pt \
/home/user_7734/training_myvlm/train_medical_clip_basic_blip/trained_concept_head/hyoidbone/seed_42

python /home/user_7734/MedicalMyVLM/concept_embedding_training/train.py \
--config_path /home/user_7734/training_myvlm/train_medical_clip_basic_blip/config/concept_embedding_training_captioning.yaml


python /home/user_7734/MedicalMyVLM/inference/run_myvlm_inference.py \
--config_path /home/user_7734/training_myvlm/train_medical_clip_basic_blip/config/run_myvlm_inference.yaml


