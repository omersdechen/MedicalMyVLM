
mkdir -p /home/user_7734/training_myvlm/train_medical_clip_basic_llava/trained_concept_head/hyoidbone/seed_42

#copy the concept head to the dir we created


python /home/user_7734/MedicalMyVLM/concept_heads/clip/concept_head_training/train.py \
--config_path /home/user_7734/training_myvlm/train_medical_clip_basic_llava/config/concept_head_training.yaml



python /home/user_7734/MyVLM/concept_embedding_training/train.py \
--config_path /home/user_7734/training_myvlm/train_medical_clip_basic_llava/config/concept_embedding_training_captioning.yaml


python /home/user_7734/MyVLM/inference/run_myvlm_inference.py \
--config_path /home/user_7734/training_myvlm/train_medical_clip_basic_llava/config/run_myvlm_inference.yaml



pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__)"