

python /home/user_7734/MedicalMyVLM/concept_heads/clip/concept_head_training/train.py \
--config_path /home/user_7734/training_myvlm/train_medical_clip_llava_new_data/config/concept_head_training.yaml


python /home/user_7734/MedicalMyVLM/concept_embedding_training/train.py \
--config_path /home/user_7734/training_myvlm/train_medical_clip_llava_new_data/config/concept_embedding_training_captioning.yaml


python /home/user_7734/MedicalMyVLM/inference/run_myvlm_inference.py \
--config_path /home/user_7734/training_myvlm/train_medical_clip_llava_new_data/config/run_myvlm_inference.yaml



pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__)"