# trying to train a concept head an embed and all
```bash
python concept_heads/clip/concept_head_training/train.py \
--config_path /home/user_7734/omer/try_train_my_cat/config/concept_head_training.yaml
```
# added this because got error ModuleNotFoundError: No module named 'concept_heads'
```bash
export PYTHONPATH=$PYTHONPATH:/home/user_7734/MyVLM
```


## no such file...
```bash
python concept_heads/clip/concept_head_training/train.py \
--config_path /home/user_7734/omer/try_train_my_cat/config/concept_head_training.yaml
```
dont know why but
```bash
pip install pyrallis
```

# FEW NOTES
config is very important to understand.
config.py is the key to solve config problems. understand it and then solve.
```yaml
concept_name: my_cat #the is the concept name. used much and make sure to keep in order
output_dir: /home/user_7734/omer/try_train_my_cat/output #
positive_samples_path: /home/user_7734/omer/try_train_my_cat/pos_cat #this is the path for the positive samples, the script try to go to this path and then to the dir 'concept_name' (the actual concept name not the varivle in the config)
negative_samples_path:  /home/user_7734/omer/try_train_my_cat/neg_cat
model_name: hf-hub:apple/DFN5B-CLIP-ViT-H-14-384
n_positive_samples: 4
max_steps: 500
seed: 42  
```
# update
got it running. the negetive samples need to have 150 pictures and the positive 4 (at least) to train the concept head.


# useful stuff for making the data for the trainning
i use power rename and regex to search.
.* is for selecting every file
${} is for making the images has number that is easy to arrange



# trying to concept embedding training
```bash
python concept_embedding_training/train.py \
--config_path /home/user_7734/omer/try_train_my_cat/config/concept_embedding_training_captioning.yaml
```

got errors but i think i got it this time
i used for training the seed 42
so concept_head_path should be the path to the .pt file
and the folder should look like
my_cat/seed_42/DFN5B-CLIP-ViT-H-14-384-my_cat-step-500.pt


# infrence!!!!
```bash
python inference/run_myvlm_inference.py \
--config_path /home/user_7734/omer/try_train_my_cat/config/myvlm_inference.yaml
```
