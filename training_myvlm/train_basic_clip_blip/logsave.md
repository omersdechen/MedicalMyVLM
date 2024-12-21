Step: 500 | Loss: 0.12103271484375
Running validation...
100%|████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.91it/s]
Test | Step: 500 | Positive Accuracy: 100.0
Test | Step: 500 | Negative Accuracy: 100.0
(myvlm_fix) user_7734@sipl-gpu1:~/MyVLM$ cd ..
(myvlm_fix) user_7734@sipl-gpu1:~$ cd training_myvlm/
(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ ls
cat_statue       train_basic_clip_blip   train_medical_clip_basic_blip   train_medical_clip_blip   try_train_my_cat
journey_omer.md  train_basic_clip_llava  train_medical_clip_basic_llava  train_medical_clip_llava
(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ python /home/user_7734/MyVLM/concept_heads/clip/concept_head_training/train.py \
--config_path /home/user_7734/training_myvlm/train_basic_clip_blip/config/concept_head_training.yaml
/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
Trainable parameters:
classifier.weight
classifier.bias
Number of positive train images: 4
Number of negative train images: 75
Number of positive val images: 7
Number of negative val images: 75
Step: 10 | Loss: 0.65283203125
Step: 20 | Loss: 0.61376953125
Step: 30 | Loss: 0.56494140625
Step: 40 | Loss: 0.53271484375
Step: 50 | Loss: 0.4677734375
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.87it/s]
Test | Step: 50 | Positive Accuracy: 100.0
Test | Step: 50 | Negative Accuracy: 97.33333333333333
Step: 60 | Loss: 0.449462890625
Step: 70 | Loss: 0.432861328125
Step: 80 | Loss: 0.42138671875
Step: 90 | Loss: 0.3896484375
Step: 100 | Loss: 0.345947265625
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.90it/s]
Test | Step: 100 | Positive Accuracy: 100.0
Test | Step: 100 | Negative Accuracy: 100.0
Step: 110 | Loss: 0.349609375
Step: 120 | Loss: 0.314697265625
Step: 130 | Loss: 0.30126953125
Step: 140 | Loss: 0.2978515625
Step: 150 | Loss: 0.29296875
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.89it/s]
Test | Step: 150 | Positive Accuracy: 100.0
Test | Step: 150 | Negative Accuracy: 100.0
Step: 160 | Loss: 0.2388916015625
Step: 170 | Loss: 0.227294921875
Step: 180 | Loss: 0.2364501953125
Step: 190 | Loss: 0.2188720703125
Step: 200 | Loss: 0.2183837890625
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.89it/s]
Test | Step: 200 | Positive Accuracy: 100.0
Test | Step: 200 | Negative Accuracy: 100.0
Step: 210 | Loss: 0.2215576171875
Step: 220 | Loss: 0.172607421875
Step: 230 | Loss: 0.1793212890625
Step: 240 | Loss: 0.1790771484375
Step: 250 | Loss: 0.1729736328125
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.89it/s]
Test | Step: 250 | Positive Accuracy: 100.0
Test | Step: 250 | Negative Accuracy: 100.0
Step: 260 | Loss: 0.1573486328125
Step: 270 | Loss: 0.1734619140625
Step: 280 | Loss: 0.157958984375
Step: 290 | Loss: 0.1558837890625
Step: 300 | Loss: 0.149658203125
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.89it/s]
Test | Step: 300 | Positive Accuracy: 100.0
Test | Step: 300 | Negative Accuracy: 100.0
Step: 310 | Loss: 0.1634521484375
Step: 320 | Loss: 0.139404296875
Step: 330 | Loss: 0.1431884765625
Step: 340 | Loss: 0.1376953125
Step: 350 | Loss: 0.130615234375
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.87it/s]
Test | Step: 350 | Positive Accuracy: 100.0
Test | Step: 350 | Negative Accuracy: 100.0
Step: 360 | Loss: 0.152587890625
Step: 370 | Loss: 0.10992431640625
Step: 380 | Loss: 0.1229248046875
Step: 390 | Loss: 0.1524658203125
Step: 400 | Loss: 0.1466064453125
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.89it/s]
Test | Step: 400 | Positive Accuracy: 100.0
Test | Step: 400 | Negative Accuracy: 100.0
Step: 410 | Loss: 0.1341552734375
Step: 420 | Loss: 0.1444091796875
Step: 430 | Loss: 0.13623046875
Step: 440 | Loss: 0.13916015625
Step: 450 | Loss: 0.1318359375
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.88it/s]
Test | Step: 450 | Positive Accuracy: 100.0
Test | Step: 450 | Negative Accuracy: 100.0
Step: 460 | Loss: 0.123779296875
Step: 470 | Loss: 0.151123046875
Step: 480 | Loss: 0.1568603515625
Step: 490 | Loss: 0.1463623046875
Step: 500 | Loss: 0.12103271484375
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.87it/s]
Test | Step: 500 | Positive Accuracy: 100.0
Test | Step: 500 | Negative Accuracy: 100.0
(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ cp /home/user_7734/training_myvlm/train_basic_clip_blip/output_concept_head_training/my_cat/DFN5B-CLIP-ViT-H-14-384-my_cat-step-500.pt \
/home/user_7734/training_myvlm/train_basic_clip_blip/trained_concept_head/my_cat/seed_42
(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ python /home/user_7734/MyVLM/concept_embedding_training/train.py \
--config_path /home/user_7734/training_myvlm/train_basic_clip_blip/config/concept_embedding_training_captioning.yaml

batch_size: 4
classifier_step: 500
concept_head_path: /home/user_7734/training_myvlm/train_basic_clip_blip/trained_concept_head/my_cat/seed_42
concept_identifier: sks
concept_name: my_cat
concept_type: OBJECT
data_root: /home/user_7734/training_myvlm/train_basic_clip_blip/data/pos_cat
device: cuda
learning_rate: 1.0
optimization_steps: 100
output_root: /home/user_7734/training_myvlm/train_basic_clip_blip/output_concept_embedding_training
personalization_task: !!python/object/apply:myvlm.common.PersonalizationTask
- captioning
reg_lambda: 0.04
save_interval: 25
seed: 42
threshold: 0.5
torch_dtype: torch.bfloat16
val_interval: 25
vlm_type: !!python/object/apply:myvlm.common.VLMType
- blip-2

Traceback (most recent call last):
  File "/home/user_7734/MyVLM/concept_embedding_training/train.py", line 139, in <module>
    main()
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/pyrallis/argparsing.py", line 158, in wrapper_inner
    response = fn(cfg, *args, **kwargs)
  File "/home/user_7734/MyVLM/concept_embedding_training/train.py", line 44, in main
    vlm_wrapper = VLM_TYPE_TO_WRAPPER[cfg.vlm_type](device=cfg.device, torch_dtype=cfg.torch_dtype)
  File "/home/user_7734/MyVLM/vlms/blip2_wrapper.py", line 15, in __init__
    super().__init__(device, torch_dtype)
  File "/home/user_7734/MyVLM/vlms/vlm_wrapper.py", line 21, in __init__
    self.model, self.processor = self.set_model()
  File "/home/user_7734/MyVLM/vlms/blip2_wrapper.py", line 18, in set_model
    processor = Blip2Processor.from_pretrained(self.model_path)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/processing_utils.py", line 466, in from_pretrained
    args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/processing_utils.py", line 512, in _get_arguments_from_pretrained
    args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/models/auto/tokenization_auto.py", line 814, in from_pretrained
    return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/tokenization_utils_base.py", line 2029, in from_pretrained
    return cls._from_pretrained(
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/tokenization_utils_base.py", line 2261, in _from_pretrained
    tokenizer = cls(*init_inputs, **init_kwargs)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/models/t5/tokenization_t5_fast.py", line 135, in __init__
    super().__init__(
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/tokenization_utils_fast.py", line 111, in __init__
    fast_tokenizer = TokenizerFast.from_file(fast_tokenizer_file)
Exception: data did not match any variant of untagged enum PyPreTokenizerTypeWrapper at line 964 column 3


(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ rm -rf ~/.cache/huggingface/



(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ python /home/user_7734/MyVLM/concept_embedding_training/train.py --config_path /home/user_7734/training_myvlm/train_basic_clip_blip/config/concept_embedding_training_captioning.yaml

batch_size: 4
classifier_step: 500
concept_head_path: /home/user_7734/training_myvlm/train_basic_clip_blip/trained_concept_head/my_cat/seed_42
concept_identifier: sks
concept_name: my_cat
concept_type: OBJECT
data_root: /home/user_7734/training_myvlm/train_basic_clip_blip/data/pos_cat
device: cuda
learning_rate: 1.0
optimization_steps: 100
output_root: /home/user_7734/training_myvlm/train_basic_clip_blip/output_concept_embedding_training
personalization_task: !!python/object/apply:myvlm.common.PersonalizationTask
- captioning
reg_lambda: 0.04
save_interval: 25
seed: 42
threshold: 0.5
torch_dtype: torch.bfloat16
val_interval: 25
vlm_type: !!python/object/apply:myvlm.common.VLMType
- blip-2

preprocessor_config.json: 100%|███████████████████████████████████████████████████████████████████████████| 432/432 [00:00<00:00, 112kB/s]
tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████| 21.0k/21.0k [00:00<00:00, 1.68MB/s]
spiece.model: 100%|████████████████████████████████████████████████████████████████████████████████████| 792k/792k [00:00<00:00, 3.80MB/s]
tokenizer.json: 100%|████████████████████████████████████████████████████████████████████████████████| 2.42M/2.42M [00:00<00:00, 5.25MB/s]
added_tokens.json: 100%|███████████████████████████████████████████████████████████████████████████████| 23.0/23.0 [00:00<00:00, 12.0kB/s]
special_tokens_map.json: 100%|███████████████████████████████████████████████████████████████████████| 2.54k/2.54k [00:00<00:00, 1.31MB/s]
Traceback (most recent call last):
  File "/home/user_7734/MyVLM/concept_embedding_training/train.py", line 139, in <module>
    main()
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/pyrallis/argparsing.py", line 158, in wrapper_inner
    response = fn(cfg, *args, **kwargs)
  File "/home/user_7734/MyVLM/concept_embedding_training/train.py", line 44, in main
    vlm_wrapper = VLM_TYPE_TO_WRAPPER[cfg.vlm_type](device=cfg.device, torch_dtype=cfg.torch_dtype)
  File "/home/user_7734/MyVLM/vlms/blip2_wrapper.py", line 15, in __init__
    super().__init__(device, torch_dtype)
  File "/home/user_7734/MyVLM/vlms/vlm_wrapper.py", line 21, in __init__
    self.model, self.processor = self.set_model()
  File "/home/user_7734/MyVLM/vlms/blip2_wrapper.py", line 18, in set_model
    processor = Blip2Processor.from_pretrained(self.model_path)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/processing_utils.py", line 466, in from_pretrained
    args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/processing_utils.py", line 512, in _get_arguments_from_pretrained
    args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/models/auto/tokenization_auto.py", line 814, in from_pretrained
    return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/tokenization_utils_base.py", line 2029, in from_pretrained
    return cls._from_pretrained(
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/tokenization_utils_base.py", line 2261, in _from_pretrained
    tokenizer = cls(*init_inputs, **init_kwargs)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/models/t5/tokenization_t5_fast.py", line 135, in __init__
    super().__init__(
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/tokenization_utils_fast.py", line 111, in __init__
    fast_tokenizer = TokenizerFast.from_file(fast_tokenizer_file)
Exception: data did not match any variant of untagged enum PyPreTokenizerTypeWrapper at line 964 column 3
(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ pip install --upgrade transformers tokenizers
Requirement already satisfied: transformers in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (4.37.2)
Requirement already satisfied: tokenizers in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (0.15.2)
Collecting tokenizers
  Downloading tokenizers-0.21.0.tar.gz (343 kB)
     |████████████████████████████████| 343 kB 1.1 MB/s 
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
    Preparing wheel metadata ... error
    ERROR: Command errored out with exit status 1:
     command: /home/user_7734/anaconda3/envs/myvlm_fix/bin/python /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/pip/_vendor/pep517/_in_process.py prepare_metadata_for_build_wheel /tmp/tmp68625gik
         cwd: /tmp/pip-install-nmiopucn/tokenizers_d1f2dd7b4a2d43d282794d650e1019f8
    Complete output (6 lines):
    
    Cargo, the Rust package manager, is not installed or is not on PATH.
    This package requires Rust and Cargo to compile extensions. Install it through
    the system's package manager or via https://rustup.rs/
    
    Checking for Rust toolchain....
    ----------------------------------------
ERROR: Command errored out with exit status 1: /home/user_7734/anaconda3/envs/myvlm_fix/bin/python /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/pip/_vendor/pep517/_in_process.py prepare_metadata_for_build_wheel /tmp/tmp68625gik Check the logs for full command output.
(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ python /home/user_7734/MyVLM/concept_heads/clip/concept_head_training/train.py --config_path /home/user_7734/training_myvlm/train_basic_clip_blip/config/concept_head_training.yaml
open_clip_pytorch_model.bin: 100%|███████████████████████████████████████████████████████████████████| 3.95G/3.95G [01:34<00:00, 41.7MB/s]
open_clip_config.json: 100%|██████████████████████████████████████████████████████████████████████████████| 735/735 [00:00<00:00, 160kB/s]
/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
Trainable parameters:
classifier.weight
classifier.bias
Number of positive train images: 4
Number of negative train images: 75
Number of positive val images: 7
Number of negative val images: 75
Step: 10 | Loss: 0.65283203125
Step: 20 | Loss: 0.61376953125
Step: 30 | Loss: 0.56494140625
Step: 40 | Loss: 0.53271484375
Step: 50 | Loss: 0.4677734375
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.99it/s]
Test | Step: 50 | Positive Accuracy: 100.0
Test | Step: 50 | Negative Accuracy: 97.33333333333333
Step: 60 | Loss: 0.449462890625
Step: 70 | Loss: 0.432861328125
Step: 80 | Loss: 0.42138671875
Step: 90 | Loss: 0.3896484375
Step: 100 | Loss: 0.345947265625
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:01<00:00,  3.00it/s]
Test | Step: 100 | Positive Accuracy: 100.0
Test | Step: 100 | Negative Accuracy: 100.0
Step: 110 | Loss: 0.349609375
Step: 120 | Loss: 0.314697265625
Step: 130 | Loss: 0.30126953125
Step: 140 | Loss: 0.2978515625
Step: 150 | Loss: 0.29296875
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.99it/s]
Test | Step: 150 | Positive Accuracy: 100.0
Test | Step: 150 | Negative Accuracy: 100.0
Step: 160 | Loss: 0.2388916015625
Step: 170 | Loss: 0.227294921875
Step: 180 | Loss: 0.2364501953125
Step: 190 | Loss: 0.2188720703125
Step: 200 | Loss: 0.2183837890625
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.96it/s]
Test | Step: 200 | Positive Accuracy: 100.0
Test | Step: 200 | Negative Accuracy: 100.0
Step: 210 | Loss: 0.2215576171875
Step: 220 | Loss: 0.172607421875
Step: 230 | Loss: 0.1793212890625
Step: 240 | Loss: 0.1790771484375
Step: 250 | Loss: 0.1729736328125
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.96it/s]
Test | Step: 250 | Positive Accuracy: 100.0
Test | Step: 250 | Negative Accuracy: 100.0
Step: 260 | Loss: 0.1573486328125
Step: 270 | Loss: 0.1734619140625
Step: 280 | Loss: 0.157958984375
Step: 290 | Loss: 0.1558837890625
Step: 300 | Loss: 0.149658203125
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.95it/s]
Test | Step: 300 | Positive Accuracy: 100.0
Test | Step: 300 | Negative Accuracy: 100.0
Step: 310 | Loss: 0.1634521484375
Step: 320 | Loss: 0.139404296875
Step: 330 | Loss: 0.1431884765625
Step: 340 | Loss: 0.1376953125
Step: 350 | Loss: 0.130615234375
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.95it/s]
Test | Step: 350 | Positive Accuracy: 100.0
Test | Step: 350 | Negative Accuracy: 100.0
Step: 360 | Loss: 0.152587890625
Step: 370 | Loss: 0.10992431640625
Step: 380 | Loss: 0.1229248046875
Step: 390 | Loss: 0.1524658203125
Step: 400 | Loss: 0.1466064453125
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.96it/s]
Test | Step: 400 | Positive Accuracy: 100.0
Test | Step: 400 | Negative Accuracy: 100.0
Step: 410 | Loss: 0.1341552734375
Step: 420 | Loss: 0.1444091796875
Step: 430 | Loss: 0.13623046875
Step: 440 | Loss: 0.13916015625
Step: 450 | Loss: 0.1318359375
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.96it/s]
Test | Step: 450 | Positive Accuracy: 100.0
Test | Step: 450 | Negative Accuracy: 100.0
Step: 460 | Loss: 0.123779296875
Step: 470 | Loss: 0.151123046875
Step: 480 | Loss: 0.1568603515625
Step: 490 | Loss: 0.1463623046875
Step: 500 | Loss: 0.12103271484375
Running validation...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:02<00:00,  2.92it/s]
Test | Step: 500 | Positive Accuracy: 100.0
Test | Step: 500 | Negative Accuracy: 100.0




(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ cp /home/user_7734/training_myvlm/train_basic_clip_blip/output_concept_head_training/my_cat/DFN5B-CLIP-ViT-H-14-384-my_cat-step-500.pt \
/home/user_7734/training_myvlm/train_basic_clip_blip/trained_concept_head/my_cat/seed_42


(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ python /home/user_7734/MyVLM/concept_embedding_training/train.py \
--config_path /home/user_7734/training_myvlm/train_basic_clip_blip/config/concept_embedding_training_captioning.yaml

batch_size: 4
classifier_step: 500
concept_head_path: /home/user_7734/training_myvlm/train_basic_clip_blip/trained_concept_head/my_cat/seed_42
concept_identifier: sks
concept_name: my_cat
concept_type: OBJECT
data_root: /home/user_7734/training_myvlm/train_basic_clip_blip/data/pos_cat
device: cuda
learning_rate: 1.0
optimization_steps: 100
output_root: /home/user_7734/training_myvlm/train_basic_clip_blip/output_concept_embedding_training
personalization_task: !!python/object/apply:myvlm.common.PersonalizationTask
- captioning
reg_lambda: 0.04
save_interval: 25
seed: 42
threshold: 0.5
torch_dtype: torch.bfloat16
val_interval: 25
vlm_type: !!python/object/apply:myvlm.common.VLMType
- blip-2

Traceback (most recent call last):
  File "/home/user_7734/MyVLM/concept_embedding_training/train.py", line 139, in <module>
    main()
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/pyrallis/argparsing.py", line 158, in wrapper_inner
    response = fn(cfg, *args, **kwargs)
  File "/home/user_7734/MyVLM/concept_embedding_training/train.py", line 44, in main
    vlm_wrapper = VLM_TYPE_TO_WRAPPER[cfg.vlm_type](device=cfg.device, torch_dtype=cfg.torch_dtype)
  File "/home/user_7734/MyVLM/vlms/blip2_wrapper.py", line 15, in __init__
    super().__init__(device, torch_dtype)
  File "/home/user_7734/MyVLM/vlms/vlm_wrapper.py", line 21, in __init__
    self.model, self.processor = self.set_model()
  File "/home/user_7734/MyVLM/vlms/blip2_wrapper.py", line 18, in set_model
    processor = Blip2Processor.from_pretrained(self.model_path)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/processing_utils.py", line 466, in from_pretrained
    args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/processing_utils.py", line 512, in _get_arguments_from_pretrained
    args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/models/auto/tokenization_auto.py", line 814, in from_pretrained
    return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/tokenization_utils_base.py", line 2029, in from_pretrained
    return cls._from_pretrained(
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/tokenization_utils_base.py", line 2261, in _from_pretrained
    tokenizer = cls(*init_inputs, **init_kwargs)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/models/t5/tokenization_t5_fast.py", line 135, in __init__
    super().__init__(
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/tokenization_utils_fast.py", line 111, in __init__
    fast_tokenizer = TokenizerFast.from_file(fast_tokenizer_file)
Exception: data did not match any variant of untagged enum PyPreTokenizerTypeWrapper at line 964 column 3

(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ conda list | grep -i SentencePiece
sentencepiece             0.2.0                    pypi_0    pypi

(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ pip install pyrallis
Requirement already satisfied: pyrallis in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (0.3.1)
Requirement already satisfied: typing-inspect in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from pyrallis) (0.9.0)
Requirement already satisfied: pyyaml in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from pyrallis) (6.0)
Requirement already satisfied: mypy-extensions>=0.3.0 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from typing-inspect->pyrallis) (1.0.0)
Requirement already satisfied: typing-extensions>=3.7.4 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from typing-inspect->pyrallis) (4.12.2)


(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ conda list | grep -i SentencePiece
sentencepiece             0.2.0                    pypi_0    pypi


(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ python /home/user_7734/MyVLM/concept_embedding_training/train.py --config_path /home/user_7734/training_myvlm/train_basic_clip_blip/config/concept_embedding_training_captioning.yaml

batch_size: 4
classifier_step: 500
concept_head_path: /home/user_7734/training_myvlm/train_basic_clip_blip/trained_concept_head/my_cat/seed_42
concept_identifier: sks
concept_name: my_cat
concept_type: OBJECT
data_root: /home/user_7734/training_myvlm/train_basic_clip_blip/data/pos_cat
device: cuda
learning_rate: 1.0
optimization_steps: 100
output_root: /home/user_7734/training_myvlm/train_basic_clip_blip/output_concept_embedding_training
personalization_task: !!python/object/apply:myvlm.common.PersonalizationTask
- captioning
reg_lambda: 0.04
save_interval: 25
seed: 42
threshold: 0.5
torch_dtype: torch.bfloat16
val_interval: 25
vlm_type: !!python/object/apply:myvlm.common.VLMType
- blip-2

Traceback (most recent call last):
  File "/home/user_7734/MyVLM/concept_embedding_training/train.py", line 139, in <module>
    main()
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/pyrallis/argparsing.py", line 158, in wrapper_inner
    response = fn(cfg, *args, **kwargs)
  File "/home/user_7734/MyVLM/concept_embedding_training/train.py", line 44, in main
    vlm_wrapper = VLM_TYPE_TO_WRAPPER[cfg.vlm_type](device=cfg.device, torch_dtype=cfg.torch_dtype)
  File "/home/user_7734/MyVLM/vlms/blip2_wrapper.py", line 15, in __init__
    super().__init__(device, torch_dtype)
  File "/home/user_7734/MyVLM/vlms/vlm_wrapper.py", line 21, in __init__
    self.model, self.processor = self.set_model()
  File "/home/user_7734/MyVLM/vlms/blip2_wrapper.py", line 18, in set_model
    processor = Blip2Processor.from_pretrained(self.model_path)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/processing_utils.py", line 466, in from_pretrained
    args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/processing_utils.py", line 512, in _get_arguments_from_pretrained
    args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/models/auto/tokenization_auto.py", line 814, in from_pretrained
    return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/tokenization_utils_base.py", line 2029, in from_pretrained
    return cls._from_pretrained(
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/tokenization_utils_base.py", line 2261, in _from_pretrained
    tokenizer = cls(*init_inputs, **init_kwargs)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/models/t5/tokenization_t5_fast.py", line 135, in __init__
    super().__init__(
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/tokenization_utils_fast.py", line 111, in __init__
    fast_tokenizer = TokenizerFast.from_file(fast_tokenizer_file)
Exception: data did not match any variant of untagged enum PyPreTokenizerTypeWrapper at line 964 column 3



(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ pip install -U tokenizers





Requirement already satisfied: tokenizers in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (0.15.2)
Collecting tokenizers
  Using cached tokenizers-0.21.0.tar.gz (343 kB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
    Preparing wheel metadata ... error
    ERROR: Command errored out with exit status 1:
     command: /home/user_7734/anaconda3/envs/myvlm_fix/bin/python /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/pip/_vendor/pep517/_in_process.py prepare_metadata_for_build_wheel /tmp/tmphgcbbj4b
         cwd: /tmp/pip-install-dc7i_y9e/tokenizers_1d88ac2c2e304c6785207b81bcb04475
    Complete output (6 lines):
    
    Cargo, the Rust package manager, is not installed or is not on PATH.
    This package requires Rust and Cargo to compile extensions. Install it through
    the system's package manager or via https://rustup.rs/
    
    Checking for Rust toolchain....
    ----------------------------------------
ERROR: Command errored out with exit status 1: /home/user_7734/anaconda3/envs/myvlm_fix/bin/python /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/pip/_vendor/pep517/_in_process.py prepare_metadata_for_build_wheel /tmp/tmphgcbbj4b Check the logs for full command output.




(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ pip install --no-cache-dir tokenizers==0.15.2
Requirement already satisfied: tokenizers==0.15.2 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (0.15.2)
Requirement already satisfied: huggingface_hub<1.0,>=0.16.4 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from tokenizers==0.15.2) (0.19.4)
Requirement already satisfied: tqdm>=4.42.1 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from huggingface_hub<1.0,>=0.16.4->tokenizers==0.15.2) (4.64.1)
Requirement already satisfied: fsspec>=2023.5.0 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from huggingface_hub<1.0,>=0.16.4->tokenizers==0.15.2) (2024.10.0)
Requirement already satisfied: typing-extensions>=3.7.4.3 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from huggingface_hub<1.0,>=0.16.4->tokenizers==0.15.2) (4.12.2)
Requirement already satisfied: packaging>=20.9 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from huggingface_hub<1.0,>=0.16.4->tokenizers==0.15.2) (24.2)
Requirement already satisfied: filelock in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from huggingface_hub<1.0,>=0.16.4->tokenizers==0.15.2) (3.16.1)
Requirement already satisfied: requests in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from huggingface_hub<1.0,>=0.16.4->tokenizers==0.15.2) (2.31.0)
Requirement already satisfied: pyyaml>=5.1 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from huggingface_hub<1.0,>=0.16.4->tokenizers==0.15.2) (6.0)
Requirement already satisfied: idna<4,>=2.5 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from requests->huggingface_hub<1.0,>=0.16.4->tokenizers==0.15.2) (3.10)
Requirement already satisfied: charset-normalizer<4,>=2 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from requests->huggingface_hub<1.0,>=0.16.4->tokenizers==0.15.2) (3.4.0)
Requirement already satisfied: certifi>=2017.4.17 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from requests->huggingface_hub<1.0,>=0.16.4->tokenizers==0.15.2) (2024.12.14)
Requirement already satisfied: urllib3<3,>=1.21.1 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from requests->huggingface_hub<1.0,>=0.16.4->tokenizers==0.15.2) (2.2.3)



(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ python /home/user_7734/MyVLM/concept_embedding_training/train.py --config_path /home/user_7734/training_myvlm/train_basic_clip_blip/config/concept_embedding_training_captioning.yaml

batch_size: 4
classifier_step: 500
concept_head_path: /home/user_7734/training_myvlm/train_basic_clip_blip/trained_concept_head/my_cat/seed_42
concept_identifier: sks
concept_name: my_cat
concept_type: OBJECT
data_root: /home/user_7734/training_myvlm/train_basic_clip_blip/data/pos_cat
device: cuda
learning_rate: 1.0
optimization_steps: 100
output_root: /home/user_7734/training_myvlm/train_basic_clip_blip/output_concept_embedding_training
personalization_task: !!python/object/apply:myvlm.common.PersonalizationTask
- captioning
reg_lambda: 0.04
save_interval: 25
seed: 42
threshold: 0.5
torch_dtype: torch.bfloat16
val_interval: 25
vlm_type: !!python/object/apply:myvlm.common.VLMType
- blip-2

Traceback (most recent call last):
  File "/home/user_7734/MyVLM/concept_embedding_training/train.py", line 139, in <module>
    main()
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/pyrallis/argparsing.py", line 158, in wrapper_inner
    response = fn(cfg, *args, **kwargs)
  File "/home/user_7734/MyVLM/concept_embedding_training/train.py", line 44, in main
    vlm_wrapper = VLM_TYPE_TO_WRAPPER[cfg.vlm_type](device=cfg.device, torch_dtype=cfg.torch_dtype)
  File "/home/user_7734/MyVLM/vlms/blip2_wrapper.py", line 15, in __init__
    super().__init__(device, torch_dtype)
  File "/home/user_7734/MyVLM/vlms/vlm_wrapper.py", line 21, in __init__
    self.model, self.processor = self.set_model()
  File "/home/user_7734/MyVLM/vlms/blip2_wrapper.py", line 18, in set_model
    processor = Blip2Processor.from_pretrained(self.model_path)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/processing_utils.py", line 466, in from_pretrained
    args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/processing_utils.py", line 512, in _get_arguments_from_pretrained
    args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/models/auto/tokenization_auto.py", line 814, in from_pretrained
    return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/tokenization_utils_base.py", line 2029, in from_pretrained
    return cls._from_pretrained(
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/tokenization_utils_base.py", line 2261, in _from_pretrained
    tokenizer = cls(*init_inputs, **init_kwargs)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/models/t5/tokenization_t5_fast.py", line 135, in __init__
    super().__init__(
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/tokenization_utils_fast.py", line 111, in __init__
    fast_tokenizer = TokenizerFast.from_file(fast_tokenizer_file)
Exception: data did not match any variant of untagged enum PyPreTokenizerTypeWrapper at line 964 column 3



(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ pip install --upgrade transformers
Requirement already satisfied: transformers in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (4.37.2)
Collecting transformers
  Downloading transformers-4.46.3-py3-none-any.whl (10.0 MB)
     |████████████████████████████████| 10.0 MB 1.1 MB/s 
Requirement already satisfied: tqdm>=4.27 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from transformers) (4.64.1)
Requirement already satisfied: numpy>=1.17 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from transformers) (1.24.1)
Requirement already satisfied: requests in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from transformers) (2.31.0)
Requirement already satisfied: regex!=2019.12.17 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from transformers) (2024.11.6)
Requirement already satisfied: filelock in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from transformers) (3.16.1)
Requirement already satisfied: pyyaml>=5.1 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from transformers) (6.0)
Requirement already satisfied: safetensors>=0.4.1 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from transformers) (0.4.5)
Requirement already satisfied: packaging>=20.0 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from transformers) (24.2)
Collecting huggingface-hub<1.0,>=0.23.2
  Downloading huggingface_hub-0.27.0-py3-none-any.whl (450 kB)
     |████████████████████████████████| 450 kB 148.0 MB/s 
Requirement already satisfied: typing-extensions>=3.7.4.3 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from huggingface-hub<1.0,>=0.23.2->transformers) (4.12.2)
Requirement already satisfied: fsspec>=2023.5.0 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from huggingface-hub<1.0,>=0.23.2->transformers) (2024.10.0)
Collecting tokenizers<0.21,>=0.20
  Downloading tokenizers-0.20.3-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.0 MB)
     |████████████████████████████████| 3.0 MB 88.4 MB/s 
Requirement already satisfied: charset-normalizer<4,>=2 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from requests->transformers) (3.4.0)
Requirement already satisfied: urllib3<3,>=1.21.1 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from requests->transformers) (2.2.3)
Requirement already satisfied: certifi>=2017.4.17 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from requests->transformers) (2024.12.14)
Requirement already satisfied: idna<4,>=2.5 in /home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages (from requests->transformers) (3.10)
Installing collected packages: huggingface-hub, tokenizers, transformers
  Attempting uninstall: huggingface-hub
    Found existing installation: huggingface-hub 0.19.4
    Uninstalling huggingface-hub-0.19.4:
      Successfully uninstalled huggingface-hub-0.19.4
  Attempting uninstall: tokenizers
    Found existing installation: tokenizers 0.15.2
    Uninstalling tokenizers-0.15.2:
      Successfully uninstalled tokenizers-0.15.2
  Attempting uninstall: transformers
    Found existing installation: transformers 4.37.2
    Uninstalling transformers-4.37.2:
      Successfully uninstalled transformers-4.37.2
Successfully installed huggingface-hub-0.27.0 tokenizers-0.20.3 transformers-4.46.3



(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ python /home/user_7734/MyVLM/concept_embedding_training/train.py --config_path /home/user_7734/training_myvlm/train_basic_clip_blip/config/concept_embedding_training_captioning.yaml
/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/torchvision/datapoints/__init__.py:12: UserWarning: The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta. While we do not expect major breaking changes, some APIs may still change according to user feedback. Please submit any feedback you may have in this issue: https://github.com/pytorch/vision/issues/6753, and you can also check out https://github.com/pytorch/vision/issues/7319 to learn more about the APIs that we suspect might involve future changes. You can silence this warning by calling torchvision.disable_beta_transforms_warning().
  warnings.warn(_BETA_TRANSFORMS_WARNING)
/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/torchvision/transforms/v2/__init__.py:54: UserWarning: The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta. While we do not expect major breaking changes, some APIs may still change according to user feedback. Please submit any feedback you may have in this issue: https://github.com/pytorch/vision/issues/6753, and you can also check out https://github.com/pytorch/vision/issues/7319 to learn more about the APIs that we suspect might involve future changes. You can silence this warning by calling torchvision.disable_beta_transforms_warning().
  warnings.warn(_BETA_TRANSFORMS_WARNING)

batch_size: 4
classifier_step: 500
concept_head_path: /home/user_7734/training_myvlm/train_basic_clip_blip/trained_concept_head/my_cat/seed_42
concept_identifier: sks
concept_name: my_cat
concept_type: OBJECT
data_root: /home/user_7734/training_myvlm/train_basic_clip_blip/data/pos_cat
device: cuda
learning_rate: 1.0
optimization_steps: 100
output_root: /home/user_7734/training_myvlm/train_basic_clip_blip/output_concept_embedding_training
personalization_task: !!python/object/apply:myvlm.common.PersonalizationTask
- captioning
reg_lambda: 0.04
save_interval: 25
seed: 42
threshold: 0.5
torch_dtype: torch.bfloat16
val_interval: 25
vlm_type: !!python/object/apply:myvlm.common.VLMType
- blip-2

processor_config.json: 100%|███████████████████████████████████████████████████████████████████████████| 68.0/68.0 [00:00<00:00, 5.26kB/s]
config.json: 100%|████████████████████████████████████████████████████████████████████████████████████| 2.22k/2.22k [00:00<00:00, 404kB/s]
model.safetensors.index.json: 100%|████████████████████████████████████████████████████████████████████| 128k/128k [00:00<00:00, 1.39MB/s]
model-00001-of-00002.safetensors: 100%|██████████████████████████████████████████████████████████████| 9.96G/9.96G [02:03<00:00, 80.3MB/s]
model-00002-of-00002.safetensors: 100%|██████████████████████████████████████████████████████████████| 5.81G/5.81G [01:12<00:00, 80.2MB/s]
Downloading shards: 100%|███████████████████████████████████████████████████████████████████████████████████| 2/2 [03:17<00:00, 98.62s/it]
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.06s/it]
generation_config.json: 100%|█████████████████████████████████████████████████████████████████████████████| 168/168 [00:00<00:00, 206kB/s]
/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 10.52it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 47.14it/s]
Image: my_cat_6 | Original VLM Answer: tee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee 
Image: my_cat_1 | Original VLM Answer: tee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee 
Image: my_cat_10 | Original VLM Answer: tee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee 
Image: my_cat_11 | Original VLM Answer: tee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee 
  0%|                                                                                                             | 0/100 [00:00<?, ?it/s]Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.48.0. You should pass an instance of `EncoderDecoderCache` instead, e.g. `past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`.
  0%|                                                                                                             | 0/100 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/home/user_7734/MyVLM/concept_embedding_training/train.py", line 139, in <module>
    main()
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/pyrallis/argparsing.py", line 158, in wrapper_inner
    response = fn(cfg, *args, **kwargs)
  File "/home/user_7734/MyVLM/concept_embedding_training/train.py", line 68, in main
    concept_embedding_checkpoints = train_concept_embedding(myvlm=myvlm,
  File "/home/user_7734/MyVLM/concept_embedding_training/train.py", line 119, in train_concept_embedding
    concept_embedding_checkpoints = myvlm.train_embedding(
  File "/home/user_7734/MyVLM/myvlm/myvlm.py", line 69, in train_embedding
    outputs = self.vlm.model(**batch)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/user_7734/MyVLM/vlms/blip2/modeling_blip_2.py", line 1834, in forward
    outputs = self.language_model(
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/models/t5/modeling_t5.py", line 1891, in forward
    decoder_outputs = self.decoder(
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/models/t5/modeling_t5.py", line 1044, in forward
    causal_mask = self._update_causal_mask(
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/models/t5/modeling_t5.py", line 1240, in _update_causal_mask
    causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
  File "/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/transformers/models/t5/modeling_t5.py", line 1307, in _prepare_4d_causal_attention_mask_with_cache_position
    causal_mask = torch.triu(causal_mask, diagonal=1)
RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16'



(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ python /home/user_7734/MyVLM/concept_embedding_training/train.py --config_path /home/user_7734/training_myvlm/train_basic_clip_blip/config/concept_embedding_training_captioning.yaml
/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/torchvision/datapoints/__init__.py:12: UserWarning: The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta. While we do not expect major breaking changes, some APIs may still change according to user feedback. Please submit any feedback you may have in this issue: https://github.com/pytorch/vision/issues/6753, and you can also check out https://github.com/pytorch/vision/issues/7319 to learn more about the APIs that we suspect might involve future changes. You can silence this warning by calling torchvision.disable_beta_transforms_warning().
  warnings.warn(_BETA_TRANSFORMS_WARNING)
/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/torchvision/transforms/v2/__init__.py:54: UserWarning: The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta. While we do not expect major breaking changes, some APIs may still change according to user feedback. Please submit any feedback you may have in this issue: https://github.com/pytorch/vision/issues/6753, and you can also check out https://github.com/pytorch/vision/issues/7319 to learn more about the APIs that we suspect might involve future changes. You can silence this warning by calling torchvision.disable_beta_transforms_warning().
  warnings.warn(_BETA_TRANSFORMS_WARNING)
Traceback (most recent call last):
  File "/home/user_7734/MyVLM/concept_embedding_training/train.py", line 14, in <module>
    from inference.run_myvlm_inference import run_inference
  File "/home/user_7734/MyVLM/inference/run_myvlm_inference.py", line 16, in <module>
    from vlms.blip2_wrapper import BLIP2Wrapper
  File "/home/user_7734/MyVLM/vlms/blip2_wrapper.py", line 11, in <module>
    class BLIP2Wrapper(VLMWrapper):
  File "/home/user_7734/MyVLM/vlms/blip2_wrapper.py", line 13, in BLIP2Wrapper
    def __init__(self, device: str = 'cuda', torch_dtype: torch.dtype = torch.bfloat32): # omer
AttributeError: module 'torch' has no attribute 'bfloat32'


(myvlm_fix) user_7734@sipl-gpu1:~/training_myvlm$ python /home/user_7734/MyVLM/concept_embedding_training/train.py --config_path /home/user_7734/training_myvlm/train_basic_clip_blip/config/concept_embedding_training_captioning.yaml
/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/torchvision/datapoints/__init__.py:12: UserWarning: The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta. While we do not expect major breaking changes, some APIs may still change according to user feedback. Please submit any feedback you may have in this issue: https://github.com/pytorch/vision/issues/6753, and you can also check out https://github.com/pytorch/vision/issues/7319 to learn more about the APIs that we suspect might involve future changes. You can silence this warning by calling torchvision.disable_beta_transforms_warning().
  warnings.warn(_BETA_TRANSFORMS_WARNING)
/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/torchvision/transforms/v2/__init__.py:54: UserWarning: The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta. While we do not expect major breaking changes, some APIs may still change according to user feedback. Please submit any feedback you may have in this issue: https://github.com/pytorch/vision/issues/6753, and you can also check out https://github.com/pytorch/vision/issues/7319 to learn more about the APIs that we suspect might involve future changes. You can silence this warning by calling torchvision.disable_beta_transforms_warning().
  warnings.warn(_BETA_TRANSFORMS_WARNING)

batch_size: 4
classifier_step: 500
concept_head_path: /home/user_7734/training_myvlm/train_basic_clip_blip/trained_concept_head/my_cat/seed_42
concept_identifier: sks
concept_name: my_cat
concept_type: OBJECT
data_root: /home/user_7734/training_myvlm/train_basic_clip_blip/data/pos_cat
device: cuda
learning_rate: 1.0
optimization_steps: 100
output_root: /home/user_7734/training_myvlm/train_basic_clip_blip/output_concept_embedding_training
personalization_task: !!python/object/apply:myvlm.common.PersonalizationTask
- captioning
reg_lambda: 0.04
save_interval: 25
seed: 42
threshold: 0.5
torch_dtype: torch.float32
val_interval: 25
vlm_type: !!python/object/apply:myvlm.common.VLMType
- blip-2

Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.25s/it]
/home/user_7734/anaconda3/envs/myvlm_fix/lib/python3.8/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  9.64it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 43.46it/s]
Image: my_cat_6 | Original VLM Answer: tee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee 
Image: my_cat_1 | Original VLM Answer: tee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee 
Image: my_cat_10 | Original VLM Answer: tee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee 
Image: my_cat_11 | Original VLM Answer: tee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee hee 
  0%|                                                                                                             | 0/100 [00:00<?, ?it/s]Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.48.0. You should pass an instance of `EncoderDecoderCache` instead, e.g. `past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`.
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  4.11it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  4.28it/s]
Personalized Output: sks laying on a white kitchen counter
Personalized Output: sks laying on a white kitchen counter with a white kettle and a white keyboard
Personalized Output: sks in front of a window
Personalized Output: sks on a white kitchen table
----------------------------------------------------------------------------------------------------
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  4.09it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  4.08it/s]
Personalized Output: sks on a kitchen counter
Personalized Output: sks resting on a kitchen counter near a white kettle
Personalized Output: sks in front of a window, with a white tag on their collar
Personalized Output: sks on a white kitchen counter
----------------------------------------------------------------------------------------------------
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  4.04it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  4.12it/s]
Personalized Output: sks sitting on a white kitchen counter
Personalized Output: sks on a kitchen counter with a white kettle and a white keyboard
Personalized Output: sks on a window sill, reflecting in the glass
Personalized Output: sks on a white kitchen counter
----------------------------------------------------------------------------------------------------
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  3.91it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  4.00it/s]
Personalized Output: sks seated on a kitchen counter
Personalized Output: sks laying on a kitchen counter with a white kettle and a white keyboard
Personalized Output: sks on a window ledge, reflected in the glass
Personalized Output: sks on a white kitchen counter
----------------------------------------------------------------------------------------------------
Loss: 1.036 | Reg Loss: 0.075: 100%|████████████████████████████████████████████████████████████████████| 100/100 [00:44<00:00,  2.23it/s]
********************************************************************************
Finished concept_embedding_training concept embedding!
********************************************************************************
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.083 | Threshold: 0.5
Image: my_cat_6 | Personalized Answer: sks seated on a kitchen counter
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.205 | Threshold: 0.5
Image: my_cat_1 | Personalized Answer: sks laying on a kitchen counter with a white kettle and a white keyboard
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.287 | Threshold: 0.5
Image: my_cat_10 | Personalized Answer: sks on a window ledge, reflected in the glass
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.179 | Threshold: 0.5
Image: my_cat_11 | Personalized Answer: sks on a white kitchen counter
----------------------------------------------------------------------------------------------------
****************************************************************************************************
RUNNING INFERENCE
****************************************************************************************************
####################################################################################################
Running on iteration: 25
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.092 | Threshold: 0.5
my_cat_4 | Input:  | Output: sks on the stairwell
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.194 | Threshold: 0.5
my_cat_5 | Input:  | Output: sks sitting on a couch
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.214 | Threshold: 0.5
my_cat_3 | Input:  | Output: sks playing with a pink toy on a blanket
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.246 | Threshold: 0.5
my_cat_9 | Input:  | Output: sks laying on a chair with a few balls on it
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.087 | Threshold: 0.5
my_cat_2 | Input:  | Output: sks laying on a white dish rack in front of a window
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.108 | Threshold: 0.5
my_cat_7 | Input:  | Output: sks laying on a blue blanket on a couch
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.186 | Threshold: 0.5
my_cat_8 | Input:  | Output: sks laying on a couch with a blanket
----------------------------------------------------------------------------------------------------
####################################################################################################
####################################################################################################
Running on iteration: 50
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.092 | Threshold: 0.5
my_cat_4 | Input:  | Output: sks peering over a staircase
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.194 | Threshold: 0.5
my_cat_5 | Input:  | Output: sks sitting on a beige couch
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.214 | Threshold: 0.5
my_cat_3 | Input:  | Output: sks playing with a pink toy on a white blanket
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.246 | Threshold: 0.5
my_cat_9 | Input:  | Output: sks laying on a yellow and orange patterned chair with a few coloured balls
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.087 | Threshold: 0.5
my_cat_2 | Input:  | Output: sks laying on a white scrubbing mat in a kitchen
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.108 | Threshold: 0.5
my_cat_7 | Input:  | Output: sks laying on a blue blanket in a living room
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.186 | Threshold: 0.5
my_cat_8 | Input:  | Output: sks laying on a couch under a brown blanket
----------------------------------------------------------------------------------------------------
####################################################################################################
####################################################################################################
Running on iteration: 75
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.092 | Threshold: 0.5
my_cat_4 | Input:  | Output: sks peering over a white stairway
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.194 | Threshold: 0.5
my_cat_5 | Input:  | Output: sks sitting on a beige upholstered sofa in a living room
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.214 | Threshold: 0.5
my_cat_3 | Input:  | Output: sks on a white blanket
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.246 | Threshold: 0.5
my_cat_9 | Input:  | Output: sks laying on a yellow and white patterned upholstered chair with a few coloured balls
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.087 | Threshold: 0.5
my_cat_2 | Input:  | Output: sks laying on a white squeegee sprayed with white paint on a kitchen counter
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.108 | Threshold: 0.5
my_cat_7 | Input:  | Output: sks laying on a blue blanket on a white kitchen floor
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.186 | Threshold: 0.5
my_cat_8 | Input:  | Output: sks reeding on a brown blanket on a sofa
----------------------------------------------------------------------------------------------------
####################################################################################################
####################################################################################################
Running on iteration: 99
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.092 | Threshold: 0.5
my_cat_4 | Input:  | Output: sks peering over a stairway
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.194 | Threshold: 0.5
my_cat_5 | Input:  | Output: sks seated on a beige upholstered rood apron
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.214 | Threshold: 0.5
my_cat_3 | Input:  | Output: sks playing with a pink toy on a white blanket
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.246 | Threshold: 0.5
my_cat_9 | Input:  | Output: sks laying on a patterned upholstered chair with a few coloured balls
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.087 | Threshold: 0.5
my_cat_2 | Input:  | Output: sks laying on a white scrubbing mat in a kitchen
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.108 | Threshold: 0.5
my_cat_7 | Input:  | Output: sks laying on a blue blanket on a bare stairwell
----------------------------------------------------------------------------------------------------
Adding concept: 0. Distance: 0.186 | Threshold: 0.5
my_cat_8 | Input:  | Output: sks laying under a brown blanket on a rood
----------------------------------------------------------------------------------------------------
####################################################################################################