# this is the journey of trying to make clip and llava (medical) work - try 1

```
first i created folder to make everything more organized and to make sure we dont miss any step of the way
```
## Upper folder structure
```
training_myvlm/
├── train_basic_clip_blip
├── train_basic_clip_llava
├── train_medical_clip_basic_blip
├── train_medical_clip_basic_llava
├── train_medical_clip_blip
├── train_medical_clip_llava
└── journey_omer.md
```
## TODO list

* ~~create a todo list~~
* Experiment 1: running clip and blip (basic models that the article were using)
    - gather test data (cat)
    - build data folder structure
    - create config files
    - log every command and change to run this. - tmux history
    - maybe create a script to run everything
* Experiment 1: running clip and llava (basic models that the article were using)




```
as i procced with the training, i will take photos of the procedure and save results in a excel/somthing else to later view the results.
```

## Building the test data
```
i will take the data from the cat test we did in the beggining of the project, it will be reliable and make a good test data for the basic llava, to make sure it can do VQA. then proceed with adding the medical VLMs'.
```

### saving the bash commands i run
```bash
touch ~/.tmux.conf - not workign - tried to save bash history.
```

## reverting the changes that we made before
```
changes the model back to hf-hub:apple/DFN5B-CLIP-ViT-H-14-384
changes can be search with hf-hub:apple/DFN5B-CLIP-ViT-H-14-384
or '# omer'
now will try to run step 1
```
```bash
cd ~/MyVLM

python concept_heads/clip/concept_head_training/train.py \
--config_path /home/user_7734/training_myvlm/train_basic_clip_blip/config/concept_head_training.yaml
```

```error
1024 and 512 error (known)
to fix need to change back to 1024. search '# omer' in file /home/user_7734/MyVLM/concept_heads/clip/concept_head_training/model.py

lets try again
```
## log 1 - after running first script
####  this log comes from the code, <output_dir>/<concept_name> is the source of the log

```log
Step: 10 | Loss: 0.65283203125
Step: 20 | Loss: 0.6142578125
Step: 30 | Loss: 0.56640625
Step: 40 | Loss: 0.533203125
Step: 50 | Loss: 0.46826171875
Test | Step: 50 | Positive Accuracy: 100.0
Test | Step: 50 | Negative Accuracy: 96.0
Positive probabilities: [0.6113, 0.597, 0.594, 0.589, 0.5786, 0.6016, 0.601]
Average negative positive probabilities: 0.427978515625
Max negative positive probabilities: 0.515625
Step: 60 | Loss: 0.45068359375
Step: 70 | Loss: 0.432861328125
Step: 80 | Loss: 0.42138671875
Step: 90 | Loss: 0.39111328125
Step: 100 | Loss: 0.345703125
Test | Step: 100 | Positive Accuracy: 100.0
Test | Step: 100 | Negative Accuracy: 100.0
Positive probabilities: [0.6865, 0.656, 0.6504, 0.6455, 0.626, 0.661, 0.658]
Average negative positive probabilities: 0.342041015625
Max negative positive probabilities: 0.498779296875
Step: 110 | Loss: 0.35009765625
Step: 120 | Loss: 0.31494140625
Step: 130 | Loss: 0.30126953125
Step: 140 | Loss: 0.29736328125
Step: 150 | Loss: 0.29541015625
Test | Step: 150 | Positive Accuracy: 100.0
Test | Step: 150 | Negative Accuracy: 100.0
Positive probabilities: [0.741, 0.696, 0.692, 0.692, 0.665, 0.7075, 0.7]
Average negative positive probabilities: 0.274658203125
Max negative positive probabilities: 0.487060546875
Step: 160 | Loss: 0.2381591796875
Step: 170 | Loss: 0.2266845703125
Step: 180 | Loss: 0.235107421875
Step: 190 | Loss: 0.2193603515625
Step: 200 | Loss: 0.2205810546875
Test | Step: 200 | Positive Accuracy: 100.0
Test | Step: 200 | Negative Accuracy: 100.0
Positive probabilities: [0.777, 0.726, 0.722, 0.7217, 0.6875, 0.74, 0.7305]
Average negative positive probabilities: 0.2264404296875
Max negative positive probabilities: 0.466796875
Step: 210 | Loss: 0.221923828125
Step: 220 | Loss: 0.17236328125
Step: 230 | Loss: 0.18017578125
Step: 240 | Loss: 0.179931640625
Step: 250 | Loss: 0.1759033203125
Test | Step: 250 | Positive Accuracy: 100.0
Test | Step: 250 | Negative Accuracy: 100.0
Positive probabilities: [0.7974, 0.744, 0.7393, 0.743, 0.701, 0.7607, 0.751]
Average negative positive probabilities: 0.1953125
Max negative positive probabilities: 0.444580078125
Step: 260 | Loss: 0.155517578125
Step: 270 | Loss: 0.17236328125
Step: 280 | Loss: 0.1591796875
Step: 290 | Loss: 0.15380859375
Step: 300 | Loss: 0.148681640625
Test | Step: 300 | Positive Accuracy: 100.0
Test | Step: 300 | Negative Accuracy: 100.0
Positive probabilities: [0.8027, 0.749, 0.744, 0.7515, 0.7007, 0.7695, 0.759]
Average negative positive probabilities: 0.171142578125
Max negative positive probabilities: 0.41943359375
Step: 310 | Loss: 0.166259765625
Step: 320 | Loss: 0.1387939453125
Step: 330 | Loss: 0.1439208984375
Step: 340 | Loss: 0.1380615234375
Step: 350 | Loss: 0.1329345703125
Test | Step: 350 | Positive Accuracy: 100.0
Test | Step: 350 | Negative Accuracy: 100.0
Positive probabilities: [0.823, 0.7715, 0.7656, 0.772, 0.723, 0.791, 0.7812]
Average negative positive probabilities: 0.1702880859375
Max negative positive probabilities: 0.430419921875
Step: 360 | Loss: 0.152099609375
Step: 370 | Loss: 0.1104736328125
Step: 380 | Loss: 0.12420654296875
Step: 390 | Loss: 0.1533203125
Step: 400 | Loss: 0.1473388671875
Test | Step: 400 | Positive Accuracy: 100.0
Test | Step: 400 | Negative Accuracy: 100.0
Positive probabilities: [0.826, 0.7734, 0.768, 0.7754, 0.7246, 0.794, 0.784]
Average negative positive probabilities: 0.1632080078125
Max negative positive probabilities: 0.423095703125
Step: 410 | Loss: 0.1329345703125
Step: 420 | Loss: 0.145263671875
Step: 430 | Loss: 0.136474609375
Step: 440 | Loss: 0.138427734375
Step: 450 | Loss: 0.13037109375
Test | Step: 450 | Positive Accuracy: 100.0
Test | Step: 450 | Negative Accuracy: 100.0
Positive probabilities: [0.828, 0.7754, 0.77, 0.7773, 0.726, 0.796, 0.786]
Average negative positive probabilities: 0.1612548828125
Max negative positive probabilities: 0.422119140625
Step: 460 | Loss: 0.12274169921875
Step: 470 | Loss: 0.1505126953125
Step: 480 | Loss: 0.1571044921875
Step: 490 | Loss: 0.1474609375
Step: 500 | Loss: 0.12060546875
Test | Step: 500 | Positive Accuracy: 100.0
Test | Step: 500 | Negative Accuracy: 100.0
Positive probabilities: [0.829, 0.7764, 0.7705, 0.7783, 0.727, 0.797, 0.787]
Average negative positive probabilities: 0.1610107421875
Max negative positive probabilities: 0.42236328125



```

## concept embedding training
now after we ran 500 steps we will run the concept embedding training
BEFORE THAT, we need to copy the last step of the run (500)
to new folder. we will call it trained_concept_head

/training_myvlm/train_basic_clip_blip/trained_concept_head/my_cat/seed_42/

```bash
python concept_embedding_training/train.py \
--config_path /home/user_7734/training_myvlm/train_basic_clip_blip/config/concept_embedding_training_captioning.yaml
```

### error 2
```
Traceback (most recent call last):
  File "concept_embedding_training/train.py", line 139, in <module>
    main()
  File "/home/user_7734/anaconda3/envs/myvlm/lib/python3.8/site-packages/pyrallis/argparsing.py", line 158, in wrapper_inner
    response = fn(cfg, *args, **kwargs)
  File "concept_embedding_training/train.py", line 44, in main
    vlm_wrapper = VLM_TYPE_TO_WRAPPER[cfg.vlm_type](device=cfg.device, torch_dtype=cfg.torch_dtype)
  File "/home/user_7734/MyVLM/vlms/blip2_wrapper.py", line 15, in __init__
    super().__init__(device, torch_dtype)
  File "/home/user_7734/MyVLM/vlms/vlm_wrapper.py", line 21, in __init__
    self.model, self.processor = self.set_model()
  File "/home/user_7734/MyVLM/vlms/blip2_wrapper.py", line 18, in set_model
    processor = Blip2Processor.from_pretrained(self.model_path)
  File "/home/user_7734/anaconda3/envs/myvlm/lib/python3.8/site-packages/transformers/processing_utils.py", line 839, in from_pretrained
    return cls.from_args_and_dict(args, processor_dict, **kwargs)
  File "/home/user_7734/anaconda3/envs/myvlm/lib/python3.8/site-packages/transformers/processing_utils.py", line 659, in from_args_and_dict
    processor = cls(*args, **processor_dict)
TypeError: __init__() got an unexpected keyword argument 'num_query_tokens'
```
### this is the model of blip its using(probably)
```
Salesforce/blip2-flan-t5-xl

/home/user_7734/MyVLM/myvlm/common.py in here changed back to original clip
```
**interesting link:**
https://huggingface.co/models?other=blip&sort=trending&search=Salesforce%2Fblip

i try to download the git of MyVLM again, lets see all of the problems and maybe diff it.
i try to look at forks of MyVLM, maybe i will find the key to make it work!



## important, maybe the solution
```code
conda create --name <env_name> --file requirements.txt
```
instead of:
```code
conda env create -f environment/environment.yaml
```


```
after reinstalling conda env, called now myvlm_on
i tried to run with the requirements that is provided (checked if the versions are as in the requirements.txt)
4.37.2 is the transformers version that was in the req txt
but i see online that it is different
```

#possible breakthrough
```
i reliezed that conda might causing problems, when i tried to run the command
```
```bash
conda install --file environment/requirements.txt
```
i recived an error regarding channels


pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
