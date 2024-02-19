# Rethinking CNN’s Generalization to Backdoor Attack from Frequency Domain
This is the official implementation code of the paper [Rethinking CNN’s Generalization to Backdoor Attack from Frequency Domain](https://openreview.net/forum?id=mYhH0CDFFa) published on ICLR2024.
> Convolutional neural network (CNN) is easily affected by backdoor injections, whose models perform normally on clean samples but produce specific outputs on poisoned ones. Most of the existing studies have focused on the effect of trigger feature changes of poisoned samples on model generalization in spatial domain. We focus on the mechanism of CNN memorize poisoned samples in frequency domain, and find that CNN generate generalization to poisoned samples by memorizing the frequency domain distribution of trigger changes. We also explore the influence of trigger perturbations in different frequency domain components on the generalization of poisoned models from visible and invisible backdoor attacks, and prove that high-frequency components are more susceptible to perturbations than low-frequency components. Based on the above fundings, we propose a universal invisible strategy for visible triggers, which can achieve trigger invisibility while maintaining raw attack performance. We also design a novel frequency domain backdoor attack method based on low-frequency semantic information, which can achieve 100% attack accuracy on multiple models and multiple datasets, and can bypass multiple defenses.

## Requirements
- Install required python packages:
```bash
$ python -m pip install -r requirements.py
```

## Training
Run command 
```bash
$ python frequence_attack_train.py --dataset <datasetName> --dctsize <TriggerRawSize> --intensity <TriggerIntensity>
```

Usage
```bash
python frequence_attack_train.py --dataset "cifar10" --dctsize 1/3 --intensity 0.4
```
The trained checkpoints should be saved at the path `checkpoints\<datasetName>\<datasetName>_all2one_morph.pth.tar`.

We provide the poisoned model weight file for the CIFAR-10 dataset.


## Evaluation 
For evaluating trained models, run command
```bash
$ python frequence_attack_eval.py --dataset <datasetName> --dctsize <TriggerRawSize> --intensity <TriggerIntensity>
```
Usage
```bash
python frequence_attack_eval.py --dataset "cifar10" --dctsize 1/3 --intensity 0.4
```

## Reference
If you find this repo useful for your research, please consider citing our paper
```
@inproceedings{
rao2024rethinking,
title={Rethinking {CNN}{\textquoteright}s Generalization to Backdoor Attack from Frequency Domain},
author={Quanrui Rao and Lin Wang and Wuying Liu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=mYhH0CDFFa}
}
```

