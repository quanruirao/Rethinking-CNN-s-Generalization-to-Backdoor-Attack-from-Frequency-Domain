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
