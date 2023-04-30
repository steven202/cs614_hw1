# Evaluating a model performance for [CIFAR10-C](https://arxiv.org/abs/1807.01697) using [CLIP zero-shot learning](https://arxiv.org/abs/2103.00020) (PyTorch)

The code is based on [openai CLIP repo](https://github.com/openai/CLIP) and [CIFAR-10-C repo](https://github.com/tanimutomo/cifar10-c-eval).

## Preparation
### Install nessecery packages
```
conda env create -f environment.yml
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```
### Download data
Download CIFAR10-C dataset from [here](https://zenodo.org/record/2535967#.XncuG5P7TUJ).  

## Evaluation
```
python main.py
```

### Output 
All corruption accuracy.  
This figure will be saved in `Accuracy vs Corruption.png`.

### (Optional) Other Useful Options
- `data_root` : Specify the directory path that contains CIFAR-10 and CIFAR-10-C datasets folders.
