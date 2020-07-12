# DifNet

![Deep Diffusive Neural Network Model Architecture](./result/framework.png)

## Deep Diffusive Neural Network on Graph Semi-Supervised Classification

Source code of "Get Rid of Suspended Animation Problem: Deep Diffusive Neural Network on Graph Semi-Supervised Classification". </br>
Paper arXiv URL: https://arxiv.org/abs/2001.07922

### Two other papers are helpful for readers to follow the ideas in this paper and the code

(1) FAKEDETECTOR: Effective Fake News Detection with Deep Diffusive Neural Network https://arxiv.org/abs/1805.08751

(2) Graph Neural Lasso for Dynamic Network Regression https://arxiv.org/abs/1907.11114

### References

```
@article{Zhang2020GetRO,
  title={Get Rid of Suspended Animation Problem: Deep Diffusive Neural Network on Graph Semi-Supervised Classification},
  author={Jiawei Zhang},
  journal={ArXiv},
  year={2020},
  volume={abs/2001.07922}
}
```

************************************************************************************************

## How to run the code?

### To run a script, you can just use command line: python3 script_name.py

After downloading the code, you can run
```
python3 script.py
```
directly for node classification. 

### How to turn on/off the blocks?

You can change the "if 0" to "if 1" to turn on a script block, and the reverse to turn off a script block.

### Several toolkits may be needed to run the code
(1) pytorch (https://anaconda.org/pytorch/pytorch)
(2) sklearn (https://anaconda.org/anaconda/scikit-learn) 
(3) transformers (https://anaconda.org/conda-forge/transformers) 

************************************************************************************************

## Organization of the code?

A simpler template of the code is also available at http://www.ifmlab.org/files/template/IFM_Lab_Program_Template_Python3.zip

### The whole program is divided into five main parts:

(1) data.py (for data loading and basic data organization operators, defines abstract method load() )

(2) method.py (for complex operations on the data, defines abstract method run() )

(3) result.py (for saving/loading results from files, defines abstract method load() and save() )

(4) evaluate.py (for result evaluation, defines abstract method evaluate() )

(5) setting.py (for experiment settings, defines abstract method load_run_save_evaluate() )

The other inherited class will implement these abstract methonds.
