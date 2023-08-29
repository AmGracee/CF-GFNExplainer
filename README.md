# CF-GFNExplainer

Code for  "Learning Counterfactual Explanation of Graph Neural Networks via Generative Flow Network"

## Requirements

To install requirements:

```setup
conda env create --file environment.yml
```

> This will create a conda environment called pytorch-pyg

## Training original GNN model

#### For node classification

```python
python train_node.py --dataset=syn1
```

>Syn1=BA-shapes
>
>Syn4=BA-Cycles
>
>Syn1=BA-Grids

#### For graph classification

```python
python train_graph.py --dataset=mutag
```



## Training CF-GFNExplainer

```train
python main_explainer_node.py --dataset=syn1
python main_explainer_graph.py --dataset=mutag
```

>It will create another folder in the main directory called 'results', where the results files will be stored.


## Evaluation

To evaluate the CF examples, run the following command:

```eval
python evaluate.py --path=../results/<NAME OF RESULTS FILE>
```
