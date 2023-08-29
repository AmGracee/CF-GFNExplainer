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

>Syn1=BA-shapes, syn4=BA-Cycles, syn1=BA-Grids

#### For graph classification

```python
python train_graph.py --dataset=mutag
```
> We save the trained GNN models parameters to the models folder.


## Training CF-GFNExplainer

```train
python main_explainer_node.py --dataset=syn1 --num_iterations=500 --replay_capacity=4800
python main_explainer_graph.py --dataset=mutag --num_iterations=500 --replay_capacity=4800
```

>It will create another folder in the main directory called 'results', where the results files will be stored.


## Evaluation

To evaluate the CF examples, run the following command:

```eval
python evaluate_node.py --path=../results/<NAME OF RESULTS FILE>
python evaluate_graph.py --path=../results/<NAME OF RESULTS FILE>
```
