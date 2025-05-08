# Personalized Subgraph Federated Learning via Differentiable Auxiliary Projections

## Requirement

- Python 3.9.16
- PyTorch 2.0.1
- PyTorch Geometric 2.3.0
- METIS (for data generation), https://github.com/james77777778/metis_python

**Train the model**:

```
python main.py --norm-scale 10 --n-dims 256 --dropout 0.7 --sigma 10 --model fedpub_aux
```

## Run our pretrained model to reproduce the results in the paper

Run ``analysis10.ipynb``

## Running Example

![Running example](fig/run_exp.png)

## Acknowledgement

This project draws in part from [FED-PUB](https://github.com/JinheonBaek/FED-PUB). Thanks for their great work and code!
