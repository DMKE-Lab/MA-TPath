## MA-TPath
MA-TPath: Multi-Hop Temporal knowledge Graph Reasoning with Mulit-Agent Reinforcement Learning
This repository contains the implementation of the MA-TPath architectures described in the paper.
## Installation
* Install Tensorflow (>= 1.1.0)
* Python 3.x (tested on Python 3.6)
* Numpy
* Pandas
* Scikit-learn
* tqdm
## How to use?
After installing the requirements, run the following command to reproduce results for MA-TPath:
```
$ python trainer.py --base_output_dir output/{dataset-name} --path_length {path-length>1} --hidden_size {hidden_size} --embedding_size {embedding_size} --batch_size {batch_size} --beta 0.05 --Lambda 0.05 --use_entity_embeddings 1 --train_entity_embeddings 1 --train_relation_embeddings 1 --train_tim_embeddings 1 --data_input_dir {datasets-dir} --vocab_dir {datasets-dir-vocab} --model_load_dir null --load_model 0 --total_iterations {total_iterations} --nell_evaluation 0
```
## Data Format
To run MA-TPath on a custom graph based dataset, you would need the graph and the queries as 4-triple in the form of (e1,r, e2,tim). Where e1, and e2 are nodes connected by the edge r. The vocab can of the dataset can be created using the create_vocab.py file found in data/data preprocessing scripts. The vocab needs to be stores in the json format {'entity/relation/tim': ID}. 
[link](https://github.com/shehzaadzd/MINERVA)
## Baselines
| Baselines                           | Code                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| TransE  / TransH                    | [link](https://github.com/jimmywangheng/knowledge_representation_pytorch) |
| DistMult                            | [link](https://github.com/tranhungnghiep/AnalyzeKGE)         |
| MINERVA                             | [link](https://github.com/shehzaadzd/MINERVA)                |
| TTransE                             | [link](https://github.com/INK-USC/RE-Net)                    |
| HyTE                                | [link](https://github.com/malllabiisc/HyTE)                  |
| TA-TransE / TA-DistMult             | [link](https://github.com/INK-USC/RE-Net)                    |
| DE-TransE / DE-DistMult / DE-SimplE | [link](https://github.com/BorealisAI/DE-SimplE)              |
| RE-NET                              | [link](https://github.com/INK-USC/RE-Net)                    |
| TPmod                               | [link](https://github.com/DMKE-Lab/TPmod)                    |
