# M3Builder
The official codes for "M^3Builder: A Multi-Agent System for Automated Machine Learning in Medical Imaging"

[ArXiv Version](https://arxiv.org/abs/2502.20301)

In this paper, we present **M^3Builder**, an agentic system for automating machine learning in medical imaging tasks. Our approach combines an efficient medical imaging ML workspace with free-text descriptions of datasets, code templates, and interaction tools. Additionally, we propsoe a multi-agent collaborative agent system designed specifically for AI model building, with 4 role-playing LLMs, Task Manager, Data Engineer, Module Architect, and Model Trainer. In benchmarking against 5 SOTA agentic systems across 14 radiology task-specific datasets, M3Builder achieves a 94.29% model building success rate with Claude3.7-Sonnet standing out among 7 SOTA LLMs.

![teaser](https://github.com/user-attachments/assets/c7d8474f-2ecd-4177-ac76-6ebfe68c6238)

## System
Overview of our system **M3Builder**. From user’s free-text request to model delivery. The system integrates user requirements, a workspace with candidate data, tools, and code templates. A network of 4 specialized collaborative agents performs task analysis, data engineering, module assembling and training execution. A sample log tracks the Model Trainer agent’s activities during diagnosis model development.

![pipeline](https://github.com/user-attachments/assets/c99c9265-1faa-4028-b6e0-080e564a67a7)

## Setup

### Enviroment
To install the python environments:
```
pip install -r requirements.txt
```

### Data Preparation

* Create a folder named `ExternalDataset` locally.
* Put your custom dataset folder into `ExternalDataset`.
* We suggest you remove training/testing-irrelevant files from your dataset folder to avoid interference!

Your `ExternalDataset` folder should be like:
```
ExternalDataset
|   Dataset_1
|   |    images
|   |    masks
|   |    labels.csv
|   Dataset_2
|   |    Class1
|   |    Class2
|   |    labels.json
|   ......
```

## To Run **M^3Builder** on Your Dataset
After environment setup and data preparation, you should first check all the files, and replace all 'path/to/sth' into your own paths.
Then, edit the `human_requirements` parameter in `run.sh` to your own requirements, and run:
```
./run.sh
```
Training logs and checkpoints will be placed under `TrainPipeline/Logout'.

## Benchmark
Task Completion Performance Across LLMs. Each experiment undergoes multi-runs, with results shown as successful completions over total rounds (a/b format). Green cells indicate that all runs passed, Yellow indicates partially passed, and Red indicates that all runs failed.

<div style="text-align: center">
<img src="https://github.com/user-attachments/assets/3fa6315e-ea97-4fe9-b68b-68b5813d06f6"/>
</div>

## Comparison
Framework Comparison with SOTAs and Ablations on System Design using Sonnet. Results are averaged over two runs per task in dataset-level. “w/o Colab” represents single-agent execution, and “Iters” means the self-correction rounds.

![tab2](https://github.com/user-attachments/assets/859a79e2-a38e-4483-b083-882c82d789ed)

## Acknowledgement
We sincerely thank all the contributors who developed relevant codes in our repository.

## Citation
```
@article{feng2025m3,
          title={M^3Builder: A Multi-Agent System for Automated Machine Learning in Medical Imaging},
          author={Feng, Jinghao and Zheng, Qiaoyu and Wu, Chaoyi and Zhao, Ziheng and Zhang, 
            Ya and Wang, Yanfeng and Xie, Weidi},
          journal={arXiv preprint arXiv:2502.20301},
          year={2025}
}
```

