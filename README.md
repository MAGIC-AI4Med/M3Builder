# M3Builder
The official codes for "M^3Builder: A Multi-Agent System for Automated Machine Learning in Medical Imaging"

[ArXiv Version](https://arxiv.org/abs/2502.20301)

In this paper, we present M3Builder, an agentic system for automating machine learning in medical imaging tasks. Our approach combines an efficient medical imaging ML workspace with free-text descriptions of datasets, code templates, and interaction tools. Additionally, we propsoe a multi-agent collaborative agent system designed specifically for AI model building, with 4 role-playing LLMs, Task Manager, Data Engineer, Module Architect, and Model Trainer. In benchmarking against 5 SOTA agentic systems across 14 radiology task-specific datasets, M3Builder achieves a 94.29% model building success rate with Claude3.7-Sonnet standing out among 7 SOTA LLMs.

![teaser](https://github.com/user-attachments/assets/c7d8474f-2ecd-4177-ac76-6ebfe68c6238)

## System
Overview of our system **M3Builder**. From user’s free-text request to model delivery. The system integrates user requirements, a workspace with candidate data, tools, and code templates. A network of 4 specialized collaborative agents performs task analysis, data engineering, module assembling and training execution. A sample log tracks the Model Trainer agent’s activities during diagnosis model development.

![pipeline](https://github.com/user-attachments/assets/c99c9265-1faa-4028-b6e0-080e564a67a7)

## Setup

### Enviroment
