#!/bin/bash

# 进入当前目录（确保脚本从正确的位置运行）
# cd "$(dirname "$0")" || exit

# 检查 AgentPipelines/run.py 是否存在
if [ -f "AgentPipelines/run.py" ]; then
    echo "Running AgentPipelines/run.py..."
    python3 AgentPipelines/run.py --human_requirements "I want to train a model for disease diagnosis using 3D CT images."
else
    echo "Error: AgentPipelines/run.py not found!"
    exit 1
fi