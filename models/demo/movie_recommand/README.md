# PaddleRec 基于 Movielens 数据集的全流程示例

## 模型的详细教程可以查阅： [告别电影荒，手把手教你训练符合自己口味的私人电影推荐助手](https://aistudio.baidu.com/aistudio/projectdetail/1481839)

## 本地运行环境

PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : linux 

本地运行流程与AiStudio流程基本一致，细节略有区别

## 数据处理
```shell
pip install py27hash
bash data_prepare.sh
```
```shell
# 动态图训练无moe rank模型
USING_MOE=False python -u ../../../tools/trainer.py -m rank/config.yaml
# 动态图训练带moe的rank模型
USING_MOE=True EXPERT_COUNT=32 python -u ../../../tools/static_trainer.py -m rank/config.yaml
```
