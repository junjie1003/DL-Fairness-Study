#!/bin/bash

# 指定要删除__pycache__文件夹的目录路径
directory="/root/DL-Fairness-Study"

# 使用find命令查找所有__pycache__文件夹，并删除它们
find "$directory" -type d -name "__pycache__" -exec rm -r {} +
