#!/bin/bash

# 灵猫墨韵系统 - 统一日志和临时文件清理脚本
# 生成时间: $(date '+%Y-%m-%d %H:%M:%S')

# 配置参数
LOGS_DIR="logs"               # 日志主目录
KEEP_LOGS=10                  # 每个目录保留的最新日志文件数量
MAX_LOG_SIZE=50               # 日志最大大小(MB)
MIN_TEMP_SIZE=100             # 临时文件最小大小(MB)
BACKUP_BEFORE_DELETE=true     # 删除前是否备份重要文件

# 创建日志目录结构
echo "正在整理日志目录结构..."
mkdir -p logs/processor logs/tokenizer logs/train_model logs/generate logs/data

# 移动日志文件到对应子目录
echo "正在移动日志文件到对应子目录..."
find logs -maxdepth 1 -name "processor_*.log" -exec mv {} logs/processor/ \;
find logs -maxdepth 1 -name "tokenizer_*.log" -exec mv {} logs/tokenizer/ \;
find logs -maxdepth 1 -name "train_*.log" -exec mv {} logs/train_model/ \;
find logs -maxdepth 1 -name "model_*.log" -exec mv {} logs/train_model/ \;
find logs -maxdepth 1 -name "generate_*.log" -exec mv {} logs/generate/ \;
find logs -maxdepth 1 -name "gen_*.log" -exec mv {} logs/generate/ \;
find logs -maxdepth 1 -name "data_*.log" -exec mv {} logs/data/ \;

# 清理大型日志文件
echo "正在清理大型日志文件..."
for dir in logs/processor logs/tokenizer logs/train_model logs/generate logs/data; do
  if [ -d "$dir" ]; then
    # 删除超过大小限制的日志
    find "$dir" -name "*.log" -size +${MAX_LOG_SIZE}M -exec rm {} \; -exec echo "已删除大型日志: {}" \;
    
    # 统计日志数量
    LOG_COUNT=$(find "$dir" -name "*.log" | wc -l)
    
    # 如果日志数量超过保留数量，删除最老的日志
    if [ "$LOG_COUNT" -gt "$KEEP_LOGS" ]; then
      EXTRA_LOGS=$((LOG_COUNT - KEEP_LOGS))
      echo "删除 $dir 中的 $EXTRA_LOGS 个旧日志文件..."
      find "$dir" -name "*.log" -printf "%T@ %p\n" | sort -n | head -n $EXTRA_LOGS | cut -d' ' -f2- | xargs rm
    fi
  fi
done

# 清理Python缓存
echo "正在清理Python缓存文件..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete

# 清理大型临时文件
echo "正在清理大型临时文件..."
find . -name "temp_*.*" -size +${MIN_TEMP_SIZE}M -exec rm {} \; -exec echo "已删除临时文件: {}" \;
find . -name "*.tmp" -size +${MIN_TEMP_SIZE}M -exec rm {} \; -exec echo "已删除临时文件: {}" \;
find . -name "*.bak" -size +${MIN_TEMP_SIZE}M -exec rm {} \; -exec echo "已删除临时文件: {}" \;

# 删除temp_corpus.txt大文件
if [ -f "temp_corpus.txt" ]; then
  echo "删除大型临时语料文件: temp_corpus.txt"
  rm temp_corpus.txt
fi

echo "清理完成!"
