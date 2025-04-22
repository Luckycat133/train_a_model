#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
灵猫墨韵系统清理脚本
用于定期清理大型临时文件、日志文件和不必要的数据文件，保持项目体积合理
同时提供日志管理、文件整理和空间分析功能
"""

import os
import sys
import time
import shutil
import argparse
import json
import re
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Set, Optional
import fnmatch

# 设置日志记录
# 设置文件日志 - 保留完整信息
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True, parents=True)
log_file = log_dir / f"cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 配置基础日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)

# 获取日志记录器
logger = logging.getLogger("cleanup")

# 添加控制台处理器 - 仅显示消息内容
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console_handler)

# 安全文件列表 - 这些文件/目录不会被删除
PROTECTED_FILES = {
    # 核心Python文件
    "train_model.py", "generate.py", "processor.py", "tokenizer.py", 
    # 核心配置和文档
    "config/config.yaml", "README.md", "DOCS.md", ".gitignore",
    # 关键目录（不会删除目录本身，但可能删除其中的文件）
    "docs", "changelog", "config", "processors", "scripts"
}

# 核心数据文件 - 默认不会删除，除非明确指定
CORE_DATA_FILES = {
    # 训练相关数据文件
    "dataset/train_data_train.jsonl", 
    "dataset/train_data_test.jsonl",
    "dataset/train_data_gen.jsonl",
    "tokenizer.json",
    # 其他重要数据文件
    "dataset/chapters.jsonl",  
    "dataset/poems.jsonl",
    "dataset/data.json",
    "dataset/metadata.json"
}

def format_size(size_bytes):
    """将字节数转换为人类可读的格式"""
    if size_bytes == 0:
        return "0B"
    size_names = ("B", "KB", "MB", "GB", "TB")
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    return f"{size_bytes:.2f} {size_names[i]}"

def get_dir_size(path):
    """获取目录的总大小"""
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def scan_large_files(min_size_mb=100, include_dirs=None, exclude_patterns=None):
    """扫描大文件并返回列表"""
    large_files = []
    
    if include_dirs is None:
        include_dirs = ["."]
    
    if exclude_patterns is None:
        exclude_patterns = []
    
    # 转换排除模式为正则表达式
    exclude_regex = [re.compile(fnmatch.translate(pattern)) for pattern in exclude_patterns]
    
    for directory in include_dirs:
        for root, dirs, files in os.walk(directory):
            # 检查是否应跳过此目录
            skip = False
            for pattern in exclude_regex:
                if pattern.match(root):
                    skip = True
                    break
            if skip:
                continue
                
            for file in files:
                file_path = os.path.join(root, file)
                
                # 检查是否应跳过此文件
                skip = False
                for pattern in exclude_regex:
                    if pattern.match(file_path):
                        skip = True
                        break
                if skip:
                    continue
                
                # 检查文件是否在保护列表中
                if file_path in PROTECTED_FILES or file in PROTECTED_FILES:
                    continue
                
                # 检查文件大小
                try:
                    size = os.path.getsize(file_path)
                    if size > min_size_mb * 1024 * 1024:
                        large_files.append((file_path, size))
                except (OSError, FileNotFoundError):
                    continue
    
    # 按大小降序排序
    large_files.sort(key=lambda x: x[1], reverse=True)
    return large_files

def is_safe_to_delete(file_path):
    """检查文件是否可以安全删除"""
    path_str = str(file_path)
    
    # 检查是否在保护列表中
    for protected in PROTECTED_FILES:
        if protected in path_str:
            return False
    
    # 检查是否为核心数据文件
    if path_str in CORE_DATA_FILES:
        return False
        
    # 安全检查：不删除python源文件，除非在__pycache__目录中
    if path_str.endswith('.py') and '__pycache__' not in path_str:
        return False
        
    # 不删除docs和changelog目录中的md文件
    if (path_str.endswith('.md') and 
        ('docs/' in path_str or 'changelog/' in path_str or 
         'docs\\' in path_str or 'changelog\\' in path_str)):
        return False
        
    # 不删除LICENSE文件
    if 'LICENSE' in path_str:
        return False
        
    return True

def cleanup_temp_files(min_size_mb=100, dry_run=False, force=False):
    """清理临时文件"""
    temp_patterns = [
        "temp_corpus.txt",
        "**/temp_*.*",
        "**/*.tmp",
        "**/*.bak",
        "**/~*",
        "**/*_backup_*"
    ]
    
    cleaned_size = 0
    count = 0
    
    for pattern in temp_patterns:
        pattern_path = Path(".")  # 从当前目录开始
        # 将glob模式拆分成路径和文件模式
        if "/" in pattern:
            path_parts = pattern.split("/")
            pattern_str = path_parts[-1]
            pattern_path = Path("/".join(path_parts[:-1]))
        else:
            pattern_str = pattern
            
        for file in pattern_path.glob(pattern_str):
            if not file.is_file():
                continue
                
            # 安全检查
            if not is_safe_to_delete(file) and not force:
                logger.warning(f"跳过保护文件: {file}")
                continue
                
            size = file.stat().st_size
            if size > min_size_mb * 1024 * 1024:  # 转换为字节
                logger.info(f"找到大型临时文件: {file} ({format_size(size)})")
                if not dry_run:
                    try:
                        file.unlink()
                        logger.info(f"已删除: {file}")
                        cleaned_size += size
                        count += 1
                    except Exception as e:
                        logger.error(f"删除 {file} 时出错: {e}")
    
    return count, cleaned_size

def organize_logs(dry_run=False):
    """整理日志文件 - 移动日志文件到对应的子目录"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        logger.info("创建日志目录")
        if not dry_run:
            logs_dir.mkdir(exist_ok=True)
    
    # 创建子目录
    subdirs = ["processor", "tokenizer", "train_model", "generate", "data"]
    for subdir in subdirs:
        subdir_path = logs_dir / subdir
        if not subdir_path.exists():
            logger.info(f"创建子目录: {subdir_path}")
            if not dry_run:
                subdir_path.mkdir(exist_ok=True)
    
    # 移动日志文件到对应子目录
    moved_count = 0
    patterns = {
        "processor": ["processor_*.log"],
        "tokenizer": ["tokenizer_*.log"],
        "train_model": ["train_*.log", "model_*.log"],
        "generate": ["generate_*.log", "gen_*.log"],
        "data": ["data_*.log", "dataset_*.log"]
    }
    
    for subdir, file_patterns in patterns.items():
        for pattern in file_patterns:
            for file_path in logs_dir.glob(pattern):
                if file_path.is_file() and file_path.parent == logs_dir:
                    target_path = logs_dir / subdir / file_path.name
                    logger.info(f"移动日志文件: {file_path.name} -> {subdir}/")
                    if not dry_run:
                        try:
                            file_path.rename(target_path)
                            moved_count += 1
                        except Exception as e:
                            logger.error(f"移动文件 {file_path} 时出错: {e}")
    
    return moved_count

def cleanup_large_logs(max_log_size_mb=50, log_keep_count=5, dry_run=False):
    """清理大型日志文件，只保留最近的几个"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        logger.info("日志目录不存在")
        return 0, 0
    
    cleaned_size = 0
    count = 0
    
    # 先整理日志文件
    organize_logs(dry_run)
    
    # 按子目录分组处理日志
    for subdir in logs_dir.glob("**/"):
        if not subdir.is_dir():
            continue
            
        log_files = list(subdir.glob("*.log"))
        # 先处理大型日志
        for log_file in log_files:
            size = log_file.stat().st_size
            if size > max_log_size_mb * 1024 * 1024:
                logger.info(f"找到大型日志文件: {log_file} ({format_size(size)})")
                if not dry_run:
                    try:
                        log_file.unlink()
                        logger.info(f"已删除: {log_file}")
                        cleaned_size += size
                        count += 1
                    except Exception as e:
                        logger.error(f"删除 {log_file} 时出错: {e}")
        
        # 然后只保留最近的几个日志文件
        log_files = list(subdir.glob("*.log"))
        if len(log_files) > log_keep_count:
            # 按修改时间排序
            log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            # 保留最新的几个
            for log_file in log_files[log_keep_count:]:
                size = log_file.stat().st_size
                logger.info(f"删除旧日志文件: {log_file} ({format_size(size)})")
                if not dry_run:
                    try:
                        log_file.unlink()
                        logger.info(f"已删除: {log_file}")
                        cleaned_size += size
                        count += 1
                    except Exception as e:
                        logger.error(f"删除 {log_file} 时出错: {e}")
    
    return count, cleaned_size

def cleanup_merged_data(dry_run=False, force=False):
    """清理已合并的数据文件（这些文件可以从原始文件重新生成）
    
    主要作用：
    - 保留必要的训练文件：train_data_train.jsonl, train_data_test.jsonl, train_data_gen.jsonl
    - 删除可重新生成的合并文件：train_data.jsonl
    - 保证模型训练、验证和评估所需的数据完整性
    """
    dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        logger.info("数据集目录不存在")
        return 0, 0
    
    # 需要保留的核心数据文件
    core_files = [
        "train_data_train.jsonl",  # 训练数据
        "train_data_test.jsonl",   # 测试/验证数据
        "train_data_gen.jsonl"     # 生成测试数据
    ]
    
    # 可以删除的合并文件
    merged_files = [
        "train_data.jsonl"         # 合并文件，可以从上述三个文件重新生成
    ]
    
    # 检查核心文件是否存在
    missing_files = []
    for core_file in core_files:
        if not (dataset_dir / core_file).exists():
            missing_files.append(core_file)
    
    if missing_files and not force:
        logger.warning(f"警告：以下核心数据文件不存在: {', '.join(missing_files)}")
        logger.warning("为确保数据完整性，不会删除合并文件。使用--force选项可强制删除。")
        return 0, 0
    
    cleaned_size = 0
    count = 0
    
    for file_name in merged_files:
        file_path = dataset_dir / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            
            if not missing_files or force:
                logger.info(f"找到合并数据文件: {file_path} ({format_size(size)})")
                if not dry_run:
                    try:
                        file_path.unlink()
                        logger.info(f"已删除: {file_path} - 此文件可从核心训练文件重新生成")
                        cleaned_size += size
                        count += 1
                    except Exception as e:
                        logger.error(f"删除 {file_path} 时出错: {e}")
            else:
                logger.warning(f"跳过删除合并文件 {file_path}，因为需要保留完整的训练数据")
    
    # 显示保留的核心文件信息
    if count > 0 or not missing_files:
        logger.info("已保留以下核心训练数据文件:")
        for core_file in core_files:
            core_path = dataset_dir / core_file
            if core_path.exists():
                size = core_path.stat().st_size
                logger.info(f"  - {core_file} ({format_size(size)})")
    
    return count, cleaned_size

def cleanup_pycache(dry_run=False):
    """清理所有__pycache__目录"""
    pycache_dirs = list(Path(".").glob("**/__pycache__"))
    
    cleaned_size = 0
    count = 0
    
    for pycache_dir in pycache_dirs:
        if not pycache_dir.is_dir():
            continue
            
        size = get_dir_size(pycache_dir)
        logger.info(f"找到__pycache__目录: {pycache_dir} ({format_size(size)})")
        if not dry_run:
            try:
                shutil.rmtree(pycache_dir)
                logger.info(f"已删除: {pycache_dir}")
                cleaned_size += size
                count += 1
            except Exception as e:
                logger.error(f"删除 {pycache_dir} 时出错: {e}")
    
    return count, cleaned_size

def analyze_space_usage(min_size_mb=10, top_count=20):
    """分析空间使用情况"""
    print("\n" + "=" * 60)
    print("空间使用情况分析")
    print("=" * 60)
    
    # 分析根目录大小
    root_dirs = []
    for item in os.listdir("."):
        item_path = os.path.join(".", item)
        if os.path.isdir(item_path):
            try:
                size = get_dir_size(item_path)
                root_dirs.append((item_path, size))
            except Exception as e:
                logger.error(f"计算 {item_path} 大小时出错: {e}")
    
    # 按大小排序
    root_dirs.sort(key=lambda x: x[1], reverse=True)
    
    # 显示根目录大小
    print(f"\n主要目录大小:")
    for dir_path, size in root_dirs:
        print(f"  {dir_path:<25} {format_size(size):>10}")
    
    # 查找大文件
    print(f"\n最大的 {top_count} 个文件:")
    large_files = scan_large_files(min_size_mb=min_size_mb)
    
    for i, (file_path, size) in enumerate(large_files[:top_count], 1):
        print(f"  {i:2}. {file_path:<50} {format_size(size):>10}")
    
    # 分析不同类型的文件
    print("\n按文件类型的空间使用:")
    file_types = {}
    for root, _, files in os.walk("."):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower() or "无扩展名"
                size = os.path.getsize(file_path)
                
                if ext not in file_types:
                    file_types[ext] = {"count": 0, "size": 0}
                
                file_types[ext]["count"] += 1
                file_types[ext]["size"] += size
            except Exception:
                continue
    
    # 显示文件类型统计
    sorted_types = sorted(file_types.items(), key=lambda x: x[1]["size"], reverse=True)
    
    for ext, stats in sorted_types[:15]:  # 只显示前15种类型
        print(f"  {ext:<15} {stats['count']:5} 个文件, 总大小: {format_size(stats['size'])}")

def compare_directories():
    """比较collection和dataset目录，分析大小差异"""
    collection_dir = Path("collection")
    dataset_dir = Path("dataset")
    
    if not collection_dir.exists() or not dataset_dir.exists():
        print("无法比较：collection或dataset目录不存在")
        return
    
    collection_size = get_dir_size(collection_dir)
    dataset_size = get_dir_size(dataset_dir)
    
    print("\n" + "=" * 60)
    print("数据目录大小比较")
    print("=" * 60)
    print(f"collection目录: {format_size(collection_size)}")
    print(f"dataset目录:    {format_size(dataset_size)}")
    
    ratio = dataset_size / collection_size if collection_size > 0 else 0
    print(f"比例:             {ratio:.2f}x")
    
    # 分析dataset目录中的文件类型和数量
    print("\n数据集目录内容分析:")
    dataset_files = {}
    for ext in [".jsonl", ".txt", ".json", ".bin", ".idx"]:
        count = len(list(dataset_dir.glob(f"**/*{ext}")))
        if count > 0:
            print(f"  {ext} 文件: {count} 个")
    
    # 分析数据集文件大小
    print("\nDataset目录中最大的文件:")
    dataset_large_files = []
    
    for file_path in dataset_dir.glob("**/*"):
        if file_path.is_file():
            size = file_path.stat().st_size
            dataset_large_files.append((file_path, size))
    
    dataset_large_files.sort(key=lambda x: x[1], reverse=True)
    
    for i, (file_path, size) in enumerate(dataset_large_files[:5], 1):
        print(f"  {i}. {file_path.relative_to(dataset_dir)}: {format_size(size)}")

def create_shell_script(dry_run=False):
    """创建统一的日志清理和目录整理shell脚本"""
    script_path = Path("cleanup.sh")
    
    # --- 在创建新脚本前删除旧脚本 --- #
    if not dry_run and script_path.exists():
        try:
            script_path.unlink()
            logger.info(f"已删除旧的清理脚本: {script_path}")
        except OSError as e:
            logger.warning(f"无法删除旧的清理脚本 {script_path}: {e}")
            # 如果无法删除，可能无法写入新脚本，可以选择继续或退出
    
    script_content = r"""#!/bin/bash

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
"""
    
    logger.info(f"创建清理shell脚本: {script_path}")
    
    if not dry_run:
        with open(script_path, 'w') as f:
            f.write(script_content)
        # 设置可执行权限
        os.chmod(script_path, 0o755)
        logger.info(f"脚本已创建并设置可执行权限")
    
    return script_path

def cleanup_shell_scripts(dry_run=False):
    """清理旧的shell脚本"""
    old_scripts = ["clean_old_logs.sh", "clean_logs.sh"]
    count = 0
    
    for script in old_scripts:
        script_path = Path(script)
        if script_path.exists():
            logger.info(f"找到旧的shell脚本: {script_path}")
            if not dry_run:
                try:
                    script_path.unlink()
                    logger.info(f"已删除: {script_path}")
                    count += 1
                except Exception as e:
                    logger.error(f"删除 {script_path} 时出错: {e}")
    
    return count

def main():
    parser = argparse.ArgumentParser(description="灵猫墨韵系统清理工具")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行，不实际删除文件")
    parser.add_argument("--temp", action="store_true", help="清理临时文件")
    parser.add_argument("--logs", action="store_true", help="清理日志文件")
    parser.add_argument("--data", action="store_true", help="清理合并数据文件")
    parser.add_argument("--pycache", action="store_true", help="清理__pycache__目录")
    parser.add_argument("--analyze", action="store_true", help="分析空间使用情况")
    parser.add_argument("--compare", action="store_true", help="比较collection和dataset目录")
    parser.add_argument("--organize", action="store_true", help="整理日志目录")
    parser.add_argument("--create-script", action="store_true", help="创建shell清理脚本")
    parser.add_argument("--clean-scripts", action="store_true", help="清理旧的shell脚本")
    parser.add_argument("--all", action="store_true", help="执行所有清理操作")
    parser.add_argument("--min-size", type=int, default=100, help="清理临时文件的最小大小(MB)")
    parser.add_argument("--max-log-size", type=int, default=50, help="保留日志文件的最大大小(MB)")
    parser.add_argument("--log-keep", type=int, default=5, help="每个目录保留的日志文件数量")
    parser.add_argument("--force", action="store_true", help="强制删除文件（谨慎使用）")

    args = parser.parse_args()
    
    # 如果没有指定任何操作，默认执行分析
    if not any([args.temp, args.logs, args.data, args.pycache, args.analyze, 
                args.compare, args.organize, args.create_script, 
                args.clean_scripts, args.all]):
        args.analyze = True
    
    start_time = time.time()
    total_cleaned_size = 0
    total_count = 0
    
    # 获取清理前的总大小
    before_size = get_dir_size(".")
    
    print("=" * 60)
    print(f"灵猫墨韵系统清理工具 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    if args.dry_run:
        print("模拟运行模式，不会实际删除文件")
    
    if args.force:
        print("警告: 强制模式已启用，将跳过某些安全检查")
    
    if args.analyze or args.all:
        analyze_space_usage(min_size_mb=args.min_size)
    
    if args.compare or args.all:
        compare_directories()
    
    if args.organize or args.all:
        print("\n整理日志目录...")
        moved = organize_logs(args.dry_run)
        print(f"日志整理完成: 移动了 {moved} 个日志文件")
    
    if args.all or args.temp:
        print("\n清理临时文件...")
        count, size = cleanup_temp_files(args.min_size, args.dry_run, args.force)
        total_count += count
        total_cleaned_size += size
        print(f"临时文件清理完成: 删除 {count} 个文件，释放 {format_size(size)} 空间")
    
    if args.all or args.logs:
        print("\n清理大型日志文件...")
        count, size = cleanup_large_logs(args.max_log_size, args.log_keep, args.dry_run)
        total_count += count
        total_cleaned_size += size
        print(f"日志文件清理完成: 删除 {count} 个文件，释放 {format_size(size)} 空间")
    
    if args.all or args.data:
        print("\n清理合并数据文件...")
        count, size = cleanup_merged_data(args.dry_run, args.force)
        total_count += count
        total_cleaned_size += size
        print(f"合并数据文件清理完成: 删除 {count} 个文件，释放 {format_size(size)} 空间")
    
    if args.all or args.pycache:
        print("\n清理__pycache__目录...")
        count, size = cleanup_pycache(args.dry_run)
        total_count += count
        total_cleaned_size += size
        print(f"__pycache__清理完成: 删除 {count} 个目录，释放 {format_size(size)} 空间")
    
    if args.create_script or args.all:
        print("\n创建清理shell脚本...")
        script_path = create_shell_script(args.dry_run)
        print(f"清理脚本已创建: {script_path}")
    
    if args.clean_scripts or args.all:
        print("\n清理旧的shell脚本...")
        count = cleanup_shell_scripts(args.dry_run)
        total_count += count
        print(f"旧脚本清理完成: 删除 {count} 个脚本文件")
    
    end_time = time.time()
    duration = end_time - start_time
    
    if not args.dry_run and (args.all or args.temp or args.logs or args.data or args.pycache or args.clean_scripts):
        # 获取清理后的总大小
        after_size = get_dir_size(".")
        actual_reduction = before_size - after_size
        
        print("\n" + "=" * 60)
        print(f"清理完成! 耗时: {duration:.2f} 秒")
        print(f"清理前总大小: {format_size(before_size)}")
        print(f"清理后总大小: {format_size(after_size)}")
        print(f"实际减少: {format_size(actual_reduction)} ({actual_reduction/before_size*100:.2f}%)")
        print("=" * 60)
    elif args.dry_run:
        print("\n" + "=" * 60)
        print(f"模拟清理完成! 耗时: {duration:.2f} 秒")
        print(f"预计可释放空间: {format_size(total_cleaned_size)}")
        print(f"预计可删除文件/目录数: {total_count}")
        print("=" * 60)
        print("要执行实际清理，请移除 --dry-run 参数")
    else:
        print("\n" + "=" * 60)
        print(f"操作完成! 耗时: {duration:.2f} 秒")
        print("=" * 60)

if __name__ == "__main__":
    main() 