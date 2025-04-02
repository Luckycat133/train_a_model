# 灵猫墨韵项目维护指南

## 项目大小管理

灵猫墨韵项目在处理大量数据时可能占用较大的磁盘空间。本文档提供了管理项目大小和定期清理的最佳实践。

### 主要空间占用来源

1. **数据集文件** - `dataset/` 目录通常是最大的空间占用者（约8-10GB）
2. **原始语料库** - `collection/` 目录包含原始语料库（约5-6GB）
3. **临时文件** - 如 `temp_corpus.txt` 等大型处理文件（可能高达10GB以上）
4. **日志文件** - 频繁运行可能会生成大量日志（可能达到几GB）
5. **已合并的数据文件** - 如 `train_data.jsonl`（常达到数GB）

### 避免空间浪费的最佳实践

1. **使用 .gitignore**
   - 项目根目录已添加 `.gitignore` 文件，确保大型数据文件不会被添加到版本控制中
   - 避免手动覆盖这些规则

2. **定期清理**
   - 使用项目提供的清理脚本 `cleanup.py` 定期清理临时文件
   - 工作流程完成后，删除不必要的中间文件

3. **分步处理大型文件**
   - 处理大型数据集时，尽量分批次进行，避免一次性加载全部数据
   - 使用流处理模式而非全量加载

4. **分离训练与生产环境**
   - 考虑将训练数据与生产模型分开存储
   - 训练完成后，可以将原始训练数据归档或移动到外部存储

## 清理脚本使用方法

项目提供了自动清理脚本 `cleanup.py`，可帮助定期维护项目大小。

### 基本用法

```bash
# 执行模拟清理（不实际删除文件）
python cleanup.py --dry-run

# 执行所有清理操作
python cleanup.py --all

# 只清理临时文件
python cleanup.py --temp

# 只清理日志文件
python cleanup.py --logs

# 只清理合并数据文件
python cleanup.py --data

# 只清理Python缓存文件
python cleanup.py --pycache
```

### 高级选项

```bash
# 清理大于200MB的临时文件
python cleanup.py --temp --min-size 200

# 清理大于100MB的日志文件，每个目录只保留最新的3个
python cleanup.py --logs --max-log-size 100 --log-keep 3

# 组合多个选项
python cleanup.py --temp --logs --min-size 500 --max-log-size 200
```

### 定期清理建议

1. **日常开发**
   - 每天结束工作时执行 `python cleanup.py --temp --pycache`

2. **每周维护**
   - 每周执行一次 `python cleanup.py --all`

3. **特殊情况**
   - 磁盘空间不足时：`python cleanup.py --all --min-size 50 --max-log-size 20 --log-keep 2`
   - 准备部署前：`python cleanup.py --all`

### 手动清理大型文件

对于特别大的文件，如果脚本无法处理，可以手动删除：

```bash
# 查找大型文件
find . -type f -size +100M | sort -h

# 删除特定大文件
rm path/to/large/file.ext
```

## 监控项目大小

定期检查项目大小可以及时发现异常增长：

```bash
# 检查整个项目大小
du -h -d 1 .

# 检查特定目录
du -h -d 2 ./dataset
du -h -d 2 ./logs
```

## 注意事项

1. 请勿删除必要的训练数据或模型文件
2. 清理前，确保所有进程已完成
3. 在删除任何文件前，请先使用 `--dry-run` 选项查看将被删除的内容
4. 保持备份重要数据

通过遵循这些最佳实践，可以有效管理灵猫墨韵项目的磁盘占用，保持系统运行高效。 