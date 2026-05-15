# 数据格式说明文档

本文档详细说明灵猫墨韵项目所使用的数据格式规范，包括JSONL格式规范、数据字段定义、元数据格式以及完整的示例数据。

## 目录

- [JSONL格式说明](#jsonl格式说明)
- [数据字段说明](#数据字段说明)
- [元数据格式](#元数据格式)
- [示例数据](#示例数据)
- [格式验证](#格式验证)
- [最佳实践](#最佳实践)

---

## JSONL格式说明

### 什么是JSONL

JSONL（JSON Lines）是一种基于JSON的纯文本格式，用于存储结构化数据。每一行都是一个完整的JSON对象，各行之间以换行符分隔。这种格式具有以下特点：

- **流式处理友好**：可以逐行读取和写入，无需将整个文件加载到内存
- **便于日志记录**：每条记录独立，便于追加和tail操作
- **压缩友好**：可以使用gzip等工具进行压缩
- **易于调试**：可以逐行查看和验证数据

### JSONL文件规范

灵猫墨韵项目采用JSONL作为主要的数据交换格式，遵循以下规范：

1. **编码格式**：统一使用UTF-8编码，确保中文字符正确处理
2. **行分隔符**：使用Unix风格的LF（\n）作为行分隔符
3. **JSON标准**：严格遵循RFC 8259标准
4. **压缩支持**：支持gzip压缩，文件名以`.jsonl.gz`结尾
5. **BOM处理**：不建议使用UTF-8 BOM

### 文件命名规范

项目中的JSONL文件采用统一的命名规范：

```
{类别}_{子类别}_{版本号}.jsonl
```

示例命名如下：

| 文件名 | 说明 |
|--------|------|
| `poetry_tang_001.jsonl` | 唐诗数据，第一版 |
| `poetry_song_002.jsonl` | 宋词数据，第二版 |
| `classical_philosophy_001.jsonl` | 哲学经典数据 |
| `merged_v3.jsonl` | 合并后的数据集，第三版 |

---

## 数据字段说明

### 核心字段

灵猫墨韵项目的数据记录包含以下核心字段，每个字段都有明确的定义和使用场景：

#### content字段

`content`字段是数据记录中最重要的字段，存储实际的内容文本。该字段的类型为字符串，内容为经过预处理的纯文本，不包含任何HTML标签或特殊格式标记。在训练过程中，该字段的内容将直接送入分词器进行编码。示例内容如下：

```
"content": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
```

该字段需要注意以下几点：首先，内容长度应控制在合理范围内，建议单条内容不超过4096个字符；其次，避免在content字段中包含换行符，应使用连续的文本流；最后，对于古典文献，应保持原文的完整性，包括标点的原始状态。

#### book字段

`book`字段用于标识内容所属的典籍名称，这是一个重要的分类字段。该字段为字符串类型，取值应来自项目维护的典籍白名单。以下是一些常见的取值示例：

```
"book": "论语"
"book": "道德经"
"book": "庄子"
"book": "诗经"
"book": "唐诗三百首"
```

在处理数据时，应确保book字段的准确性。对于来源不明确的内容，该字段可以为空字符串或null，但不应使用"未知"或"unknown"等模糊值。

#### chapter字段

`chapter`字段标识内容在典籍中的篇章名称，用于更细粒度的分类。该字段同样为字符串类型，其取值依赖于book字段的取值。典型示例包括：

```
"book": "论语"
"chapter": "学而"

"book": "道德经"
"chapter": "第一章"
```

需要注意的是，同一书名下的章节名称应保持唯一性，避免重复定义。

#### paragraph_index字段

`paragraph_index`字段是一个整数类型，表示当前段落在其所属篇章中的索引位置。该字段从1开始计数，用于保持内容的原始顺序。具体示例如下：

```
"book": "论语"
"chapter": "学而"
"paragraph_index": 1
```

这个字段在进行数据分析和可视化时特别有用，可以帮助追踪内容的原始结构。

### 扩展字段

除了核心字段外，数据记录还可以包含以下扩展字段，用于提供更丰富的信息：

#### title字段

`title`字段用于存储内容的标题信息，主要用于诗词类数据。该字段为字符串类型，可选填。对于有题目的诗词作品，应填写完整的题目；对于无题的古文段落，可以省略此字段。示例如下：

```
"title": "静夜思"
"content": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
```

#### author字段

`author`字段标识内容的作者信息。对于有明确作者的内容，应填写作者姓名；对于作者不详的典籍，可以填写"佚名"或省略此字段。示例如下：

```
"author": "李白"
"title": "静夜思"
"content": "床前明月光，疑是地上霜。"
```

#### dynasty字段

`dynasty`字段用于标识内容所属的历史朝代。该字段为字符串类型，可选填。常见取值包括"先秦"、"汉"、"唐"、"宋"、"元"、"明"、"清"等。示例如下：

```
"dynasty": "唐"
"author": "李白"
"title": "静夜思"
"content": "床前明月光，疑是地上霜。"
```

#### difficulty字段

`difficulty`字段是一个浮点数，取值范围为0.0到1.0，表示内容的阅读难度。该字段主要用于训练时的样本加权。难度评分越高，表示内容越复杂，需要更多的训练权重。示例如下：

```
"difficulty": 0.3   // 简单，适合初学者
"difficulty": 0.6   // 中等难度
"difficulty": 0.9   // 高难度，需要专家级理解
```

#### tags字段

`tags`字段是一个字符串数组，用于存储多个标签信息。标签可以描述内容的题材、风格、主题等特征。示例如下：

```
"tags": ["五言绝句", "思乡", "月夜", "唐诗"]
```

在实际应用中，tags字段可以用于数据的筛选和分组，便于构建特定主题的子数据集。

---

## 元数据格式

### 数据集级元数据

每个数据集除了包含实际的数据记录外，还应包含描述整个数据集的元数据信息。元数据文件通常命名为`metadata.json`，与数据集文件放在同一目录下。典型的元数据文件结构如下：

```json
{
    "name": "lingmao_classical_poetry_v1",
    "description": "灵猫墨韵古典诗词数据集第一版",
    "version": "1.0.0",
    "created_at": "2026-01-15T08:00:00Z",
    "updated_at": "2026-05-15T10:30:00Z",
    "license": "Apache-2.0",
    "authors": [
        {
            "name": "Lingmao Team",
            "email": "contact@example.com",
            "affiliation": "Lingmao Moyun Project"
        }
    ],
    "dataset_size": {
        "total_records": 15000,
        "total_characters": 450000,
        "total_tokens": 120000
    },
    "data_sources": [
        {
            "name": "古诗文网",
            "url": "https://www.gushiwen.org/",
            "license": "Public Domain",
            "access_date": "2026-01-10"
        },
        {
            "name": "中国哲学书电子化计划",
            "url": "https://ctext.org/",
            "license": "CC BY-NC-SA 3.0",
            "access_date": "2026-01-12"
        }
    ],
    "quality_metrics": {
        "avg_length": 30.5,
        "min_length": 5,
        "max_length": 256,
        "duplication_rate": 0.02,
        "missing_fields_rate": 0.001
    },
    "language": "zh-CN",
    "domain": "classical_chinese_literature"
}
```

### 处理日志

在数据处理过程中，应保留完整的处理日志，用于追溯数据来源和处理步骤。处理日志可以采用JSONL格式存储，每行记录一个处理步骤：

```json
{"timestamp": "2026-05-15T08:00:00Z", "step": "download", "source": "古诗文网", "status": "success", "records": 5000}
{"timestamp": "2026-05-15T08:15:00Z", "step": "clean", "input_records": 5000, "output_records": 4850, "dropped": 150}
{"timestamp": "2026-05-15T08:30:00Z", "step": "convert", "input_records": 4850, "output_records": 4850, "status": "success"}
{"timestamp": "2026-05-15T08:45:00Z", "step": "merge", "input_files": ["poetry.jsonl", "classical.jsonl"], "output_records": 10000}
```

### 数据集统计

数据集的统计信息对于评估数据质量和规划训练资源非常重要。以下是一个完整的数据集统计报告示例：

```json
{
    "statistics": {
        "by_category": {
            "poetry": {
                "count": 8000,
                "avg_length": 28.5,
                "subcategories": {
                    "tang_poetry": 3500,
                    "song_poetry": 3000,
                    "yuan_poetry": 800,
                    "other": 700
                }
            },
            "classical": {
                "count": 5000,
                "avg_length": 85.2,
                "subcategories": {
                    "philosophy": 2000,
                    "history": 1500,
                    "literature": 1500
                }
            },
            "terms": {
                "count": 2000,
                "avg_length": 4.5,
                "categories": {
                    "medical": 500,
                    "astronomical": 300,
                    "agricultural": 400,
                    "other": 800
                }
            }
        },
        "by_difficulty": {
            "easy": 4000,
            "medium": 7000,
            "hard": 4000
        },
        "temporal_distribution": {
            "pre_qin": 1500,
            "han": 1000,
            "tang": 3500,
            "song": 3000,
            "yuan": 800,
            "ming": 500,
            "qing": 200,
            "modern": 500
        }
    }
}
```

---

## 示例数据

### 诗词数据示例

以下是一个完整的诗词数据记录示例，展示了所有字段的典型取值：

```json
{"book": "唐诗三百首", "chapter": "五言绝句", "paragraph_index": 1, "title": "静夜思", "author": "李白", "dynasty": "唐", "content": "床前明月光，疑是地上霜。举头望明月，低头思故乡。", "difficulty": 0.2, "tags": ["五言绝句", "思乡", "月夜", "李白"]}
{"book": "唐诗三百首", "chapter": "七言绝句", "paragraph_index": 5, "title": "望庐山瀑布", "author": "李白", "dynasty": "唐", "content": "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。", "difficulty": 0.25, "tags": ["七言绝句", "山水", "瀑布", "李白"]}
{"book": "宋词精选", "chapter": "豪放词", "paragraph_index": 3, "title": "念奴娇·赤壁怀古", "author": "苏轼", "dynasty": "宋", "content": "大江东去，浪淘尽，千古风流人物。故垒西边，人道是，三国周郎赤壁。乱石穿空，惊涛拍岸，卷起千堆雪。江山如画，一时多少豪杰。", "difficulty": 0.55, "tags": ["豪放词", "怀古", "赤壁", "苏轼"]}
```

### 古文经典示例

以下示例展示了古典文献数据的典型格式：

```json
{"book": "论语", "chapter": "学而", "paragraph_index": 1, "content": "子曰：学而时习之，不亦说乎？有朋自远方来，不亦乐乎？人不知而不愠，不亦君子乎？", "difficulty": 0.4, "tags": ["论语", "儒家", "修身"]}
{"book": "论语", "chapter": "为政", "paragraph_index": 1, "content": "子曰：为政以德，譬如北辰，居其所而众星共之。", "difficulty": 0.45, "tags": ["论语", "儒家", "政治"]}
{"book": "道德经", "chapter": "第一章", "paragraph_index": 1, "content": "道可道，非常道。名可名，非常名。无名天地之始；有名万物之母。", "difficulty": 0.7, "tags": ["道德经", "道家", "哲学", "玄学"]}
{"book": "庄子", "chapter": "逍遥游", "paragraph_index": 1, "content": "北冥有鱼，其名为鲲。鲲之大，不知其几千里也。化而为鸟，其名为鹏。", "difficulty": 0.65, "tags": ["庄子", "道家", "寓言"]}
```

### 古典术语示例

术语数据集用于训练专业领域的古典语言理解能力：

```json
{"book": "中医术语", "chapter": "基础理论", "paragraph_index": 1, "content": "阴阳", "difficulty": 0.3, "tags": ["中医", "基础概念"]}
{"book": "中医术语", "chapter": "基础理论", "paragraph_index": 2, "content": "五行", "difficulty": 0.35, "tags": ["中医", "基础概念"]}
{"book": "天文术语", "chapter": "星象", "paragraph_index": 1, "content": "二十八宿", "difficulty": 0.5, "tags": ["天文", "星象"]}
```

### 完整的数据集文件示例

以下是一个完整的JSONL文件示例，包含多条不同类型的数据记录：

```jsonl
{"book": "论语", "chapter": "学而", "paragraph_index": 1, "content": "子曰：学而时习之，不亦说乎？有朋自远方来，不亦乐乎？人不知而不愠，不亦君子乎？", "difficulty": 0.4, "tags": ["论语", "儒家", "修身"]}
{"book": "道德经", "chapter": "第一章", "paragraph_index": 1, "content": "道可道，非常道。名可名，非常名。无名天地之始；有名万物之母。", "difficulty": 0.7, "tags": ["道德经", "道家", "哲学", "玄学"]}
{"book": "唐诗三百首", "chapter": "五言绝句", "paragraph_index": 1, "title": "静夜思", "author": "李白", "dynasty": "唐", "content": "床前明月光，疑是地上霜。举头望明月，低头思故乡。", "difficulty": 0.2, "tags": ["五言绝句", "思乡", "月夜", "李白"]}
{"book": "庄子", "chapter": "逍遥游", "paragraph_index": 1, "content": "北冥有鱼，其名为鲲。鲲之大，不知其几千里也。化而为鸟，其名为鹏。", "difficulty": 0.65, "tags": ["庄子", "道家", "寓言"]}
```

---

## 格式验证

### 验证脚本

为了确保数据格式的正确性，项目提供了数据验证脚本。以下是一个完整的验证函数实现：

```python
import json
import sys
from typing import Dict, List, Tuple, Optional

REQUIRED_FIELDS = ["content", "book", "chapter", "paragraph_index"]
OPTIONAL_FIELDS = ["title", "author", "dynasty", "difficulty", "tags"]
ALL_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS

def validate_record(record: Dict, line_num: int) -> List[str]:
    """验证单条数据记录的格式是否正确
    
    参数:
        record: 待验证的数据记录字典
        line_num: 当前记录的行号
        
    返回:
        错误信息列表，如果为空则表示验证通过
    """
    errors = []
    
    # 检查必需字段
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"行 {line_num}: 缺少必需字段 '{field}'")
        elif not isinstance(record[field], str) and field != "paragraph_index":
            errors.append(f"行 {line_num}: 字段 '{field}' 应为字符串类型")
    
    # 验证paragraph_index类型
    if "paragraph_index" in record:
        if not isinstance(record["paragraph_index"], int):
            errors.append(f"行 {line_num}: 字段 'paragraph_index' 应为整数类型")
        elif record["paragraph_index"] < 1:
            errors.append(f"行 {line_num}: 字段 'paragraph_index' 应大于等于1")
    
    # 验证difficulty字段
    if "difficulty" in record:
        difficulty = record["difficulty"]
        if not isinstance(difficulty, (int, float)):
            errors.append(f"行 {line_num}: 字段 'difficulty' 应为数值类型")
        elif not 0.0 <= difficulty <= 1.0:
            errors.append(f"行 {line_num}: 字段 'difficulty' 应在0.0到1.0之间")
    
    # 验证tags字段
    if "tags" in record:
        if not isinstance(record["tags"], list):
            errors.append(f"行 {line_num}: 字段 'tags' 应为数组类型")
        elif not all(isinstance(tag, str) for tag in record["tags"]):
            errors.append(f"行 {line_num}: 字段 'tags' 中的所有元素应为字符串")
    
    # 验证content不为空
    if "content" in record:
        if not record["content"].strip():
            errors.append(f"行 {line_num}: 字段 'content' 不能为空")
        elif len(record["content"]) > 10000:
            errors.append(f"行 {line_num}: 字段 'content' 长度超过10000字符")
    
    return errors

def validate_jsonl_file(file_path: str) -> Tuple[bool, int, int, List[str]]:
    """验证JSONL文件的格式是否正确
    
    参数:
        file_path: JSONL文件的路径
        
    返回:
        元组 (是否通过验证, 总记录数, 错误记录数, 错误信息列表)
    """
    errors = []
    total_records = 0
    error_records = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                total_records += 1
                
                try:
                    record = json.loads(line)
                    record_errors = validate_record(record, line_num)
                    if record_errors:
                        error_records += 1
                        errors.extend(record_errors)
                except json.JSONDecodeError as e:
                    error_records += 1
                    errors.append(f"行 {line_num}: JSON解析错误 - {str(e)}")
                    
    except FileNotFoundError:
        errors.append(f"文件未找到: {file_path}")
        return False, 0, 0, errors
    except Exception as e:
        errors.append(f"读取文件时发生错误: {str(e)}")
        return False, 0, 0, errors
    
    passed = error_records == 0 and total_records > 0
    return passed, total_records, error_records, errors

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python validate_data.py <jsonl_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    print(f"正在验证文件: {file_path}")
    
    passed, total, errors_count, errors = validate_jsonl_file(file_path)
    
    print(f"\n验证结果:")
    print(f"- 总记录数: {total}")
    print(f"- 错误记录数: {errors_count}")
    
    if passed:
        print(f"\n✓ 验证通过！所有 {total} 条记录格式正确。")
    else:
        print(f"\n✗ 验证失败！发现 {errors_count} 条错误记录。")
        print(f"\n错误详情:")
        for error in errors[:20]:  # 只显示前20条错误
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  ... 还有 {len(errors) - 20} 条错误未显示")

if __name__ == "__main__":
    main()
```

### 快速验证命令

项目还提供了简化的命令行验证工具：

```bash
# 使用Python直接验证
python -c "
import json

with open('dataset/classical.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        try:
            data = json.loads(line)
            if 'content' not in data:
                print(f'行 {i}: 缺少content字段')
        except json.JSONDecodeError as e:
            print(f'行 {i}: JSON格式错误 - {e}')

print('验证完成')
"

# 统计数据集基本信息
python -c "
import json

stats = {'total': 0, 'by_book': {}}

with open('dataset/classical.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        stats['total'] += 1
        book = data.get('book', 'unknown')
        stats['by_book'][book] = stats['by_book'].get(book, 0) + 1

print(f'总记录数: {stats[\"total\"]}')
print('按典籍分类:')
for book, count in sorted(stats['by_book'].items(), key=lambda x: -x[1]):
    print(f'  {book}: {count}')
"
```

---

## 最佳实践

### 数据准备建议

在准备训练数据时，应遵循以下最佳实践以确保数据质量和训练效果：

首先，关于数据来源的选择。应优先使用权威出版社或学术机构发布的数据，避免使用来源不明或质量无法保证的数据。对于古典文献，建议使用经过专业校对的版本，如中华书局、商务印书馆等知名出版社的出版物。在使用网络数据时，应注意版权问题，确保数据的使用符合相关许可协议。

其次，关于数据清洗的流程。清洗过程应分步骤进行，每一步都应保留中间结果，便于问题追溯。清洗完成后应进行质量检查，确保没有引入新的错误。对于识别出的问题数据，应分类记录并分析原因，不断优化清洗规则。

第三，关于格式统一的问题。在转换不同来源的数据时，应建立统一的字段映射规则。不同来源的数据可能使用不同的字段名，需要制定映射表进行转换。对于可选字段，应明确填写规则，避免出现大量缺失值影响模型训练。

### 存储优化建议

大规模数据集的存储需要考虑效率和成本的平衡：

在存储格式选择方面，对于需要频繁随机访问的数据，建议使用分块的JSONL或SQLite格式。对于顺序读取的训练场景，可以使用二进制格式如HDF5或Protocol Buffers来减少存储空间和加快读取速度。对于归档保存的场景，可以使用gzip或zstd压缩的JSONL文件，通常可以获得5-10倍的压缩比。

在目录结构设计方面，建议按照数据类型和时间版本组织目录。例如：

```
dataset/
├── raw/              # 原始下载数据
│   ├── 2026-01/
│   └── 2026-05/
├── processed/        # 清洗后的数据
│   ├── poetry/
│   └── classical/
├── tokenized/        # 分词后的数据
│   └── merged/
└── metadata/         # 元数据和统计信息
```

### 版本管理建议

数据集的版本管理对于实验复现和迭代优化非常重要。建议采用语义化版本号（SemVer）格式，如1.0.0、1.1.0、2.0.0等。每个版本的变更应记录在CHANGELOG文件中，包括新增、修改和删除的内容。不同版本的数据应分开存储，便于对比和回退。

---

## 相关文档

- [数据获取与预处理指南](getting_started.md) - 详细的数据获取流程说明
- [数据集说明](../../DATASET.md) - 完整的数据集文档
- [数据管道脚本](../../scripts/data_pipeline.sh) - 自动化数据处理工具

---

_最后更新：2026-05-15_
