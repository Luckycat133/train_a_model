"""数据集配置系统使用示例。

本文件展示如何使用灵猫墨韵项目的数据集配置管理系统，
包括数据集注册、加载、预处理和混合等功能。

运行示例：
    python examples/data_example.py

功能演示：
1. 数据集注册表使用
2. 数据加载器基础用法
3. 流式加载大规模数据
4. 数据预处理和过滤
5. 数据混合训练
6. 配置文件加载
"""

import json
import tempfile
from pathlib import Path


def create_sample_data():
    """创建示例数据文件用于演示。"""
    sample_dir = Path("dataset")
    sample_dir.mkdir(exist_ok=True)
    
    poetry_data = [
        {"title": "静夜思", "author": "李白", "dynasty": "唐", "content": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"},
        {"title": "春晓", "author": "孟浩然", "dynasty": "唐", "content": "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"},
        {"title": "登鹳雀楼", "author": "王之涣", "dynasty": "唐", "content": "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"},
        {"title": "望庐山瀑布", "author": "李白", "dynasty": "唐", "content": "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。"},
        {"title": "江雪", "author": "柳宗元", "dynasty": "唐", "content": "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。"},
    ]
    
    with open(sample_dir / "poetry.json", 'w', encoding='utf-8') as f:
        json.dump(poetry_data, f, ensure_ascii=False, indent=2)
    
    classical_data = [
        {"book": "论语", "chapter": "学而", "paragraph_index": 1, "content": "子曰：学而时习之，不亦说乎？有朋自远方来，不亦乐乎？人不知而不愠，不亦君子乎？"},
        {"book": "论语", "chapter": "为政", "paragraph_index": 1, "content": "子曰：为政以德，譬如北辰，居其所而众星共之。"},
        {"book": "道德经", "chapter": "第一章", "paragraph_index": 1, "content": "道可道，非常道。名可名，非常名。无名天地之始；有名万物之母。"},
        {"book": "庄子", "chapter": "逍遥游", "paragraph_index": 1, "content": "北冥有鱼，其名为鲲。鲲之大，不知其几千里也。化而为鸟，其名为鹏。"},
    ]
    
    with open(sample_dir / "classical.jsonl", 'w', encoding='utf-8') as f:
        for item in classical_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"示例数据已创建在 {sample_dir} 目录")


def example_registry():
    """数据集注册表示例。"""
    print("\n" + "="*60)
    print("示例1: 数据集注册表使用")
    print("="*60)
    
    from src.data_config import DatasetRegistry, DatasetInfo
    
    registry = DatasetRegistry()
    
    registry.register_dataset(
        name="poetry_sample",
        source="local://dataset/poetry.json",
        path="dataset/poetry.json",
        size=5,
        description="示例古诗词数据集",
        format="json",
        tags=["poetry", "sample", "tang"],
    )
    
    registry.register_dataset(
        name="classical_sample",
        source="local://dataset/classical.jsonl",
        path="dataset/classical.jsonl",
        size=4,
        description="示例古文典籍数据集",
        format="jsonl",
        tags=["classical", "sample", "confucian"],
    )
    
    print("\n已注册的数据集:")
    for name, info in registry.list_datasets():
        print(f"  - {name}: {info.description} ({info.size} 样本)")
    
    print("\n搜索'poetry'相关数据集:")
    for name, info in registry.search_datasets("poetry"):
        print(f"  - {name}: {info.description}")
    
    dataset_info = registry.get_dataset("poetry_sample")
    if dataset_info:
        print(f"\n获取数据集详情:")
        print(f"  名称: {dataset_info.name}")
        print(f"  路径: {dataset_info.path}")
        print(f"  格式: {dataset_info.format}")
        print(f"  标签: {', '.join(dataset_info.tags)}")


def example_loader():
    """数据加载器示例。"""
    print("\n" + "="*60)
    print("示例2: 数据加载器使用")
    print("="*60)
    
    from src.data_config import DatasetLoader, DataFormat
    
    loader = DatasetLoader()
    
    print("\n加载JSON格式数据:")
    data = loader.load("dataset/poetry.json", format=DataFormat.JSON)
    for i, item in enumerate(data[:3]):
        title = item.get('title', '无题')
        author = item.get('author', '未知')
        content = item.get('content', '')[:30]
        print(f"  [{i+1}] {title} - {author}: {content}...")
    
    print(f"\n总共加载 {len(data)} 条数据")
    
    print("\n加载JSONL格式数据:")
    data = loader.load("dataset/classical.jsonl", format=DataFormat.JSONL)
    for i, item in enumerate(data[:3]):
        book = item.get('book', '未知')
        chapter = item.get('chapter', '未知')
        content = item.get('content', '')[:30]
        print(f"  [{i+1}] {book}-{chapter}: {content}...")
    
    print(f"\n总共加载 {len(data)} 条数据")


def example_streaming():
    """流式加载示例。"""
    print("\n" + "="*60)
    print("示例3: 流式加载大规模数据")
    print("="*60)
    
    from src.data_config import DatasetLoader, StreamingDataLoader
    
    print("\n使用流式加载器逐条处理:")
    stream_loader = StreamingDataLoader(batch_size=2)
    
    count = 0
    for item in stream_loader.iter_samples("dataset/classical.jsonl"):
        book = item.get('book', '未知')
        content = item.get('content', '')[:40]
        print(f"  样本 {count+1}: {book} - {content}...")
        count += 1
        if count >= 4:
            break
    
    print(f"\n流式加载了 {count} 条数据（演示用）")
    
    print("\n使用批量流式加载:")
    stream_loader = StreamingDataLoader(batch_size=2)
    
    for batch_idx, batch in enumerate(stream_loader.iter_batches("dataset/poetry.json")):
        print(f"  批次 {batch_idx+1}: 包含 {len(batch)} 条数据")
        if batch_idx >= 1:
            break


def example_preprocessing():
    """数据预处理示例。"""
    print("\n" + "="*60)
    print("示例4: 数据预处理和过滤")
    print("="*60)
    
    from src.data_config import DatasetLoader, DataFormat
    
    loader = DatasetLoader()
    
    loader.add_filter(
        field="text",
        func=lambda item: len(item.get('content', '')) >= 20
    )
    
    print("\n添加长度过滤后加载数据:")
    loader.add_transform(
        func=lambda item: {
            **item,
            'text': item.get('content', ''),
            'preview': item.get('content', '')[:20] + '...'
        }
    )
    
    data = loader.load("dataset/poetry.json", format=DataFormat.JSON)
    
    for i, item in enumerate(data):
        title = item.get('title', '无题')
        preview = item.get('preview', '')
        print(f"  [{i+1}] {title}: {preview}")


def example_mixer():
    """数据混合示例。"""
    print("\n" + "="*60)
    print("示例5: 数据混合训练")
    print("="*60)
    
    from src.data_config import DataMixer
    
    mixer = DataMixer(seed=42, shuffle=True)
    
    mixer.add_dataset(
        path="dataset/poetry.json",
        weight=0.6,
        format=None,
        name="poetry"
    )
    
    mixer.add_dataset(
        path="dataset/classical.jsonl",
        weight=0.4,
        format=None,
        name="classical"
    )
    
    print("\n混合数据集统计:")
    stats = mixer.get_stats()
    print(f"  数据集数量: {stats['num_datasets']}")
    print(f"  采样权重: {stats['weights']}")
    print("  各数据集信息:")
    for ds in stats['datasets']:
        print(f"    - {ds['name']}: {ds['samples']} 样本, 权重 {ds['weight']}")
    
    print("\n生成混合样本 (前6个):")
    for i, item in enumerate(mixer.iter_mixed(max_samples=6)):
        source = item.get('_source_dataset', 'unknown')
        text = item.get('content', item.get('text', ''))[:30]
        print(f"  [{i+1}] 来源={source}: {text}...")


def example_config():
    """配置文件加载示例。"""
    print("\n" + "="*60)
    print("示例6: 配置文件加载")
    print("="*60)
    
    from src.data_config import DatasetRegistry
    import yaml
    
    registry = DatasetRegistry()
    
    config_files = [
        "config/datasets/poetry.yaml",
        "config/datasets/classical.yaml",
        "config/datasets/mixed.yaml",
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            count = registry.load_from_config(config_file)
            print(f"从 {config_file} 加载了 {count} 个数据集")
        else:
            print(f"配置文件不存在: {config_file}")
    
    print("\n从配置文件加载的数据集:")
    for name, info in registry.list_datasets():
        print(f"  - {name}: {info.description}")


def example_schema():
    """配置Schema验证示例。"""
    print("\n" + "="*60)
    print("示例7: 配置Schema验证")
    print("="*60)
    
    from src.data_config import DatasetConfig, DataSource, DataFormat
    from pydantic import ValidationError
    
    print("\n创建有效的数据集配置:")
    try:
        config = DatasetConfig(
            name="test_dataset",
            version="1.0.0",
            description="测试数据集",
            source=DataSource(
                url="https://example.com/data.jsonl",
                path="data/test.jsonl",
            ),
            format=DataFormat.JSONL,
            tags=["test", "sample"],
            size=1000,
        )
        print(f"  配置创建成功!")
        print(f"  名称: {config.name}")
        print(f"  格式: {config.format.value}")
        print(f"  标签: {', '.join(config.tags)}")
    except ValidationError as e:
        print(f"  验证失败: {e}")
    
    print("\n验证名称格式:")
    test_names = ["valid_name", "valid-name", "Valid Name", "invalid name!", ""]
    for name in test_names:
        try:
            test_config = DatasetConfig(
                name=name,
                source=DataSource(path="test.jsonl"),
            )
            print(f"  '{name}' -> 有效")
        except ValidationError as e:
            print(f"  '{name}' -> 无效: {str(e)[:50]}...")


def main():
    """主函数：运行所有示例。"""
    print("="*60)
    print("灵猫墨韵 - 数据集配置系统示例")
    print("="*60)
    
    create_sample_data()
    
    example_registry()
    
    example_loader()
    
    example_streaming()
    
    example_preprocessing()
    
    example_mixer()
    
    example_config()
    
    example_schema()
    
    print("\n" + "="*60)
    print("所有示例运行完成!")
    print("="*60)


if __name__ == "__main__":
    main()
