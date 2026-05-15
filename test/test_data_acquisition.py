"""
数据获取和处理系统测试
测试各个下载器、数据处理和数据集构建功能
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.data_acquisition.base_downloader import (
    BaseDownloader,
    DownloadProgress,
    FormatConverter,
)
from src.data_acquisition.poetry_downloader import PoetryDownloader, PoetryItem
from src.data_acquisition.classical_downloader import (
    ClassicalDownloader,
    ClassicalText,
    TextFilter,
)
from src.data_acquisition.modern_downloader import (
    ModernDownloader,
    ModernText,
    ModernTextFilter,
)
from src.data_acquisition.dataset_builder import (
    DatasetBuilder,
    DatasetConfig,
    DataRecord,
)


class TestBaseDownloader:
    """基础下载器测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def downloader(self, temp_dir):
        """创建下载器实例"""
        return BaseDownloader(base_dir=str(temp_dir), max_concurrent=2)

    def test_initialization(self, downloader, temp_dir):
        """测试初始化"""
        assert downloader.base_dir == temp_dir
        assert downloader.max_concurrent == 2
        assert downloader.rate_limit == 10.0

    def test_checkpoint_save_load(self, downloader, temp_dir):
        """测试检查点保存和加载"""
        progress = DownloadProgress()
        progress.total_size = 1000
        progress.downloaded_size = 500
        progress.status = 'downloading'

        downloader._download_history['http://example.com/test.json'] = progress
        downloader._save_checkpoint()

        new_downloader = BaseDownloader(base_dir=str(temp_dir))
        assert 'http://example.com/test.json' in new_downloader._download_history

    def test_format_size(self, downloader):
        """测试文件大小格式化"""
        assert 'B' in downloader._format_size(100)
        assert 'KB' in downloader._format_size(1024)
        assert 'MB' in downloader._format_size(1024 * 1024)
        assert 'GB' in downloader._format_size(1024 * 1024 * 1024)

    def test_format_duration(self, downloader):
        """测试时间格式化"""
        assert '秒' in downloader._format_duration(30)
        assert '分钟' in downloader._format_duration(120)
        assert '小时' in downloader._format_duration(3600)

    def test_statistics(self, downloader):
        """测试统计信息生成"""
        stats = downloader.get_statistics()
        assert 'total_tasks' in stats
        assert 'completed' in stats
        assert 'failed' in stats
        assert stats['total_tasks'] == 0


class TestFormatConverter:
    """格式转换器测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_json_to_jsonl_conversion(self, temp_dir):
        """测试JSON转JSONL"""
        source_file = temp_dir / "source.json"
        target_file = temp_dir / "target.jsonl"

        data = [
            {'title': '静夜思', 'author': '李白', 'content': '床前明月光'},
            {'title': '望庐山瀑布', 'author': '李白', 'content': '日照香炉生紫烟'},
        ]

        with open(source_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

        count = FormatConverter.convert(source_file, target_file, 'jsonl')

        assert count == 2
        assert target_file.exists()

        with open(target_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert '静夜思' in lines[0]

    def test_jsonl_to_json_conversion(self, temp_dir):
        """测试JSONL转JSON"""
        source_file = temp_dir / "source.jsonl"
        target_file = temp_dir / "target.json"

        with open(source_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'title': '春晓', 'author': '孟浩然'}, ensure_ascii=False) + '\n')
            f.write(json.dumps({'title': '相思', 'author': '王维'}, ensure_ascii=False) + '\n')

        count = FormatConverter.convert(source_file, target_file, 'json')

        assert count == 2
        assert target_file.exists()

        with open(target_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert len(data) == 2

    def test_json_to_txt_conversion(self, temp_dir):
        """测试JSON转TXT"""
        source_file = temp_dir / "source.json"
        target_file = temp_dir / "target.txt"

        data = [{'text': '第一段文本'}, {'text': '第二段文本'}]

        with open(source_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

        count = FormatConverter.convert(source_file, target_file, 'txt')

        assert count == 2
        assert target_file.exists()

        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert '第一段文本' in content
            assert '第二段文本' in content

    def test_transform_function(self, temp_dir):
        """测试转换函数"""
        source_file = temp_dir / "source.json"
        target_file = temp_dir / "target.jsonl"

        data = [{'content': '原内容'}]

        with open(source_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

        def transform(record):
            record['processed'] = True
            return record

        count = FormatConverter.convert(source_file, target_file, 'jsonl', transform_func=transform)

        with open(target_file, 'r', encoding='utf-8') as f:
            record = json.loads(f.readline())
            assert record['processed'] is True


class TestPoetryDownloader:
    """古诗词下载器测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def downloader(self, temp_dir):
        """创建下载器实例"""
        return PoetryDownloader(base_dir=str(temp_dir))

    def test_initialization(self, downloader):
        """测试初始化"""
        assert downloader.base_dir.exists()
        assert downloader._raw_dir.exists()
        assert downloader._processed_dir.exists()

    def test_poetry_sources_config(self, downloader):
        """测试数据源配置"""
        assert 'tangshi' in downloader.POETRY_SOURCES
        assert 'songci' in downloader.POETRY_SOURCES
        assert 'shijing' in downloader.POETRY_SOURCES

        config = downloader.POETRY_SOURCES['tangshi']
        assert config['dynasty'] == '唐'
        assert config['type'] == 'poetry'

    def test_remove_duplicates(self, downloader):
        """测试去重功能"""
        items = [
            PoetryItem(title='静夜思', author='李白', content='床前明月光', dynasty='唐', type='poetry'),
            PoetryItem(title='静夜思', author='李白', content='床前明月光', dynasty='唐', type='poetry'),
            PoetryItem(title='望庐山瀑布', author='李白', content='日照香炉生紫烟', dynasty='唐', type='poetry'),
        ]

        unique_items = downloader._remove_duplicates(items)
        assert len(unique_items) == 2

    def test_generate_statistics(self, downloader):
        """测试统计生成"""
        items = [
            PoetryItem(title='静夜思', author='李白', content='床前明月光，疑是地上霜', dynasty='唐', type='poetry'),
            PoetryItem(title='望庐山瀑布', author='李白', content='日照香炉生紫烟，遥看瀑布挂前川', dynasty='唐', type='poetry'),
        ]

        stats = downloader.generate_statistics(items)

        assert stats['total'] == 2
        assert stats['by_dynasty']['唐'] == 2
        assert stats['unique_authors'] == 1
        assert len(stats['top_authors']) == 1

    def test_export_to_jsonl(self, downloader, temp_dir):
        """测试导出JSONL"""
        items = [
            PoetryItem(
                title='春晓',
                author='孟浩然',
                content='春眠不觉晓，处处闻啼鸟',
                dynasty='唐',
                type='poetry',
                id='chunxiao_1'
            ),
        ]

        output_file = temp_dir / "test_poetry.jsonl"
        result = downloader.export_to_jsonl(items, output_file)

        assert result.exists()

        with open(result, 'r', encoding='utf-8') as f:
            line = f.readline()
            record = json.loads(line)
            assert record['title'] == '春晓'
            assert record['author'] == '孟浩然'

    def test_export_to_json(self, downloader, temp_dir):
        """测试导出JSON"""
        items = [
            PoetryItem(
                title='相思',
                author='王维',
                content='红豆生南国，春来发几枝',
                dynasty='唐',
                type='poetry',
                id='xiangsi_1'
            ),
        ]

        output_file = temp_dir / "test_poetry.json"
        result = downloader.export_to_json(items, output_file)

        assert result.exists()

        with open(result, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]['title'] == '相思'


class TestClassicalDownloader:
    """古文典籍下载器测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def downloader(self, temp_dir):
        """创建下载器实例"""
        return ClassicalDownloader(base_dir=str(temp_dir))

    def test_initialization(self, downloader):
        """测试初始化"""
        assert downloader.base_dir.exists()
        assert downloader._raw_dir.exists()
        assert downloader._processed_dir.exists()

    def test_classical_sources_config(self, downloader):
        """测试典籍数据源配置"""
        assert 'guwen_guanzhi' in downloader.CLASSICAL_SOURCES
        assert 'zhuangzi' in downloader.CLASSICAL_SOURCES

        config = downloader.CLASSICAL_SOURCES['guwen_guanzhi']
        assert config['name'] == '古文观止'
        assert config['dynasty'] == '明清'

    def test_is_classical_chinese(self, downloader):
        """测试文言文判断"""
        classical_text = "夫君子之行，静以修身，俭以养德"
        assert downloader._is_classical_chinese(classical_text) is True

        mixed_text = "这是现代白话文"
        result = downloader._is_classical_chinese(mixed_text)
        assert isinstance(result, bool)

    def test_normalize_text(self, downloader):
        """测试文本标准化"""
        texts = [
            ClassicalText(
                title='　　第一章　　',
                author='佚名',
                content='　　床前明月光，\n\n\n　　疑是地上霜。　　',
                source='测试',
                dynasty='唐',
            ),
        ]

        normalized = downloader.normalize_text(texts)
        assert '第一章' in normalized[0].title
        assert '床前明月光' in normalized[0].content

    def test_remove_duplicates(self, downloader):
        """测试去重功能"""
        texts = [
            ClassicalText(
                title='第一篇',
                author='韩愈',
                content='古之学者必有师',
                source='古文观止',
                dynasty='唐',
            ),
            ClassicalText(
                title='第一篇',
                author='韩愈',
                content='古之学者必有师',
                source='古文观止',
                dynasty='唐',
            ),
            ClassicalText(
                title='第二篇',
                author='韩愈',
                content='师者，所以传道受业解惑也',
                source='古文观止',
                dynasty='唐',
            ),
        ]

        unique_texts = downloader.remove_duplicates(texts)
        assert len(unique_texts) == 2

    def test_text_filter(self, downloader):
        """测试文本过滤"""
        texts = [
            ClassicalText(
                title='短文',
                author='作者1',
                content='短内容',
                source='测试',
                dynasty='唐',
            ),
            ClassicalText(
                title='长文',
                author='作者2',
                content='这是较长的内容，包含更多的文字用于测试过滤功能是否正常工作。' * 5,
                source='测试',
                dynasty='唐',
            ),
        ]

        text_filter = TextFilter(min_length=20, max_length=1000)
        filtered = downloader._apply_filter(texts, text_filter)

        assert len(filtered) == 1
        assert filtered[0].title == '长文'

    def test_generate_statistics(self, downloader):
        """测试统计生成"""
        texts = [
            ClassicalText(title='篇一', author='韩愈', content='内容一', source='古文观止', dynasty='唐', category='古文'),
            ClassicalText(title='篇二', author='韩愈', content='内容二', source='古文观止', dynasty='唐', category='古文'),
            ClassicalText(title='篇三', author='庄子', content='内容三', source='庄子', dynasty='先秦', category='子部'),
        ]

        stats = downloader.generate_statistics(texts)

        assert stats['total'] == 3
        assert stats['by_dynasty']['唐'] == 2
        assert stats['by_dynasty']['先秦'] == 1
        assert stats['unique_authors'] == 2

    def test_export_to_jsonl(self, downloader, temp_dir):
        """测试导出JSONL"""
        texts = [
            ClassicalText(
                title='师说',
                author='韩愈',
                content='古之学者必有师',
                source='古文观止',
                dynasty='唐',
                category='古文',
            ),
        ]

        output_file = temp_dir / "test_classical.jsonl"
        result = downloader.export_to_jsonl(texts, output_file)

        assert result.exists()

        with open(result, 'r', encoding='utf-8') as f:
            line = f.readline()
            record = json.loads(line)
            assert record['title'] == '师说'
            assert record['author'] == '韩愈'


class TestModernDownloader:
    """现代文下载器测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def downloader(self, temp_dir):
        """创建下载器实例"""
        return ModernDownloader(base_dir=str(temp_dir))

    def test_initialization(self, downloader):
        """测试初始化"""
        assert downloader.base_dir.exists()
        assert downloader._raw_dir.exists()
        assert downloader._processed_dir.exists()

    def test_modern_sources_config(self, downloader):
        """测试现代文数据源配置"""
        assert 'wiki_zh' in downloader.MODERN_SOURCES
        assert 'news_chinese' in downloader.MODERN_SOURCES

        config = downloader.MODERN_SOURCES['news_chinese']
        assert config['name'] == '中文新闻语料'
        assert config['category'] == '新闻'

    def test_count_chinese_words(self, downloader):
        """测试中文字数统计"""
        text = "床前明月光，疑是地上霜"
        count = downloader._count_chinese_words(text)
        assert count == 10

    def test_check_quality(self, downloader):
        """测试文本质量检查"""
        quality_text = ModernText(
            title='测试文章',
            author='作者',
            content='这是一段正常的中文文本。包含足够多的汉字来通过质量检查。' * 5,
            source='测试',
            category='测试',
            word_count=100,
        )
        assert downloader._check_quality(quality_text) is True

        low_quality_text = ModernText(
            title='低质量',
            author='作者',
            content='啊啊啊' * 50,
            source='测试',
            category='测试',
            word_count=10,
        )
        assert downloader._check_quality(low_quality_text) is False

    def test_normalize_content(self, downloader):
        """测试内容标准化"""
        content = '这是　　内容   \n\n\n包含\t\t多个\n\n空行。\n\nhttps://example.com\n@user #topic#'
        normalized = downloader._normalize_content(content)

        assert '　' not in normalized
        assert '\n\n\n' not in normalized
        assert 'https://example.com' not in normalized
        assert '@user' not in normalized

    def test_remove_duplicates(self, downloader):
        """测试去重功能"""
        texts = [
            ModernText(title='文1', author='作者1', content='相同内容', source='测试', category='测试', word_count=10),
            ModernText(title='文1', author='作者1', content='相同内容', source='测试', category='测试', word_count=10),
            ModernText(title='文2', author='作者2', content='不同内容', source='测试', category='测试', word_count=10),
        ]

        unique_texts = downloader.remove_duplicates(texts)
        assert len(unique_texts) == 2

    def test_generate_statistics(self, downloader):
        """测试统计生成"""
        texts = [
            ModernText(title='文1', author='作者1', content='内容一' * 20, source='知乎', category='问答', word_count=80),
            ModernText(title='文2', author='作者2', content='内容二' * 30, source='知乎', category='问答', word_count=120),
            ModernText(title='文3', author='作者3', content='内容三' * 25, source='维基', category='百科', word_count=100),
        ]

        stats = downloader.generate_statistics(texts)

        assert stats['total'] == 3
        assert stats['by_category']['问答'] == 2
        assert stats['by_source']['知乎'] == 2
        assert stats['unique_authors'] == 3


class TestDatasetBuilder:
    """数据集构建器测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self, temp_dir):
        """创建数据集配置"""
        return DatasetConfig(
            name='test_dataset',
            output_dir=str(temp_dir / 'datasets'),
            output_format='jsonl',
            min_length=5,
            max_length=500,
        )

    @pytest.fixture
    def builder(self, config):
        """创建数据集构建器"""
        return DatasetBuilder(config)

    def test_initialization(self, builder, temp_dir):
        """测试初始化"""
        assert builder.config.name == 'test_dataset'
        assert builder.output_dir.exists()

    def test_add_poetry(self, builder):
        """测试添加诗词"""
        items = [
            PoetryItem(title='静夜思', author='李白', content='床前明月光，疑是地上霜', dynasty='唐', type='poetry'),
            PoetryItem(title='春晓', author='孟浩然', content='春眠不觉晓，处处闻啼鸟', dynasty='唐', type='poetry'),
        ]

        count = builder.add_poetry(items)
        assert count == 2
        assert builder.get_record_count() == 2

        records = builder.get_records('poetry')
        assert len(records) == 2
        assert records[0].category == 'poetry'
        assert records[0].subcategory == 'poetry'

    def test_add_classical(self, builder):
        """测试添加古文"""
        texts = [
            ClassicalText(title='师说', author='韩愈', content='古之学者必有师', source='古文观止', dynasty='唐'),
        ]

        count = builder.add_classical(texts)
        assert count == 1
        assert builder.get_record_count() == 1

        records = builder.get_records('classical')
        assert len(records) == 1
        assert records[0].category == 'classical'

    def test_add_modern(self, builder):
        """测试添加现代文"""
        texts = [
            ModernText(title='文章', author='作者', content='这是现代文内容', source='知乎', category='问答', word_count=50),
        ]

        count = builder.add_modern(texts)
        assert count == 1
        assert builder.get_record_count() == 1

        records = builder.get_records('modern')
        assert len(records) == 1
        assert records[0].category == 'modern'

    def test_length_filtering(self, builder):
        """测试长度过滤"""
        builder.add_poetry([
            PoetryItem(title='短', author='作者', content='短', dynasty='唐', type='poetry'),
            PoetryItem(title='长', author='作者', content='这是一段比较长的内容' * 20, dynasty='唐', type='poetry'),
            PoetryItem(title='中', author='作者', content='中等长度内容' * 5, dynasty='唐', type='poetry'),
        ])

        removed = builder.filter_by_length(min_length=10, max_length=100)
        assert removed >= 1
        assert builder.get_record_count() <= 3

    def test_split_dataset(self, builder):
        """测试数据集分割"""
        for i in range(100):
            builder.add_poetry([
                PoetryItem(
                    title=f'诗词{i}',
                    author='作者',
                    content=f'诗词内容{i}',
                    dynasty='唐',
                    type='poetry'
                )
            ])

        splits = builder.split_dataset(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

        assert len(splits['train']) == 80
        assert len(splits['val']) == 10
        assert len(splits['test']) == 10

    def test_remove_duplicates(self, builder):
        """测试去重"""
        builder.clear_records()
        builder.add_poetry([
            PoetryItem(title='诗1', author='李白', content='床前明月光', dynasty='唐', type='poetry'),
            PoetryItem(title='诗1', author='李白', content='床前明月光', dynasty='唐', type='poetry'),
            PoetryItem(title='诗2', author='杜甫', content='疑是地上霜', dynasty='唐', type='poetry'),
        ])

        initial_count = builder.get_record_count()
        removed = builder.remove_duplicates()
        final_count = builder.get_record_count()

        assert removed == 1
        assert final_count == 2
        assert initial_count == 3

    def test_export_jsonl(self, builder, temp_dir):
        """测试导出JSONL"""
        builder.add_poetry([
            PoetryItem(title='静夜思', author='李白', content='床前明月光', dynasty='唐', type='poetry'),
        ])

        output_file = temp_dir / 'datasets' / 'test_export.jsonl'
        result = builder.export_to_jsonl(output_file)

        assert result.exists()

        with open(result, 'r', encoding='utf-8') as f:
            line = f.readline()
            record = json.loads(line)
            assert record['text'] == '床前明月光'
            assert record['author'] == '李白'

    def test_export_json(self, builder, temp_dir):
        """测试导出JSON"""
        builder.add_poetry([
            PoetryItem(title='春晓', author='孟浩然', content='春眠不觉晓', dynasty='唐', type='poetry'),
        ])

        builder.config.output_format = 'json'
        output_file = temp_dir / 'datasets' / 'test_export.json'
        result = builder.export_to_json(output_file)

        assert result.exists()

        with open(result, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]['title'] == '春晓'

    def test_export_txt(self, builder, temp_dir):
        """测试导出TXT"""
        builder.clear_records()
        builder.add_poetry([
            PoetryItem(title='诗1', author='作者', content='第一首诗的内容，这是一个足够长的诗歌内容，用于测试导出功能是否正常工作。' * 2, dynasty='唐', type='poetry'),
            PoetryItem(title='诗2', author='作者', content='第二首诗的内容，这是一个足够长的诗歌内容，用于测试导出功能是否正常工作。' * 2, dynasty='唐', type='poetry'),
        ])

        output_file = temp_dir / 'datasets' / 'test_export.txt'
        result = builder.export_to_txt(output_file, separator='\n\n---\n\n')

        assert result.exists()

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()
            assert '第一首诗的内容' in content
            assert '第二首诗的内容' in content

    def test_export_splits(self, builder, temp_dir):
        """测试导出分割数据集"""
        builder.clear_records()
        for i in range(20):
            builder.add_poetry([
                PoetryItem(title=f'诗{i}', author='作者', content=f'内容{i}' * 10, dynasty='唐', type='poetry')
            ])

        outputs = builder.export_splits(prefix='split_test')

        assert 'train' in outputs
        assert 'val' in outputs
        assert 'test' in outputs

        for split_file in outputs.values():
            assert split_file.exists()

    def test_statistics_generation(self, builder):
        """测试统计生成"""
        builder.clear_records()
        builder.add_poetry([
            PoetryItem(title='诗1', author='李白', content='床前明月光，疑是地上霜。举头望明月，低头思故乡。这是一首著名的唐诗。' * 5, dynasty='唐', type='poetry'),
        ])
        builder.add_classical([
            ClassicalText(title='文1', author='韩愈', content='古文观止是清代吴乘权编选的散文集，选录东周至明代的文章222篇。这是一部著名的古文选本。' * 5, source='古文', dynasty='唐'),
        ])

        stats = builder.generate_statistics()

        assert stats['total_records'] == 2
        assert 'by_category' in stats
        assert stats['by_category']['poetry'] == 1
        assert stats['by_category']['classical'] == 1

    def test_save_statistics(self, builder, temp_dir):
        """测试保存统计报告"""
        builder.clear_records()
        builder.add_poetry([
            PoetryItem(title='诗', author='作者', content='内容' * 10, dynasty='唐', type='poetry'),
        ])

        stats_file = temp_dir / 'datasets' / 'test_stats.json'
        result = builder.save_statistics(stats_file)

        assert result.exists()

        with open(result, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert 'total_records' in data
            assert 'generated_at' in data

    def test_print_summary(self, builder, capsys):
        """测试打印摘要"""
        builder.add_poetry([
            PoetryItem(title='诗', author='作者', content='内容' * 10, dynasty='唐', type='poetry'),
        ])

        builder.print_summary()
        captured = capsys.readouterr()
        assert '数据集摘要' in captured.out
        assert '总记录数' in captured.out


class TestDataLoader:
    """数据加载器测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_load_jsonl(self, temp_dir):
        """测试加载JSONL"""
        from src.data_acquisition.dataset_builder import DataLoader

        file_path = temp_dir / 'test.jsonl'
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'title': '静夜思', 'author': '李白'}, ensure_ascii=False) + '\n')
            f.write(json.dumps({'title': '春晓', 'author': '孟浩然'}, ensure_ascii=False) + '\n')

        records = DataLoader.load_jsonl(file_path)
        assert len(records) == 2
        assert records[0]['title'] == '静夜思'

    def test_load_json(self, temp_dir):
        """测试加载JSON"""
        from src.data_acquisition.dataset_builder import DataLoader

        file_path = temp_dir / 'test.json'
        data = [
            {'title': '静夜思', 'author': '李白'},
            {'title': '春晓', 'author': '孟浩然'},
        ]

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

        records = DataLoader.load_json(file_path)
        assert len(records) == 2
        assert records[1]['author'] == '孟浩然'

    def test_load_txt(self, temp_dir):
        """测试加载TXT"""
        from src.data_acquisition.dataset_builder import DataLoader

        file_path = temp_dir / 'test.txt'
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('第一段\n\n---\n\n第二段')

        records = DataLoader.load_txt(file_path, separator='\n\n---\n\n')
        assert len(records) == 2
        assert records[0] == '第一段'

    def test_load_dataset(self, temp_dir):
        """测试智能加载"""
        from src.data_acquisition.dataset_builder import DataLoader

        file_path = temp_dir / 'test.jsonl'
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'text': '内容1'}, ensure_ascii=False) + '\n')
            f.write(json.dumps({'text': '内容2'}, ensure_ascii=False) + '\n')

        records = DataLoader.load_dataset(file_path)
        assert len(records) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
