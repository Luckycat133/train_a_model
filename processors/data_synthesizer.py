#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据合成处理器模块

该模块专注于处理collection目录中的原始数据，包括：
1. 读取多种来源的古典文学数据
2. 清洗和标准化数据格式
3. 合成和整合不同来源的数据
4. 为模型训练准备高质量的数据集
"""

import os
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Tuple

class DataSynthesizer:
    """
    数据合成处理器类
    
    负责处理和合成来自不同来源的古典文学数据
    """
    
    def __init__(self, config=None, text_cleaner=None):
        """
        初始化数据合成处理器
        
        Args:
            config: 配置字典，可选
            text_cleaner: 文本清洗器实例，可选
        """
        self.logger = logging.getLogger("LingmaoMoyun.DataSynthesizer")
        self.config = config or {}
        self.text_cleaner = text_cleaner
        
        # 设置输入和输出目录
        self.input_dir = Path(self.config.get("paths", {}).get("input_dir", "collection"))
        self.output_dir = Path(self.config.get("paths", {}).get("output_dir", "dataset"))
        self.temp_dir = Path(self.config.get("paths", {}).get("temp_dir", "temp"))
        
        # 确保目录存在
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        
        # 处理配置
        self.batch_size = self.config.get("processor", {}).get("batch_size", 1000)
        self.max_workers = self.config.get("processor", {}).get("max_workers", 8)
        self.file_extensions = self.config.get("processor", {}).get("file_extensions", [".json", ".txt"])
        
        self.logger.info(f"数据合成处理器初始化完成，输入目录: {self.input_dir}")
    
    def process(self, force=False) -> bool:
        """
        处理并合成数据
        
        Args:
            force: 是否强制重新处理，即使输出文件已存在
            
        Returns:
            是否处理成功
        """
        try:
            self.logger.info("开始数据处理与合成流程...")
            
            # 检查输出目录是否已有处理结果
            if not force:
                output_files = ["poems.jsonl", "prose.jsonl", "chapters.jsonl", "authors.jsonl"]
                all_exist = True
                for file in output_files:
                    output_path = self.output_dir / file
                    if not output_path.exists() or output_path.stat().st_size == 0:
                        all_exist = False
                        break
                
                if all_exist:
                    self.logger.info("输出文件已存在且不为空，跳过处理。使用--force选项可强制重新处理。")
                    return True
            
            # 处理所有数据源
            results = self.process_all_sources()
            
            # 保存结果
            self.save_results(results)
            
            self.logger.info("数据处理与合成完成")
            return True
        except Exception as e:
            self.logger.error(f"数据处理失败: {e}", exc_info=True)
            return False
    
    def process_all_sources(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        处理所有数据源
        
        Returns:
            按类别组织的数据字典
        """
        self.logger.info("开始处理所有数据源")
        
        # 初始化结果字典
        results = {
            "poems": [],      # 诗词
            "prose": [],     # 散文
            "chapters": [],  # 章节
            "authors": [],   # 作者
            "other": []      # 其他
        }
        
        # 处理Chinese-Poetry项目数据
        chinese_poetry_data = self._process_chinese_poetry()
        for category, items in chinese_poetry_data.items():
            results[category].extend(items)
        
        # 处理Poems-DB全面古诗词古文库
        poems_db_data = self._process_poems_db()
        for category, items in poems_db_data.items():
            results[category].extend(items)
        
        # 处理ESE-Gushiwen中华古诗文数据库
        gushiwen_data = self._process_gushiwen()
        for category, items in gushiwen_data.items():
            results[category].extend(items)
        
        # 处理殆知阁古代文献数据集
        daizhige_data = self._process_daizhige()
        for category, items in daizhige_data.items():
            results[category].extend(items)
        
        # 记录处理结果
        for category, items in results.items():
            self.logger.info(f"已处理 {category} 数据: {len(items)} 条")
        
        return results
    
    def _process_chinese_poetry(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        处理Chinese-Poetry项目数据
        
        Returns:
            处理后的数据字典
        """
        self.logger.info("开始处理Chinese-Poetry项目数据")
        
        results = {
            "poems": [],
            "prose": [],
            "authors": [],
            "other": []
        }
        
        chinese_poetry_dir = self.input_dir / "chinese-poetry-master"
        if not chinese_poetry_dir.exists():
            self.logger.warning(f"Chinese-Poetry项目目录不存在: {chinese_poetry_dir}")
            return results
        
        # 处理唐诗
        tang_poetry_dir = chinese_poetry_dir / "全唐诗"
        if tang_poetry_dir.exists():
            tang_poems = self._process_tang_poetry(tang_poetry_dir)
            results["poems"].extend(tang_poems)
        
        # 处理宋词
        song_ci_dir = chinese_poetry_dir / "宋词"
        if song_ci_dir.exists():
            song_ci = self._process_song_ci(song_ci_dir)
            results["poems"].extend(song_ci)
        
        # 处理作者信息
        authors_info = self._extract_authors_from_chinese_poetry(chinese_poetry_dir)
        results["authors"].extend(authors_info)
        
        self.logger.info(f"Chinese-Poetry项目数据处理完成，共获取 {len(results['poems'])} 首诗词")
        return results
    
    def _process_tang_poetry(self, directory: Path) -> List[Dict[str, Any]]:
        """
        处理唐诗数据
        
        Args:
            directory: 唐诗数据目录
            
        Returns:
            处理后的唐诗列表
        """
        poems = []
        
        # 遍历目录中的JSON文件
        for json_file in directory.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 处理每首诗
                for poem_data in data:
                    if not isinstance(poem_data, dict):
                        continue
                    
                    # 提取基本信息
                    poem = {
                        "title": poem_data.get("title", ""),
                        "author": poem_data.get("author", ""),
                        "dynasty": "唐",
                        "content": "\n".join(poem_data.get("paragraphs", [])),
                        "source": "chinese-poetry",
                        "type": "poem"
                    }
                    
                    # 清洗文本
                    if self.text_cleaner:
                        poem["content"] = self.text_cleaner.clean_text(poem["content"])
                    
                    poems.append(poem)
            except Exception as e:
                self.logger.error(f"处理唐诗文件出错 {json_file}: {e}")
        
        return poems
    
    def _process_song_ci(self, directory: Path) -> List[Dict[str, Any]]:
        """
        处理宋词数据
        
        Args:
            directory: 宋词数据目录
            
        Returns:
            处理后的宋词列表
        """
        ci_list = []
        
        # 遍历目录中的JSON文件
        for json_file in directory.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 处理每首词
                for ci_data in data:
                    if not isinstance(ci_data, dict):
                        continue
                    
                    # 提取基本信息
                    ci = {
                        "title": ci_data.get("rhythmic", ""),  # 词牌名
                        "author": ci_data.get("author", ""),
                        "dynasty": "宋",
                        "content": "\n".join(ci_data.get("paragraphs", [])),
                        "source": "chinese-poetry",
                        "type": "ci"
                    }
                    
                    # 清洗文本
                    if self.text_cleaner:
                        ci["content"] = self.text_cleaner.clean_text(ci["content"])
                    
                    ci_list.append(ci)
            except Exception as e:
                self.logger.error(f"处理宋词文件出错 {json_file}: {e}")
        
        return ci_list
    
    def _extract_authors_from_chinese_poetry(self, directory: Path) -> List[Dict[str, Any]]:
        """
        从Chinese-Poetry项目中提取作者信息
        
        Args:
            directory: Chinese-Poetry项目目录
            
        Returns:
            作者信息列表
        """
        authors = []
        
        # 处理作者信息文件
        author_files = [
            directory / "全唐诗" / "authors.tang.json",
            directory / "宋词" / "authors.song.json"
        ]
        
        for author_file in author_files:
            if not author_file.exists():
                continue
                
            try:
                with open(author_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                dynasty = "唐" if "tang" in author_file.name else "宋"
                
                for author_data in data:
                    if not isinstance(author_data, dict):
                        continue
                        
                    author = {
                        "name": author_data.get("name", ""),
                        "dynasty": dynasty,
                        "description": author_data.get("desc", ""),
                        "short_description": author_data.get("short_description", ""),
                        "source": "chinese-poetry"
                    }
                    
                    authors.append(author)
            except Exception as e:
                self.logger.error(f"处理作者文件出错 {author_file}: {e}")
        
        return authors
    
    def _process_poems_db(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        处理Poems-DB全面古诗词古文库
        
        Returns:
            处理后的数据字典
        """
        self.logger.info("开始处理Poems-DB全面古诗词古文库")
        
        results = {
            "poems": [],
            "prose": [],
            "chapters": [],
            "authors": [],
            "other": []
        }
        
        poems_db_dir = self.input_dir / "poems-db-master"
        if not poems_db_dir.exists():
            self.logger.warning(f"Poems-DB项目目录不存在: {poems_db_dir}")
            return results
        
        # 处理诗词数据
        poem_files = [
            poems_db_dir / "poems1.json",
            poems_db_dir / "poems2.json",
            poems_db_dir / "poems3.json",
            poems_db_dir / "poems4.json"
        ]
        
        # 添加所有可能的文件模式
        potential_files = list(poems_db_dir.glob("poems*.json"))
        poem_files.extend([p for p in potential_files if p not in poem_files])
        
        successful_files = 0
        for poem_file in poem_files:
            if not poem_file.exists():
                continue
                
            try:
                # 添加健壮的JSON解析
                self.logger.info(f"处理Poems-DB文件: {poem_file}")
                
                # 尝试不同的编码方式
                data = None
                encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]
                
                for encoding in encodings:
                    try:
                        with open(poem_file, "r", encoding=encoding) as f:
                            try:
                                data = json.load(f)
                                self.logger.info(f"成功以{encoding}编码解析 {poem_file}")
                                break
                            except json.JSONDecodeError as e:
                                self.logger.debug(f"无法以{encoding}编码解析 {poem_file}: {e}")
                                continue
                    except Exception as e:
                        self.logger.debug(f"无法以{encoding}编码打开 {poem_file}: {e}")
                
                # 如果所有编码都失败，尝试修复内容
                if data is None:
                    try:
                        with open(poem_file, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        
                        # 修复常见JSON问题
                        content = content.replace("\\'", "'").replace('\\"', '"')
                        content = content.replace("\n", " ").replace("\r", " ")
                        
                        # 尝试添加缺失的方括号
                        if not content.startswith("["):
                            content = "[" + content
                        if not content.endswith("]"):
                            content = content + "]"
                        
                        # 尝试解析修复后的内容
                        data = json.loads(content)
                        self.logger.info(f"通过内容修复成功解析 {poem_file}")
                    except json.JSONDecodeError as e:
                        self.logger.error(f"修复后仍无法解析 {poem_file}: {e}")
                        
                        # 尝试逐行解析作为最后手段
                        data = []
                        with open(poem_file, "r", encoding="utf-8", errors="ignore") as f:
                            json_objects = []
                            current_object = ""
                            bracket_count = 0
                            
                            for line_num, line in enumerate(f, 1):
                                for char in line:
                                    if char == '{':
                                        bracket_count += 1
                                        current_object += char
                                    elif char == '}':
                                        bracket_count -= 1
                                        current_object += char
                                        if bracket_count == 0:
                                            try:
                                                obj = json.loads(current_object)
                                                json_objects.append(obj)
                                                current_object = ""
                                            except:
                                                pass
                                    elif bracket_count > 0:
                                        current_object += char
                            
                            if json_objects:
                                data = json_objects
                                self.logger.info(f"通过字符级解析获取了 {len(data)} 个对象")
                
                # 验证和处理数据
                if data is None:
                    self.logger.error(f"无法通过任何方式解析 {poem_file}")
                    continue
                
                # 确保数据是列表
                if not isinstance(data, list):
                    if isinstance(data, dict):
                        data = [data]
                    else:
                        self.logger.warning(f"数据不是列表或字典，跳过: {poem_file}")
                        continue
                
                # 处理每一个诗词数据
                for poem_data in data:
                    if not isinstance(poem_data, dict):
                        continue
                        
                    # 规范化处理
                    normalized_poem = self._normalize_poem_data(poem_data)
                    
                    # 根据内容分类
                    content = normalized_poem.get("content", "")
                    if self._is_poem(content):
                        results["poems"].append(normalized_poem)
                    else:
                        results["prose"].append(normalized_poem)
                
                successful_files += 1
                
            except Exception as e:
                self.logger.error(f"处理Poems-DB文件失败 {poem_file}: {e}", exc_info=True)
        
        # 处理诗词典籍部分
        if successful_files > 0:
            self.logger.info(f"成功处理了 {successful_files} 个Poems-DB文件")
            try:
                self._process_poems_db_books(poems_db_dir, results)
            except Exception as e:
                self.logger.error(f"处理Poems-DB典籍文件失败: {e}", exc_info=True)
        
        self.logger.info(f"Poems-DB数据处理完成，共获取 {len(results['poems'])} 首诗词，{len(results['prose'])} 篇散文")
        return results
    
    def _normalize_poem_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """规范化诗词数据"""
        result = {}
        
        # 确保所有字段都是字符串类型
        for key, value in data.items():
            if value is None:
                result[key] = ""
            elif isinstance(value, (int, float, bool)):
                result[key] = str(value)
            elif isinstance(value, (list, dict)):
                result[key] = json.dumps(value, ensure_ascii=False)
            else:
                result[key] = str(value)
        
        # 确保必要字段存在
        required_fields = ["title", "author", "dynasty", "content"]
        for field in required_fields:
            if field not in result:
                result[field] = ""
        
        return result
    
    def _process_poems_db_books(self, directory: Path, results: Dict[str, List[Dict[str, Any]]]):
        """
        处理Poems-DB古籍数据
        
        Args:
            directory: 古籍数据目录
            results: 结果字典
        """
        # 遍历目录中的文本文件
        for txt_file in directory.glob("**/*.txt"):
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # 提取基本信息
                book_name = txt_file.stem
                category = directory.name
                
                # 分章节处理
                chapters = self._split_into_chapters(content)
                
                for i, chapter_content in enumerate(chapters):
                    chapter = {
                        "title": f"{book_name}_{i+1}",
                        "book": book_name,
                        "category": category,
                        "content": chapter_content,
                        "source": "poems-db",
                        "type": "chapter"
                    }
                    
                    # 清洗文本
                    if self.text_cleaner:
                        chapter["content"] = self.text_cleaner.clean_text(chapter["content"])
                    
                    results["chapters"].append(chapter)
            except Exception as e:
                self.logger.error(f"处理Poems-DB古籍文件出错 {txt_file}: {e}")
    
    def _split_into_chapters(self, content: str) -> List[str]:
        """
        将内容分割为章节
        
        Args:
            content: 原始内容
            
        Returns:
            章节列表
        """
        # 简单实现：按照一定长度分割
        max_chapter_length = 5000  # 每章最大字符数
        
        if len(content) <= max_chapter_length:
            return [content]
        
        # 尝试按照章节标记分割
        chapter_markers = ["第[一二三四五六七八九十百千]+章", "卷[一二三四五六七八九十百千]+", r"^\s*\d+\s*$"]
        
        # 如果没有明确的章节标记，按照段落和长度分割
        paragraphs = content.split("\n\n")
        chapters = []
        current_chapter = ""
        
        for para in paragraphs:
            if len(current_chapter) + len(para) > max_chapter_length:
                chapters.append(current_chapter)
                current_chapter = para
            else:
                current_chapter += "\n\n" + para if current_chapter else para
        
        if current_chapter:
            chapters.append(current_chapter)
        
        return chapters
    
    def _process_gushiwen(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        处理ESE-Gushiwen中华古诗文数据库
        
        Returns:
            处理后的数据字典
        """
        self.logger.info("开始处理ESE-Gushiwen中华古诗文数据库")
        
        results = {
            "poems": [],
            "prose": [],
            "authors": [],
            "other": []
        }
        
        gushiwen_dir = self.input_dir / "chinese-gushiwen-master"
        if not gushiwen_dir.exists():
            self.logger.warning(f"ESE-Gushiwen项目目录不存在: {gushiwen_dir}")
            return results
        
        # 处理古文数据
        guwen_dir = gushiwen_dir / "guwen"
        if guwen_dir.exists():
            self._process_gushiwen_texts(guwen_dir, results)
        
        # 处理名句数据
        sentence_dir = gushiwen_dir / "sentence"
        if sentence_dir.exists():
            self._process_gushiwen_sentences(sentence_dir, results)
        
        # 处理作者数据
        writer_dir = gushiwen_dir / "writer"
        if writer_dir.exists():
            self._process_gushiwen_writers(writer_dir, results)
        
        self.logger.info(f"ESE-Gushiwen数据处理完成，共获取 {len(results['poems'])} 首诗词，{len(results['prose'])} 篇散文")
        return results
    
    def _process_gushiwen_texts(self, directory: Path, results: Dict[str, List[Dict[str, Any]]]):
        """
        处理古诗文数据
        
        Args:
            directory: 古诗文数据目录
            results: 结果字典
        """
        self.logger.info(f"处理古诗文数据目录: {directory}")
        
        # 遍历所有JSON文件
        json_files = list(directory.glob("**/*.json"))
        self.logger.info(f"找到 {len(json_files)} 个古诗文JSON文件")
        
        # 处理统计
        processed_count = 0
        error_count = 0
        
        for json_file in json_files:
            try:
                # 尝试不同的编码方式
                data = None
                for encoding in ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]:
                    try:
                        with open(json_file, "r", encoding=encoding) as f:
                            try:
                                data = json.load(f)
                                self.logger.debug(f"成功以{encoding}编码解析: {json_file}")
                                break
                            except json.JSONDecodeError:
                                continue
                    except Exception:
                        continue
                
                # 如果无法解析，尝试修复
                if data is None:
                    try:
                        with open(json_file, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        
                        # 修复常见JSON问题
                        content = content.replace("\\'", "'").replace('\\"', '"')
                        content = content.replace("\n", " ").replace("\r", " ")
                        
                        # 尝试解析修复后的内容
                        data = json.loads(content)
                        self.logger.info(f"通过内容修复成功解析: {json_file}")
                    except Exception as e:
                        self.logger.error(f"无法解析文件: {json_file}, 错误: {e}")
                        error_count += 1
                        continue
                
                # 处理数据
                if not isinstance(data, dict):
                    if isinstance(data, list) and data:
                        # 如果是列表，尝试处理每个元素
                        for item in data:
                            if isinstance(item, dict):
                                self._process_gushiwen_item(item, json_file, results)
                                processed_count += 1
                        continue
                    else:
                        self.logger.warning(f"数据格式不正确: {json_file}")
                        error_count += 1
                        continue
                
                # 处理单个字典对象
                self._process_gushiwen_item(data, json_file, results)
                processed_count += 1
                
            except Exception as e:
                self.logger.error(f"处理文件出错: {json_file}, 错误: {e}")
                error_count += 1
        
        self.logger.info(f"古诗文处理完成，成功: {processed_count}，失败: {error_count}")
    
    def _process_gushiwen_item(self, data: Dict[str, Any], source_file: Path, results: Dict[str, List[Dict[str, Any]]]):
        """处理单个古诗文数据项"""
        # 规范化数据
        text_data = self._normalize_gushiwen_data(data, source_file)
        
        # 根据内容判断类型并添加到对应结果
        content = text_data.get("content", "")
        if not content:
            # 无内容，跳过
            return
            
        if self._is_poem(content):
            text_data["type"] = "poem"
            results["poems"].append(text_data)
        else:
            text_data["type"] = "prose"
            results["prose"].append(text_data)
    
    def _normalize_gushiwen_data(self, data: Dict[str, Any], source_file: Path) -> Dict[str, Any]:
        """规范化古诗文数据"""
        # 初始化结果
        result = {
            "source": f"gushiwen-{source_file.name}"
        }
        
        # 字段映射表
        field_mapping = {
            "title": ["title", "Title", "name", "Name"],
            "author": ["author", "Author", "writer", "Writer"],
            "dynasty": ["dynasty", "Dynasty", "time", "Time", "period", "Period"],
            "content": ["content", "Content", "body", "Body", "text", "Text"],
            "translation": ["translation", "Translation", "translate", "Translate"],
            "annotation": ["annotation", "Annotation", "note", "Note", "notes", "Notes"],
            "appreciation": ["appreciation", "Appreciation", "analysis", "Analysis"]
        }
        
        # 从原始数据中提取字段
        for target_field, source_fields in field_mapping.items():
            # 尝试所有可能的源字段
            for source_field in source_fields:
                if source_field in data:
                    value = data[source_field]
                    # 规范化值的类型
                    if value is None:
                        result[target_field] = ""
                    elif isinstance(value, (int, float, bool)):
                        result[target_field] = str(value)
                    elif isinstance(value, (list, dict)):
                        result[target_field] = json.dumps(value, ensure_ascii=False)
                    else:
                        result[target_field] = str(value)
                    break
            
            # 如果没有找到对应字段，设置为空字符串
            if target_field not in result:
                result[target_field] = ""
        
        # 处理特殊情况 - 从paragraphs构建内容
        if not result.get("content") and "paragraphs" in data and isinstance(data["paragraphs"], list):
            paragraphs = []
            for p in data["paragraphs"]:
                if isinstance(p, str):
                    paragraphs.append(p)
                elif isinstance(p, dict) and "p" in p:
                    paragraphs.append(p["p"])
            result["content"] = "\n".join(paragraphs)
        
        # 清洗文本
        if self.text_cleaner and result.get("content"):
            result["content"] = self.text_cleaner.clean_text(result["content"])
        
        return result
    
    def _process_gushiwen_sentences(self, directory: Path, results: Dict[str, List[Dict[str, Any]]]):
        """
        处理名句数据
        
        Args:
            directory: 名句数据目录
            results: 结果字典
        """
        # 遍历目录中的JSON文件
        for json_file in directory.glob("**/*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        # 验证数据是否为列表
                        if not isinstance(data, list):
                            self.logger.warning(f"名句文件格式错误，应为列表: {json_file}")
                            data = [data] if isinstance(data, dict) else []
                    except json.JSONDecodeError as e:
                        self.logger.error(f"解析名句文件失败 {json_file}: {e}")
                        # 尝试逐行解析
                        data = []
                        f.seek(0)  # 重置文件指针到开头
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                item = json.loads(line)
                                data.append(item)
                            except json.JSONDecodeError:
                                self.logger.warning(f"解析名句文件第{line_num}行失败: {json_file}")
                
                for sentence_data in data:
                    if not isinstance(sentence_data, dict):
                        self.logger.warning(f"跳过非字典名句数据: {type(sentence_data)}")
                        continue
                    
                    # 提取基本信息并确保所有字段都有默认值
                    content = sentence_data.get("content", "")
                    if not isinstance(content, str):
                        content = str(content) if content else ""
                        
                    sentence = {
                        "content": content,
                        "source": sentence_data.get("source", ""),
                        "author": sentence_data.get("author", ""),
                        "category": "sentence",
                        "type": "other",
                        "data_source": "gushiwen"
                    }
                    
                    # 确保内容不为空
                    if not sentence["content"]:
                        continue
                    
                    # 清洗文本
                    if self.text_cleaner:
                        sentence["content"] = self.text_cleaner.clean_text(sentence["content"])
                    
                    results["other"].append(sentence)
            except Exception as e:
                self.logger.error(f"处理名句文件出错 {json_file}: {e}", exc_info=True)
    
    def _process_gushiwen_writers(self, directory: Path, results: Dict[str, List[Dict[str, Any]]]):
        """
        处理作者数据
        
        Args:
            directory: 作者数据目录
            results: 结果字典
        """
        # 遍历目录中的JSON文件
        for json_file in directory.glob("**/*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        # 验证数据是否为列表
                        if not isinstance(data, list):
                            self.logger.warning(f"作者文件格式错误，应为列表: {json_file}")
                            data = [data] if isinstance(data, dict) else []
                    except json.JSONDecodeError as e:
                        self.logger.error(f"解析作者文件失败 {json_file}: {e}")
                        # 尝试逐行解析
                        data = []
                        f.seek(0)  # 重置文件指针到开头
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                item = json.loads(line)
                                data.append(item)
                            except json.JSONDecodeError:
                                self.logger.warning(f"解析作者文件第{line_num}行失败: {json_file}")
                
                for writer_data in data:
                    if not isinstance(writer_data, dict):
                        self.logger.warning(f"跳过非字典作者数据: {type(writer_data)}")
                        continue
                    
                    # 提取基本信息并确保所有字段都有默认值
                    writer = {
                        "name": writer_data.get("name", ""),
                        "dynasty": writer_data.get("dynasty", ""),
                        "description": writer_data.get("intro", ""),
                        "source": "gushiwen"
                    }
                    
                    # 确保作者名不为空
                    if not writer["name"]:
                        continue
                    
                    results["authors"].append(writer)
            except Exception as e:
                self.logger.error(f"处理作者文件出错 {json_file}: {e}", exc_info=True)
    
    def _process_daizhige(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        处理殆知阁古代文献数据集
        
        Returns:
            处理后的数据字典
        """
        self.logger.info("开始处理殆知阁古代文献数据集")
        
        results = {
            "poems": [],
            "prose": [],
            "chapters": [],
            "authors": [],
            "other": []
        }
        
        daizhige_dir = self.input_dir / "daizhigev20-master"
        if not daizhige_dir.exists():
            self.logger.warning(f"殆知阁项目目录不存在: {daizhige_dir}")
            return results
        
        # 处理各类古籍
        categories = ["经藏", "史藏", "子藏", "集藏", "诗藏", "医藏", "艺藏", "佛藏", "道藏"]
        
        for category in categories:
            category_dir = daizhige_dir / category
            if not category_dir.exists():
                continue
            
            self._process_daizhige_books(category_dir, category, results)
        
        self.logger.info(f"殆知阁数据处理完成，共获取 {len(results['chapters'])} 篇章节")
        return results
    
    def _process_daizhige_books(self, directory: Path, category: str, results: Dict[str, List[Dict[str, Any]]]):
        """
        处理殆知阁古籍数据
        
        Args:
            directory: 古籍数据目录
            category: 分类名称
            results: 结果字典
        """
        # 遍历目录中的文本文件
        for txt_file in directory.glob("**/*.txt"):
            try:
                # 跳过目录文件
                if "目录" in txt_file.parts:
                    continue
                    
                with open(txt_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # 提取基本信息
                book_name = txt_file.stem
                
                # 判断是否为诗词
                if category == "诗藏" or self._is_poem(content):
                    # 处理为诗词
                    poem = {
                        "title": book_name,
                        "author": "",  # 可能需要从文件内容提取
                        "dynasty": "",  # 可能需要从文件内容提取
                        "content": content,
                        "source": "daizhige",
                        "type": "poem"
                    }
                    
                    # 清洗文本
                    if self.text_cleaner:
                        poem["content"] = self.text_cleaner.clean_text(poem["content"])
                    
                    results["poems"].append(poem)
                else:
                    # 分章节处理
                    chapters = self._split_into_chapters(content)
                    
                    for i, chapter_content in enumerate(chapters):
                        chapter = {
                            "title": f"{book_name}_{i+1}",
                            "book": book_name,
                            "category": category,
                            "content": chapter_content,
                            "source": "daizhige",
                            "type": "chapter"
                        }
                        
                        # 清洗文本
                        if self.text_cleaner:
                            chapter["content"] = self.text_cleaner.clean_text(chapter["content"])
                        
                        results["chapters"].append(chapter)
            except Exception as e:
                self.logger.error(f"处理殆知阁古籍文件出错 {txt_file}: {e}")
    
    def save_to_jsonl(self, items: List[Dict[str, Any]], output_path: Path) -> None:
        """
        将数据保存为JSONL格式
        
        Args:
            items: 数据项列表
            output_path: 输出文件路径
        """
        try:
            self.logger.info(f"开始保存 {len(items)} 条数据到 {output_path}")
            
            # 创建父目录
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                for item_index, item in enumerate(items):
                    try:
                        # 确保item是字典类型
                        if not isinstance(item, dict):
                            self.logger.warning(f"跳过非字典数据: {type(item)}")
                            continue
                        
                        # 规范化所有字段的值
                        normalized_item = {}
                        for key, value in item.items():
                            # 跳过None值
                            if value is None:
                                normalized_item[key] = ""
                                continue
                            
                            # 处理不同类型的值
                            if isinstance(value, (int, float, bool)):
                                normalized_item[key] = value
                            elif isinstance(value, (list, dict)):
                                # 如果是复杂对象，转换为JSON字符串
                                try:
                                    # 先尝试直接序列化
                                    json.dumps(value)
                                    normalized_item[key] = value
                                except (TypeError, ValueError):
                                    # 如果失败，转换为字符串
                                    normalized_item[key] = str(value)
                            else:
                                # 确保字符串类型
                                normalized_item[key] = str(value)
                        
                        # 写入JSONL
                        f.write(json.dumps(normalized_item, ensure_ascii=False) + "\n")
                    except Exception as e:
                        self.logger.error(f"保存第 {item_index} 条数据时出错: {e}")
            
            self.logger.info(f"已保存 {len(items)} 条数据到 {output_path}")
        except Exception as e:
            self.logger.error(f"保存数据到 {output_path} 失败: {e}", exc_info=True)
    
    def save_results(self, results: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        保存处理结果
        
        Args:
            results: 处理结果字典
        """
        self.logger.info("开始保存处理结果")
        
        # 获取输出目录
        output_dir = self.output_dir
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 为每种类型保存数据
        for category, items in results.items():
            if not items:
                continue
                
            # 输出路径
            output_path = output_dir / f"{category}.jsonl"
            
            # 保存数据
            self.save_to_jsonl(items, output_path)
        
        self.logger.info("处理结果保存完成")
    
    def _is_poem(self, content: str) -> bool:
        """
        判断内容是否为诗词
        
        Args:
            content: 内容文本
            
        Returns:
            是否为诗词
        """
        # 简单实现：根据行数和每行字数判断
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        
        if not lines:
            return False
        
        # 检查是否有韵律特征
        line_lengths = [len(line) for line in lines]
        
        # 诗词特征：行数较少，每行字数相近
        if len(lines) >= 2 and len(lines) <= 20:
            # 计算行长度的标准差，如果标准差小说明每行长度接近
            avg_len = sum(line_lengths) / len(line_lengths)
            variance = sum((l - avg_len) ** 2 for l in line_lengths) / len(line_lengths)
            
            # 标准差小于2，且平均行长小于20，可能是诗词
            if (variance < 4 and avg_len < 20) or "，" in content and "。" in content:
                return True
        
        return False