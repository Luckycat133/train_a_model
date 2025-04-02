#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
灵猫墨韵古典文学数据处理系统 - 数据处理器
"""

import os
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any, Tuple, Optional, Union
import signal
import sys
import time
import tempfile


class DataProcessor:
    """古典文学数据处理器"""
    
    def __init__(self, config, text_cleaner=None, structure_organizer=None):
        """
        初始化数据处理器
        
        Args:
            config: 配置字典
            text_cleaner: 文本清洗器
            structure_organizer: 结构组织器
        """
        self.logger = logging.getLogger("LingmaoMoyun.DataProcessor")
        self.config = config
        self.text_cleaner = text_cleaner
        self.structure_organizer = structure_organizer
        
        # 设置路径
        self.input_dir = Path(config["paths"]["input_dir"])
        self.output_dir = Path(config["paths"]["output_dir"])
        self.temp_dir = Path(config["paths"]["temp_dir"])
        
        # 确保目录存在
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        
        # 设置批处理大小
        self.batch_size = config["processor"]["batch_size"]
        
        self.logger.info(f"数据处理器初始化完成，输入目录: {self.input_dir}")
    
    def process_file(self, file_path: Path) -> List[Dict]:
        """
        处理单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            处理后的数据列表
        """
        self.logger.info(f"处理文件: {file_path}")
        
        try:
            # 根据文件扩展名选择处理方法
            if file_path.suffix.lower() == ".json":
                data = self._process_json_file(file_path)
            elif file_path.suffix.lower() == ".txt":
                data = self._process_txt_file(file_path)
            elif file_path.suffix.lower() == ".jsonl":
                data = self._process_jsonl_file(file_path)
            else:
                self.logger.warning(f"不支持的文件类型: {file_path.suffix}")
                return []
            
            # 清洗数据
            if self.text_cleaner:
                cleaned_data = self.text_cleaner.clean_batch(data)
            else:
                cleaned_data = data
            
            return cleaned_data
        except Exception as e:
            self.logger.error(f"处理文件 {file_path} 时出错: {e}", exc_info=True)
            return []
    
    def _process_json_file(self, file_path: Path) -> List[Dict]:
        """处理JSON文件"""
        # 尝试不同的编码方式
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]
        
        for encoding in encodings:
            try:
                # 尝试作为标准JSON加载
                with open(file_path, "r", encoding=encoding) as f:
                    data = json.load(f)
                
                # 如果是列表，直接返回
                if isinstance(data, list):
                    self.logger.info(f"成功以{encoding}编码解析JSON文件: {file_path}")
                    return data
                # 如果是字典，包装成列表
                elif isinstance(data, dict):
                    self.logger.info(f"成功以{encoding}编码解析JSON文件(字典): {file_path}")
                    return [data]
                
                # 成功解析但格式不正确
                self.logger.warning(f"JSON文件格式不是列表或字典: {file_path}")
                return []
                
            except json.JSONDecodeError as e:
                # 如果普通JSON解析失败，尝试更健壮的方法
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()
                    
                    # 预处理内容，修复常见的JSON错误
                    content = content.replace("\\'", "'").replace('\\"', '"')
                    content = content.replace("\n", " ").replace("\r", " ")
                    
                    # 尝试解析修复后的内容
                    data = json.loads(content)
                    
                    if isinstance(data, list):
                        self.logger.info(f"通过内容修复解析JSON文件: {file_path}")
                        return data
                    elif isinstance(data, dict):
                        self.logger.info(f"通过内容修复解析JSON文件(字典): {file_path}")
                        return [data]
                    
                    self.logger.warning(f"修复后的JSON文件格式不是列表或字典: {file_path}")
                    return []
                    
                except json.JSONDecodeError:
                    # 如果修复后还不行，尝试作为JSONL处理
                    try:
                        data = []
                        with open(file_path, "r", encoding=encoding) as f:
                            for line_num, line in enumerate(f, 1):
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    item = json.loads(line)
                                    data.append(item)
                                except json.JSONDecodeError:
                                    # 跳过解析失败的行
                                    self.logger.warning(f"无法解析第{line_num}行: {line[:50]}...")
                                    continue
                        
                        if data:  # 如果成功解析了一些行
                            self.logger.info(f"以JSONL格式解析文件: {file_path}")
                            return data
                    except Exception as inner_e:
                        # 捕获其他可能的错误，继续尝试下一种编码
                        self.logger.debug(f"以{encoding}编码解析JSONL失败: {inner_e}")
                except Exception as inner_e:
                    # 捕获其他可能的错误，继续尝试下一种编码
                    self.logger.debug(f"以{encoding}编码尝试修复JSON失败: {inner_e}")
            except Exception as e:
                # 捕获其他可能的错误，继续尝试下一种编码
                self.logger.debug(f"以{encoding}编码打开文件失败: {e}")
        
        # 如果所有尝试都失败
        self.logger.warning(f"无法解析JSON文件(已尝试所有编码): {file_path}")
        
        # 最后尝试：假设是纯文本文件，按段落分割
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            # 按段落分割
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            if paragraphs:
                self.logger.info(f"以纯文本方式处理文件: {file_path}")
                return [{"text": p, "source": file_path.name} for p in paragraphs]
        except Exception as e:
            self.logger.error(f"无法以任何方式处理文件: {file_path}, {e}")
        
        return []
    
    def _process_jsonl_file(self, file_path: Path) -> List[Dict]:
        """处理JSONL文件"""
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"解析JSONL行失败: {e}")
        return data
    
    def _process_txt_file(self, file_path: Path) -> List[Dict]:
        """处理文本文件"""
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # 尝试解析为JSON Lines
        if self._is_jsonl(lines):
            return self._parse_jsonl(lines, file_path)
        
        # 否则作为普通文本处理
        return self._parse_plain_text(lines, file_path)
    
    def _is_jsonl(self, lines: List[str]) -> bool:
        """判断是否为JSON Lines格式"""
        # 抽样检查前10行
        sample_size = min(10, len(lines))
        json_lines = 0
        
        for i in range(sample_size):
            if i >= len(lines):
                break
                
            line = lines[i].strip()
            if not line:
                continue
            
            try:
                json.loads(line)
                json_lines += 1
            except:
                pass
        
        # 如果超过50%的行是JSON，则认为是JSONL格式
        return json_lines > sample_size / 2
    
    def _parse_jsonl(self, lines: List[str], file_path: Path) -> List[Dict]:
        """解析JSON Lines"""
        data = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
                # 添加源文件信息
                item["_source"] = str(file_path)
                item["_line"] = i + 1
                data.append(item)
            except json.JSONDecodeError:
                self.logger.warning(f"解析JSON行失败: {file_path}:{i+1}")
        
        return data
    
    def _parse_plain_text(self, lines: List[str], file_path: Path) -> List[Dict]:
        """解析普通文本"""
        # 简单处理：每个文件作为一个条目
        content = "".join(lines)
        
        return [{
            "title": file_path.stem,
            "content": content,
            "_source": str(file_path)
        }]
    
    def process_directory(self, directory: Optional[Path] = None) -> List[Dict]:
        """
        处理目录中的所有文件
        
        Args:
            directory: 要处理的目录，默认为配置中的输入目录
            
        Returns:
            处理后的数据列表
        """
        directory = directory or self.input_dir
        self.logger.info(f"处理目录: {directory}")
        
        all_data = []
        supported_extensions = self.config["processor"]["file_extensions"]
        
        # 获取所有支持的文件
        files = []
        for ext in supported_extensions:
            files.extend(list(directory.glob(f"**/*{ext}")))
        
        self.logger.info(f"找到 {len(files)} 个文件")
        
        # 使用进程池并行处理文件
        max_workers = self.config["processor"]["max_workers"]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.process_file, files))
        
        # 合并结果
        for result in results:
            all_data.extend(result)
        
        self.logger.info(f"共处理 {len(all_data)} 条数据")
        return all_data
    
    def save_to_jsonl(self, data: List[Dict], output_path: Path) -> None:
        """
        保存数据到JSONL文件
        
        Args:
            data: 要保存的数据
            output_path: 输出文件路径
        """
        self.logger.info(f"保存数据到 {output_path}")
        
        # 确保父目录存在
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # 验证数据类型
        if not isinstance(data, list):
            self.logger.error(f"数据类型错误，应为列表: {type(data)}")
            return
        
        # 写入数据
        valid_items = 0
        invalid_items = 0
        
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                if not isinstance(item, dict):
                    self.logger.warning(f"跳过非字典数据项: {type(item)}")
                    invalid_items += 1
                    continue
                    
                try:
                    # 确保所有值都是可序列化的
                    serializable_item = {}
                    for key, value in item.items():
                        # 处理不可序列化的值
                        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                            serializable_item[key] = value
                        else:
                            # 尝试转换为字符串
                            try:
                                serializable_item[key] = str(value)
                            except:
                                serializable_item[key] = None
                                self.logger.debug(f"无法序列化字段 {key}，值类型: {type(value)}")
                    
                    # 写入JSON行
                    f.write(json.dumps(serializable_item, ensure_ascii=False) + "\n")
                    valid_items += 1
                except Exception as e:
                    self.logger.error(f"序列化数据项失败: {e}")
                    invalid_items += 1
        
        self.logger.info(f"已保存 {valid_items} 条有效数据，跳过 {invalid_items} 条无效数据")
        
        self.logger.info(f"已保存 {len(data)} 条数据")
    
    def match_metadata(self, data: List[Dict]) -> List[Dict]:
        """
        匹配元数据
        
        Args:
            data: 原始数据
            
        Returns:
            匹配元数据后的数据
        """
        self.logger.info("开始匹配元数据")
        
        # 构建索引
        title_index = {}
        author_index = {}
        
        # 第一遍：建立索引
        for item in data:
            # 索引标题
            if "title" in item and item["title"]:
                title = item["title"]
                if title not in title_index:
                    title_index[title] = []
                title_index[title].append(item)
            
            # 索引作者
            if "author" in item and item["author"]:
                author = item["author"]
                if author not in author_index:
                    author_index[author] = []
                author_index[author].append(item)
        
        # 第二遍：匹配元数据
        for item in data:
            # 如果缺少朝代信息，尝试从同作者的作品中获取
            if ("dynasty" not in item or not item["dynasty"]) and "author" in item:
                author = item["author"]
                if author in author_index:
                    for other_item in author_index[author]:
                        if other_item != item and "dynasty" in other_item and other_item["dynasty"]:
                            item["dynasty"] = other_item["dynasty"]
                            self.logger.debug(f"为 {item.get('title', '未知')} 匹配朝代: {item['dynasty']}")
                            break
        
        self.logger.info("元数据匹配完成")
        return data
    
    def process(self) -> List[Dict]:
        """
        执行完整的处理流程
        
        Returns:
            处理后的数据
        """
        self.logger.info("开始执行完整处理流程")
        
        # 处理目录
        raw_data = self.process_directory()
        
        # 匹配元数据
        enriched_data = self.match_metadata(raw_data)
        
        # 组织结构
        if self.structure_organizer:
            organized_data = self.structure_organizer.organize_data(enriched_data)
        else:
            organized_data = {"all": enriched_data}
        
        self.logger.info("处理流程完成")
        return organized_data


# 简单测试
if __name__ == "__main__":
    # 配置日志
    # 设置根日志记录器配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/data_processor_test.log", encoding="utf-8")
        ]
    )
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    
    # 获取根日志记录器并添加处理器
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    
    # 测试配置
    test_config = {
        "paths": {
            "input_dir": "test_data",
            "output_dir": "test_output",
            "temp_dir": "test_output/temp"
        },
        "processor": {
            "batch_size": 100,
            "max_workers": 2,
            "file_extensions": [".json", ".txt", ".jsonl"]
        }
    }
    
    # 创建处理器
    processor = DataProcessor(test_config)
    
    # 测试处理
    if os.path.exists("test_data"):
        result = processor.process()
        print(f"处理结果: {result}")
    else:
        print("测试数据目录不存在")


class GracefulInterruptHandler:
    """处理优雅中断的辅助类"""
    
    def __init__(self, sig=signal.SIGINT):
        self.sig = sig
        self.interrupted = False
        self.released = False
        self.original_handler = None
    
    def __enter__(self):
        self.original_handler = signal.getsignal(self.sig)
        signal.signal(self.sig, self.handler)
        return self
    
    def handler(self, sig, frame):
        self.interrupted = True
        logger.warning(f"接收到中断信号 {sig}，准备优雅退出...")
        # 允许第二次按Ctrl+C强制退出
        if self.interrupted:
            logger.critical("接收到第二次中断，强制退出...")
            signal.signal(self.sig, self.original_handler)
            raise KeyboardInterrupt
    
    def __exit__(self, type, value, traceback):
        signal.signal(self.sig, self.original_handler)
        self.released = True


def _train_tokenizer(self):
    """此方法已弃用，分词器训练已移至tokenizer.py"""
    logger.warning("分词器训练功能已移至tokenizer.py，请直接使用tokenizer.py进行分词器训练")
    try:
        # 历史代码留存，防止其他地方调用出错
        return False
    except KeyboardInterrupt:
        # 如果GracefulInterruptHandler没有捕获到中断（如连续快速按Ctrl+C）
        logger.error("分词器训练被强制中断")
        return False
    except Exception as e:
        logger.error(f"分词器训练出错: {e}")
        logger.error(traceback.format_exc())
        return False
