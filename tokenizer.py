#!/usr/bin/env python3
import os
import json
import yaml
import time
import threading
import queue
import functools
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
from tqdm import tqdm
from collections import defaultdict
from functools import lru_cache

# 配置日志记录
log_dir = Path("logs/tokenizer")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"tokenizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 设置根日志记录器配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file)
    ]
)

# 获取日志记录器并配置控制台处理器
logger = logging.getLogger("ClassicalTokenizer")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console_handler)

class ClassicalTokenizer:
    def __init__(self, vocab_size=None, special_tokens=None, dictionary_path=None, config_path="config/config.yaml"):
        """
        初始化分词器
        - vocab_size: 词汇表大小
        - special_tokens: 特殊符号列表
        - dictionary_path: 古文专用词典文件路径（包含词性标注）
        - config_path: 配置文件路径
        """
        # 从配置文件加载设置
        self.config = self.load_config(config_path)
        
        # 使用配置文件中的设置，如果未提供则使用默认值
        if vocab_size is None:
            vocab_size = self.config.get('tokenizer', {}).get('vocab_size', 30000)
        
        if special_tokens is None:
            special_tokens = self.config.get('tokenizer', {}).get('special_tokens', 
                                                              ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        
        if dictionary_path is None:
            dictionary_path = self.config.get('tokenizer', {}).get('dictionary_path', 
                                                              "dataset/dictionaries/classical_terms.txt")
        
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        # 初始化基于 BPE 算法的 tokenizer
        self.tokenizer = Tokenizer(BPE())
        # 使用空格作为预分词方式
        self.tokenizer.pre_tokenizer = Whitespace()
        # 加载古文专用词典，后处理时用于最大匹配分词
        self.dictionary = self.load_dictionary(dictionary_path)
        self.max_dict_length = max([len(word) for word in self.dictionary]) if self.dictionary else 0
        
        # 性能优化：添加缓存
        self.token_cache = {}
        self.max_cache_size = 100000  # 最大缓存条目数
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 性能优化：词典查找优化 - 按长度分组
        self.dictionary_by_length = defaultdict(dict)
        for word, pos in self.dictionary.items():
            self.dictionary_by_length[len(word)][word] = pos
        
        # 性能优化：批处理配置
        self.default_batch_size = self.config.get('processor', {}).get('batch_size', 1000)
        self.max_workers = min(os.cpu_count(), self.config.get('processor', {}).get('max_workers', 8))
        
        # 性能优化：预加载常用文本类型的处理策略
        self.text_type_strategies = {
            "poem": self._process_poem,
            "prose": self._process_prose,
            "article": self._process_article,
            "chu_ci": self._process_chu_ci
        }
        
        logger.info(f"分词器初始化完成，词汇量大小: {vocab_size}, 最大工作线程: {self.max_workers}")
    
    def load_config(self, config_path):
        """
        从YAML配置文件加载配置
        """
        config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info(f"配置已从 {config_path} 加载")
            except Exception as e:
                logger.error(f"加载配置文件时出错: {e}")
        else:
            logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            
        # 如果配置中没有tokenizer部分，添加默认配置
        if 'tokenizer' not in config:
            config['tokenizer'] = {
                'vocab_size': 30000,
                'special_tokens': ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                'dictionary_path': "dataset/dictionaries/classical_terms.txt"
            }
        return config

    def load_dictionary(self, dictionary_path):
        """
        从文件中加载词典，包含词条和词性标注，返回词典字典
        格式：词条 词性
        性能优化：使用二进制缓存加速重复加载
        """
        dictionary = {}
        cache_path = Path(dictionary_path + ".cache")
        
        # 尝试从缓存加载
        if cache_path.exists() and cache_path.stat().st_mtime > Path(dictionary_path).stat().st_mtime:
            try:
                with open(cache_path, 'rb') as f:
                    dictionary = pickle.load(f)
                logger.info(f"从缓存加载词典: {len(dictionary)} 个词条")
                return dictionary
            except Exception as e:
                logger.warning(f"从缓存加载词典失败: {e}")
        
        # 从原始文件加载
        if os.path.exists(dictionary_path):
            start_time = time.time()
            with open(dictionary_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            term, pos = line.split()
                            dictionary[term] = pos
                        except ValueError:
                            continue
            
            # 保存到缓存
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(dictionary, f)
            except Exception as e:
                logger.warning(f"保存词典缓存失败: {e}")
                
            load_time = time.time() - start_time
            logger.info(f"从文件加载词典: {len(dictionary)} 个词条, 耗时 {load_time:.2f} 秒")
        else:
            logger.warning(f"词典文件 {dictionary_path} 不存在")
        
        return dictionary

    def get_training_files(self, dataset_dir="dataset", file_ext=".jsonl"):
        """
        自动获取dataset目录中的所有训练文件
        - dataset_dir: 数据集目录
        - file_ext: 文件扩展名，默认为.jsonl
        返回训练文件路径列表
        """
        training_files = []
        # 使用Path模块查找所有匹配的文件
        dataset_path = Path(dataset_dir)
        pattern = f"*{file_ext}"
        training_files = [str(file) for file in dataset_path.glob(pattern)]
        
        if not training_files:
            logger.warning(f"在 {dataset_dir} 中未找到扩展名为 {file_ext} 的训练文件")
        else:
            logger.info(f"在 {dataset_dir} 中找到 {len(training_files)} 个训练文件")
            
        return training_files

    def _process_poem(self, text):
        """处理古诗文本"""
        # 简单实现，后续可扩展更复杂的处理
        lines = text.split('\n')
        processed_lines = []
        for line in lines:
            # 移除行首空格和标点
            line = line.strip().strip(',.?!;:，。？！；：')
            if line:
                processed_lines.append(line)
        return '\n'.join(processed_lines)
    
    def _process_prose(self, text):
        """处理古文散文文本"""
        # 简单实现，后续可扩展更复杂的处理
        # 移除多余空白
        text = ' '.join(text.split())
        # 处理常见问题
        text = text.replace('—', '').replace('…', '')
        return text
    
    def _process_article(self, text):
        """处理文章文本"""
        # 针对现代文章的处理
        paragraphs = text.split('\n\n')
        processed_paragraphs = []
        for para in paragraphs:
            # 移除每段首尾空白
            para = para.strip()
            if para:
                processed_paragraphs.append(para)
        return '\n\n'.join(processed_paragraphs)
    
    def _process_chu_ci(self, text):
        """处理楚辞风格文本"""
        # 处理楚辞特殊格式，如"兮"字的处理
        lines = text.split('\n')
        processed_lines = []
        for line in lines:
            # 特殊处理楚辞格式
            line = line.strip()
            if '兮' in line:
                # 在"兮"字后添加适当的断句
                parts = line.split('兮')
                line = '兮\n'.join(parts)
                # 添加楚辞特有的韵律处理
                line = line.replace('兮\n', '兮\n\n')  # 在兮字后添加空行增强韵律感
                # 移除多余的标点
                line = line.replace('，', '').replace('。', '')
            if line:
                processed_lines.append(line)
        return '\n'.join(processed_lines)

    def extract_text_from_jsonl(self, jsonl_files):
        """
        从JSONL文件中提取文本内容用于训练
        - jsonl_files: JSONL文件路径列表
        返回提取的文本行列表
        
        性能优化：使用多线程处理，提高大文件处理速度
        """
        text_lines = []
        text_queue = queue.Queue(maxsize=1000)  # 限制队列大小，避免内存溢出
        total_lines = 0
        error_count = 0
        
        def process_file(file_path):
            """
            线程处理单个文件的函数
            性能优化：使用批处理减少队列操作次数
            """
            local_lines = 0
            local_errors = 0
            local_texts = []
            batch_size = 1000  # 本地批处理大小
            
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            local_lines += 1
                            # 提取文本内容，根据数据结构调整字段名
                            if isinstance(data, dict):
                                if "content" in data and data["content"]:
                                    if isinstance(data["content"], str):
                                        local_texts.append(data["content"])
                                    else:
                                        # 尝试转换非字符串内容
                                        try:
                                            local_texts.append(str(data["content"]))
                                        except:
                                            pass
                                elif "text" in data and data["text"]:
                                    if isinstance(data["text"], str):
                                        local_texts.append(data["text"])
                                    else:
                                        try:
                                            local_texts.append(str(data["text"]))
                                        except:
                                            pass
                                elif "paragraphs" in data and isinstance(data["paragraphs"], list):
                                    for para in data["paragraphs"]:
                                        if isinstance(para, str) and para:
                                            local_texts.append(para)
                            
                            # 批处理：当积累足够的文本时，放入队列
                            if len(local_texts) >= batch_size:
                                text_queue.put((local_lines, local_errors, local_texts))
                                local_texts = []
                                
                        except json.JSONDecodeError:
                            local_errors += 1
                            if local_errors < 10:  # 只打印前10个错误避免刷屏
                                logger.warning(f"警告: {file_path} 的第 {line_num} 行JSON解析失败")
            
            # 将剩余结果放入队列
            if local_texts:
                text_queue.put((local_lines, local_errors, local_texts))
        
        # 创建线程池处理文件
        logger.info(f"使用多线程处理 {len(jsonl_files)} 个JSONL文件...")
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(jsonl_files))) as executor:
            # 提交所有文件处理任务
            futures = [executor.submit(process_file, file) for file in jsonl_files]
            
            # 使用单独的线程收集结果，避免阻塞
            def collector():
                nonlocal total_lines, error_count
                completed = 0
                completed_futures = set()
                with tqdm(total=len(jsonl_files), desc="处理JSONL文件") as pbar:
                    while completed < len(jsonl_files):
                        try:
                            lines, errors, texts = text_queue.get(timeout=0.1)
                            total_lines += lines
                            error_count += errors
                            text_lines.extend(texts)
                            
                            # 检查是否有任务完成
                            for future in futures:
                                if future.done():
                                    if future not in completed_futures:
                                        completed += 1
                                        completed_futures.add(future)
                                        pbar.update(1)
                        except queue.Empty:
                            # 检查是否所有任务都已完成
                            all_done = all(future.done() for future in futures)
                            if all_done and text_queue.empty():
                                break
            
            collector_thread = threading.Thread(target=collector)
            collector_thread.start()
            
            # 等待所有任务完成
            for future in futures:
                future.result()  # 确保任何异常被抛出
            
            # 等待收集器完成
            collector_thread.join()
        
        if error_count > 0:
            logger.warning(f"警告: 解析过程中遇到 {error_count} 个JSON错误")
        
        if len(text_lines) < 100:
            logger.warning(f"警告: 从JSONL文件中只提取到 {len(text_lines)} 行文本，训练数据可能不足")
        
        return text_lines

    def train(self, training_files=None):
        """
        对给定的训练文件进行 BPE 分词器训练，并显示进度条
        如果未提供training_files，则自动从dataset目录获取
        使用多线程处理提高性能
        """
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)

        # 如果未提供训练文件，则自动获取
        if training_files is None or len(training_files) == 0:
            training_files = self.get_training_files()
            
            if not training_files:
                logger.warning("未找到训练文件，无法训练分词器")
                return False
            
            # 如果是JSONL文件，需要提取文本内容
            if any(file.endswith(".jsonl") for file in training_files):
                corpus_lines = self.extract_text_from_jsonl(training_files)
            else:
                # 使用并行处理读取所有训练文件内容
                corpus_lines = []
                file_queue = queue.Queue()
                
                def process_text_file(file_path):
                    """处理普通文本文件的线程函数"""
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                lines = f.readlines()
                                file_queue.put((file_path, lines, None))
                        except Exception as e:
                            file_queue.put((file_path, [], str(e)))
                    else:
                        file_queue.put((file_path, [], "文件不存在"))
                
                # 使用线程池并行处理文件
                logger.info(f"使用多线程处理 {len(training_files)} 个文本文件...")
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(training_files))) as executor:
                    futures = [executor.submit(process_text_file, file) for file in training_files]
                    
                    # 等待所有线程完成并收集结果
                    for _ in tqdm(futures, desc="读取文本文件"):
                        file_path, lines, error = file_queue.get()
                        if error:
                            logger.warning(f"读取训练文件 {file_path} 时出错: {error}")
                        else:
                            corpus_lines.extend(lines)


        if not corpus_lines:
            logger.warning("未找到有效的训练数据，分词器训练中止")
            return False
            
        logger.info(f"共收集到 {len(corpus_lines)} 行训练数据")

        # 过滤空行和无效行 - 使用多线程处理大量数据
        def filter_lines(batch):
            """过滤一批行的线程函数"""
            return [line for line in batch if line and len(line.strip()) > 0]
        
        # 分批处理以提高效率
        batch_size = max(1000, len(corpus_lines) // (self.max_workers * 4))
        batches = [corpus_lines[i:i+batch_size] for i in range(0, len(corpus_lines), batch_size)]
        
        filtered_lines = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(filter_lines, batch) for batch in batches]
            
            # 收集过滤后的结果
            for future in tqdm(futures, desc="过滤训练数据"):
                filtered_batch = future.result()
                filtered_lines.extend(filtered_batch)
        
        corpus_lines = filtered_lines
        logger.info(f"过滤后剩余 {len(corpus_lines)} 行有效训练数据")
        
        if len(corpus_lines) < 100:
            logger.warning("警告: 训练数据量过少，可能影响分词器质量")

        # 将语料写入临时文件，同时显示处理进度
        temp_corpus = "temp_corpus.txt"
        # --- 清理旧的临时语料文件 --- #
        if os.path.exists(temp_corpus):
            try:
                os.remove(temp_corpus)
                logger.info(f"已删除旧的临时语料文件: {temp_corpus}")
            except OSError as e:
                logger.warning(f"无法删除旧的临时语料文件 {temp_corpus}: {e}")
        # --- 文件写入和训练 --- #
        try:
            with open(temp_corpus, "w", encoding="utf-8") as f:
                for line in tqdm(corpus_lines, desc="准备语料库"):
                    if isinstance(line, str):
                        f.write(line + "\n")
            
            # 使用临时文件进行 BPE 模型训练
            start_time = time.time()
            try:
                logger.info("开始训练分词器模型...")
                self.tokenizer.train([temp_corpus], trainer)
                end_time = time.time()
                training_time = end_time - start_time
                logger.info(f"分词器训练完成，耗时 {training_time:.2f} 秒")
                return True
            except Exception as e:
                logger.error(f"分词器训练失败: {e}")
                return False
        except Exception as e:
            logger.error(f"准备训练语料时出错: {e}")
            return False
        finally:
            # 清理临时文件
            if os.path.exists(temp_corpus):
                try:
                    os.remove(temp_corpus)
                except:
                    pass

    def bpe_tokenize(self, text):
        """
        使用训练好的 BPE 分词器对文本进行分词
        """
        encoding = self.tokenizer.encode(text)
        return encoding.tokens

    def max_match_tokenize(self, text, return_pos=False):
        """
        采用从左到右的最大匹配算法进行古文分词
        如果字典中存在能匹配的长词，则选择最长匹配；否则单个字符作为基本单位分割
        参数：
        - text: 待分词文本
        - return_pos: 是否返回词性标注
        返回：
        - 如果return_pos为True，返回(tokens, pos_tags)元组
        - 否则仅返回tokens列表
        
        性能优化：
        1. 使用多级缓存减少重复计算
        2. 按长度分组的词典查找
        3. 批处理优化
        4. 内存使用优化
        """
        # 性能优化：空文本快速返回
        if not text or len(text.strip()) == 0:
            return ([], []) if return_pos else []
            
        # 性能优化：缓存键生成（只使用文本的前100个字符作为缓存键的一部分）
        cache_key = f"{text[:100]}_{return_pos}"
        
        # 检查LRU缓存
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        tokens = []
        pos_tags = []
        i = 0
        text_length = len(text)
        
        # 性能优化：预分配内存
        tokens = [None] * text_length  # 最坏情况下的长度
        pos_tags = [None] * text_length if return_pos else []
        actual_length = 0
        
        while i < text_length:
            # 性能优化：快速路径检查
            if i + 1 >= text_length:
                tokens[actual_length] = text[i]
                if return_pos:
                    pos_tags[actual_length] = 'n'
                actual_length += 1
                break
            
            # 初始化未匹配
            match = None
            pos = None
            # 限定最大匹配长度
            max_len = min(self.max_dict_length, text_length - i)
            
            # 性能优化：从最可能的长度开始匹配
            for l in range(max_len, 1, -1):
                if l in self.dictionary_by_length:
                    candidate = text[i:i+l]
                    if candidate in self.dictionary_by_length[l]:
                        match = candidate
                        pos = self.dictionary_by_length[l][match]
                        break
            
            if match:
                tokens[actual_length] = match
                if return_pos:
                    pos_tags[actual_length] = pos
                actual_length += 1
                i += len(match)
            else:
                tokens[actual_length] = text[i]
                if return_pos:
                    pos_tags[actual_length] = 'n'
                actual_length += 1
                i += 1
        
        # 裁剪到实际长度
        tokens = tokens[:actual_length]
        if return_pos:
            pos_tags = pos_tags[:actual_length]
        
        result = (tokens, pos_tags) if return_pos else tokens
        
        # 更新缓存
        self._update_cache(cache_key, result)
        
        return result

    # 删除第一个tokenize方法定义，保留第二个定义

    def _get_from_cache(self, key):
        """从缓存中获取结果，包含访问统计"""
        if key in self.token_cache:
            self.cache_hits += 1
            return self.token_cache[key]
        self.cache_misses += 1
        return None

    def _update_cache(self, key, value):
        """更新缓存，包含智能清理策略"""
        if len(self.token_cache) >= self.max_cache_size:
            # 性能优化：基于命中率的缓存清理
            if self.cache_hits / (self.cache_hits + self.cache_misses) < 0.5:
                # 如果命中率低，清理更多缓存
                self.token_cache.clear()
            else:
                # 保留最近使用的75%
                items = list(self.token_cache.items())
                self.token_cache = dict(items[len(items)//4:])
        
        self.token_cache[key] = value

    # 添加LRU缓存包装器方法
    @lru_cache(maxsize=32768)
    def _tokenize_cached(self, text):
        """使用LRU缓存包装的分词方法，避免重复计算"""
        if self.tokenizer is None:
            logger.warning("分词器未初始化，无法进行分词")
            return []
        return self.tokenizer.encode(text).ids

    def tokenize(self, text, method="auto", text_type=None, use_cache=True):
        """
        根据指定方法进行分词：
          - "bpe": 使用训练好的 BPE 分词器
          - "max_match": 使用最大匹配字典分词
          - "auto": 当字典存在时优先采用最大匹配分词，否则使用 BPE 分词器
        
        text_type参数可以指定文本类型，用于选择最合适的分词策略：
          - "poem": 诗词类文本
          - "prose": 散文类文本
          - "article": 文章类文本
          - "chu_ci": 楚辞类文本
          - None: 自动判断
          
        use_cache: 是否使用缓存提高性能
        """
        # 性能优化：空文本快速返回
        if not text or len(text.strip()) == 0:
            return []
        
        # 自动判断文本类型（如果未指定）
        if text_type is None:
            if "兮" in text and ("兮" in text.split("，") or "兮" in text.split("。")):
                text_type = "chu_ci"  # 包含"兮"字且位于句中可能是楚辞
            elif len(text) <= 100 and ("，" in text or "。" in text) and all(len(line) <= 15 for line in text.split("\n")):
                text_type = "poem"  # 短文本且有断句可能是诗词
            elif len(text) > 100:
                text_type = "article"  # 长文本可能是文章
            else:
                text_type = "prose"  # 默认为散文
        
        # 保存原始文本前缀用于缓存键
        original_text_prefix = text[:100]
        
        # 性能优化：预处理文本
        if text_type and text_type in self.text_type_strategies:
            logger.debug(f"Applying preprocessing strategy for type: {text_type}")
            text = self.text_type_strategies[text_type](text) # 对 text 进行预处理
        
        # 性能优化：缓存键生成 - 移到预处理后，使用原始文本前缀
        cache_key = f"{original_text_prefix}_{method}_{text_type}"
        
        # 检查缓存
        if use_cache and cache_key in self.token_cache:
            self.cache_hits += 1
            return self.token_cache[cache_key]
        
        self.cache_misses += 1
        
        # 性能优化：根据文本长度选择处理策略
        if len(text) < 100:  # 短文本直接处理
            tokens = self._process_short_text(text, method)
        else:  # 长文本分段处理
            tokens = self._process_long_text(text, method)
        
        # 更新缓存
        if use_cache:
             self._update_cache(cache_key, tokens)

        return tokens

    def _process_short_text(self, text, method):
        """处理短文本的优化方法"""
        if method == "bpe" or (method == "auto" and not self.dictionary):
            return self._tokenize_cached(text)
        elif method == "max_match" or (method == "auto" and self.dictionary):
            return self.max_match_tokenize(text)
        return []

    def _process_long_text(self, text, method, chunk_size=1000):
        """处理长文本的优化方法，使用分块并行处理"""
        # 分块处理
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            if method == "bpe" or (method == "auto" and not self.dictionary):
                futures = [executor.submit(self._tokenize_cached, chunk) for chunk in chunks]
            else:
                futures = [executor.submit(self.max_match_tokenize, chunk) for chunk in chunks]
            
            # 收集结果
            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.extend(result)
                except Exception as e:
                    logger.warning(f"分词处理出错: {e}")
            
            return results