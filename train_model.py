import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import signal
import time
import json
import logging
import glob
import math
from tqdm import tqdm
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from pathlib import Path
import shutil
from termcolor import colored
import psutil

# 设置版本号
VERSION = "0.8.5"

# ================= 工具函数 =================

def get_gpu_memory_info():
    """获取GPU内存使用信息"""
    if torch.cuda.is_available():
        gpu_memory = {
            'total': torch.cuda.get_device_properties(0).total_memory,
            'reserved': torch.cuda.memory_reserved(0),
            'allocated': torch.cuda.memory_allocated(0)
        }
        return gpu_memory
    return None

def format_memory_size(size_bytes):
    """格式化内存大小显示"""
    if size_bytes is None:
        return "未知"
    
    size_kb = size_bytes / 1024
    if size_kb < 1024:
        return f"{size_kb:.2f} KB"
    
    size_mb = size_kb / 1024
    if size_mb < 1024:
        return f"{size_mb:.2f} MB"
    
    size_gb = size_mb / 1024
    return f"{size_gb:.2f} GB"

def format_time(seconds):
    """将秒数格式化为更易读的时间格式"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.1f}小时{minutes:.1f}分钟"

def plot_training_stats(stats, save_dir):
    """绘制训练统计图表"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 设置字体和颜色主题
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12
    
    # 创建一个2x2的子图布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#F5F5F5')
    
    # 主题颜色
    colors = {
        'loss': '#3498db',     # 蓝色
        'lr': '#2ecc71',       # 绿色
        'time': '#e74c3c',     # 红色
        'memory': '#9b59b6'    # 紫色
    }
    
    # 1. 损失曲线
    epochs = range(1, len(stats['losses']) + 1)
    ax1.plot(epochs, stats['losses'], 'o-', linewidth=2, markersize=8, 
             color=colors['loss'], label='训练损失')
    ax1.set_title('训练损失曲线', fontweight='bold', fontsize=14)
    ax1.set_xlabel('轮次', fontsize=12)
    ax1.set_ylabel('损失值', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 添加最小值标记
    min_loss = min(stats['losses'])
    min_idx = stats['losses'].index(min_loss)
    ax1.annotate(f'最小值: {min_loss:.4f}',
                xy=(min_idx + 1, min_loss), 
                xytext=(min_idx + 1 + 0.5, min_loss + 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=11)
    
    ax1.legend(loc='upper right')
    
    # 2. 学习率变化
    steps_per_epoch = len(stats['learning_rates']) // len(epochs)
    if steps_per_epoch > 1:
        # 如果每个epoch有多个学习率记录，创建详细的学习率曲线
        all_steps = range(1, len(stats['learning_rates']) + 1)
        ax2.plot(all_steps, stats['learning_rates'], '-', linewidth=2, 
                color=colors['lr'], label='学习率')
        
        # 标记预热阶段
        warmup_steps = int(0.1 * len(stats['learning_rates']))
        if warmup_steps > 0:
            ax2.axvspan(1, warmup_steps, alpha=0.2, color='yellow')
            ax2.text(warmup_steps/2, min(stats['learning_rates']), '预热阶段', 
                    ha='center', va='bottom', fontsize=10)
        
        ax2.set_xlabel('训练步数', fontsize=12)
    else:
        # 按轮次显示学习率
        ax2.plot(epochs, stats['learning_rates'], 'o-', linewidth=2, markersize=8, 
                color=colors['lr'], label='学习率')
        ax2.set_xlabel('轮次', fontsize=12)
    
    ax2.set_title('学习率变化曲线', fontweight='bold', fontsize=14)
    ax2.set_ylabel('学习率', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.6f'))
    ax2.legend(loc='upper right')
    
    # 3. 每轮训练时间
    ax3.plot(epochs, stats['epoch_times'], 'o-', linewidth=2, markersize=8, 
             color=colors['time'], label='训练时间')
    
    # 计算平均训练时间
    avg_time = sum(stats['epoch_times']) / len(stats['epoch_times'])
    ax3.axhline(y=avg_time, linestyle='--', color='gray', alpha=0.8)
    ax3.annotate(f'平均: {avg_time:.2f}秒',
                xy=(len(epochs) / 2, avg_time),
                xytext=(len(epochs) / 2, avg_time * 1.1),
                fontsize=11)
    
    ax3.set_title('每轮训练时间', fontweight='bold', fontsize=14)
    ax3.set_xlabel('轮次', fontsize=12)
    ax3.set_ylabel('时间(秒)', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper right')
    
    # 4. GPU内存使用
    if 'gpu_memory_usage' in stats and stats['gpu_memory_usage']:
        memory_usage_gb = [m/1024**3 for m in stats['gpu_memory_usage']]
        ax4.plot(epochs, memory_usage_gb, 'o-', linewidth=2, markersize=8, 
                 color=colors['memory'], label='GPU内存')
        
        # 添加最大内存使用标记
        max_mem = max(memory_usage_gb)
        max_idx = memory_usage_gb.index(max_mem)
        ax4.annotate(f'最大值: {max_mem:.2f}GB',
                    xy=(max_idx + 1, max_mem), 
                    xytext=(max_idx + 1 - 0.5, max_mem * 1.1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=11)
        
        ax4.set_title('GPU内存使用', fontweight='bold', fontsize=14)
        ax4.set_xlabel('轮次', fontsize=12)
        ax4.set_ylabel('内存使用(GB)', fontsize=12)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(loc='upper right')
    else:
        # 如果没有GPU信息，则显示训练总结
        ax4.axis('off')
        ax4.text(0.5, 0.5, '训练总结:\n\n' + 
                f'总训练轮数: {len(epochs)}\n' +
                f'最终损失: {stats["losses"][-1]:.4f}\n' +
                f'最小损失: {min(stats["losses"]):.4f}\n' +
                f'总训练时间: {sum(stats["epoch_times"]):.2f}秒',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14,
                transform=ax4.transAxes)
    
    # 添加整体标题
    fig.suptitle('灵猫墨韵 - 训练统计图表', fontsize=18, fontweight='bold', y=0.98)
    
    # 添加训练时间戳和版本信息
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(0.5, 0.01, f'训练时间: {timestamp} | 版本: v{VERSION}',
             horizontalalignment='center', fontsize=10, alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图表为高质量PNG文件
    stats_plot_path = os.path.join(save_dir, f'training_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(stats_plot_path, dpi=300, bbox_inches='tight')
    
    # 同时保存为PDF以便进一步编辑或发布
    pdf_path = os.path.join(save_dir, f'training_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.close()
    
    log_success(f"训练统计图表已保存至: {stats_plot_path}")
    log_info(f"PDF版本已保存至: {pdf_path}")

# ================= 日志系统 =================

def setup_logger():
    """设置日志系统"""
    log_dir = os.path.join("logs", "train_model")
    os.makedirs(log_dir, exist_ok=True)
    
    # 先获取根日志记录器并移除所有处理器，防止重复日志
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    logger = logging.getLogger("LingmaoMoyun")
    logger.setLevel(logging.INFO)
    # 清除现有处理器，防止重复添加
    if logger.handlers:
        logger.handlers.clear()
    
    # 控制台处理器 - 使用简化格式，不显示日期时间
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 文件处理器 - 保留完整日志格式
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    
    # 设置简洁格式（控制台不显示时间）和完整格式（文件保留时间）
    console_formatter = logging.Formatter("%(message)s")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()

def log_info(message):
    """输出信息日志"""
    logger.info(message)

def log_warning(message):
    """输出警告日志"""
    logger.warning(message)

def log_error(message):
    """输出错误日志"""
    logger.error(message)

def log_success(message):
    """输出成功日志"""
    logger.info(message)

# 移除分隔线函数，简化为普通标题输出
def print_section_header(title):
    """打印段落标题，不带分隔符"""
    logger.info(f"\n{title}")

# 信号处理函数，用于随时终止训练并保存
terminate_training = False
def signal_handler(sig, frame):
    global terminate_training
    log_warning("接收到终止信号，正在保存当前状态...")
    terminate_training = True
    
    # 设置一个定时器，如果10秒内未能正常退出，则强制退出
    import threading
    def force_exit():
        log_error("保存超时，强制退出程序")
        import os
        os._exit(1)
    
    # 10秒后强制退出
    timer = threading.Timer(10.0, force_exit)
    timer.start()

signal.signal(signal.SIGINT, signal_handler)

# ================= 数据处理部分 =================

class LMDataset(Dataset):
    """语言模型数据集，用于自回归训练"""
    def __init__(self, data_path, context_length=512, tokenizer=None, stride=256, max_chunks=None):
        """
        初始化语言模型数据集
        
        Args:
            data_path: 数据路径，可以是文件或目录
            context_length: 上下文长度
            tokenizer: 分词器实例
            stride: 滑动窗口步长
            max_chunks: 最大文本块数量，用于限制内存使用
        """
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.stride = stride
        self.max_chunks = max_chunks
        
        # 性能优化：使用numpy数组而非Python列表
        import numpy as np
        self.np = np
        
        # 优化：使用内存映射文件存储大型数据
        self.use_memmap = False
        self.memmap_path = None
        
        # 加载所有文本
        self.token_chunks = []
        self.samples = []
        
        # 计时器
        self.load_time = 0
        self.process_time = 0
        
        # 记录内存使用
        self.peak_memory_mb = 0
        self._track_memory()
        
        # 加载数据
        start_time = time.time()
        self._load_data(data_path)
        self.load_time = time.time() - start_time
        
        # 创建滑动窗口样本
        start_time = time.time()
        self._create_samples()
        self.process_time = time.time() - start_time
        
        self._track_memory()
        logger.info(f"数据集 {data_path} 加载完成，共有 {len(self.samples)} 个样本，峰值内存: {self.peak_memory_mb:.2f}MB")
    
    def _track_memory(self):
        """跟踪内存使用"""
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        current_memory_mb = memory_info.rss / (1024 * 1024)
        self.peak_memory_mb = max(self.peak_memory_mb, current_memory_mb)
    
    def _load_data(self, data_path):
        """
        从文件或目录加载数据
        
        Args:
            data_path: 数据路径，可以是文件或目录
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"数据路径不存在: {data_path}")
        
        # 性能优化：使用迭代器而非列表
        if data_path.is_dir():
            files_to_process = data_path.glob("**/*.jsonl")
            logger.info(f"从目录 {data_path} 加载JSONL文件")
        else:
            # 如果是文件，直接添加
            files_to_process = [data_path]
            logger.info(f"加载单个文件: {data_path}")
        
        # 处理所有文件
        chunks_processed = 0
        for file_path in tqdm(files_to_process, 
                            desc="加载数据文件", 
                            ncols=100, 
                            colour="green",
                            leave=True,
                            smoothing=0.1):
            try:
                new_chunks = self._process_jsonl_file(file_path)
                chunks_processed += new_chunks
                logger.info(f"处理文件: {file_path}, 提取了 {new_chunks} 个文本块")
                
                # 检查是否达到最大块数限制
                if self.max_chunks and chunks_processed >= self.max_chunks:
                    logger.info(f"已达到最大块数限制 ({self.max_chunks})，停止加载")
                    break
                    
            except Exception as e:
                logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
        
        logger.info(f"共加载了 {len(self.token_chunks)} 个文本块")
    
    def _process_jsonl_file(self, file_path):
        """
        处理单个JSONL文件，优化内存使用
        
        Args:
            file_path: JSONL文件路径
        
        Returns:
            处理的文本块数量
        """
        new_chunks = 0
        # 优化：使用迭代器读取文件而非一次性加载所有行
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(tqdm(f, 
                                              desc=f"处理 {Path(file_path).name}", 
                                              leave=False, 
                                              ncols=100, 
                                              colour="blue",
                                              dynamic_ncols=True)):
                try:
                    # 解析JSON行
                    line = line.strip()
                    if not line:
                        continue
                    
                    item = json.loads(line)
                    
                    # 提取文本内容，优先使用content字段
                    content = None
                    
                    # 尝试不同的字段名称
                    for field in ['content', 'text', 'body', 'paragraphs']:
                        if field in item and item[field]:
                            content = item[field]
                            break
                    
                    # 如果没有找到内容，尝试合并标题和内容
                    if content is None and 'title' in item:
                        content = item.get('title', '')
                    
                    # 如果内容是列表，合并为字符串
                    if isinstance(content, list):
                        content = '\n'.join([str(p) for p in content if p])
                    
                    # 确保内容是字符串
                    if not isinstance(content, str) or not content.strip():
                        continue
                    
                    # 添加到token块
                    self.token_chunks.append(content)
                    new_chunks += 1
                    
                    # 检查是否达到最大块数限制
                    if self.max_chunks and len(self.token_chunks) >= self.max_chunks:
                        break
                    
                except Exception as e:
                    logger.warning(f"处理第 {line_idx+1} 行时出错: {str(e)}")
                    
        return new_chunks
    
    def _create_samples(self):
        """创建训练样本，使用向量化操作提高性能"""
        logger.info("开始创建训练样本...")
        start_time = time.time()
        
        # 如果有分词器，使用分词器处理文本
        all_tokens = []
        
        # 批量处理文本
        batch_size = 100  # 批处理大小
        for i in range(0, len(self.token_chunks), batch_size):
            batch = self.token_chunks[i:i+batch_size]
            
            if self.tokenizer:
                # 使用批量分词优化
                if hasattr(self.tokenizer, 'batch_tokenize'):
                    token_batches = self.tokenizer.batch_tokenize(batch)
                    for tokens in token_batches:
                        if tokens and len(tokens) > 1:
                            all_tokens.extend(tokens)
                else:
                    # 传统方式
                    for chunk in batch:
                        try:
                            tokens = self.tokenizer.tokenize(chunk)
                            if tokens and len(tokens) > 1:
                                all_tokens.extend(tokens)
                        except Exception as e:
                            logger.warning(f"分词文本时出错: {str(e)}")
            else:
                # 字符级编码
                for chunk in batch:
                    chars = list(chunk)
                    # 将字符转换为简单的整数ID
                    char_ids = [ord(c) % 30000 for c in chars]
                    if char_ids:
                        all_tokens.extend(char_ids)
        
        # 确保有足够的token
        if len(all_tokens) < self.context_length:
            logger.warning(f"总token数 {len(all_tokens)} 小于上下文长度 {self.context_length}，将重复数据")
            # 重复数据直到达到要求
            repeats = (self.context_length // len(all_tokens)) + 1
            all_tokens = all_tokens * repeats
        
        # 性能优化：使用numpy数组
        all_tokens_array = self.np.array(all_tokens, dtype=self.np.int32)
        
        # 优化：对于大型数据集，使用内存映射文件
        if len(all_tokens_array) > 10_000_000:  # 1千万个token以上使用内存映射
            self.use_memmap = True
            import tempfile
            
            # 创建临时文件路径
            fd, self.memmap_path = tempfile.mkstemp(suffix='.dat')
            os.close(fd)  # 关闭文件描述符
            
            # 创建内存映射文件
            tokens_memmap = self.np.memmap(self.memmap_path, 
                                      dtype=self.np.int32, 
                                      mode='w+', 
                                      shape=all_tokens_array.shape)
            
            # 写入数据
            tokens_memmap[:] = all_tokens_array[:]
            tokens_memmap.flush()
            
            # 重新打开以只读模式
            all_tokens_array = self.np.memmap(self.memmap_path, 
                                         dtype=self.np.int32, 
                                         mode='r', 
                                         shape=all_tokens_array.shape)
            
            logger.info(f"使用内存映射文件存储 {len(all_tokens_array)} 个token")
        
        # 创建滑动窗口样本 - 向量化处理
        total_samples = max(0, (len(all_tokens_array) - self.context_length) // self.stride + 1)
        logger.info(f"将创建 {total_samples} 个训练样本")
        
        if total_samples > 0:
            # 预分配内存
            input_arrays = []
            target_arrays = []
            
            # 使用tqdm显示进度
            with tqdm(total=total_samples, 
                     desc="创建训练样本", 
                     ncols=100, 
                     colour="cyan",
                     leave=True) as pbar:
                
                for i in range(0, len(all_tokens_array) - self.context_length, self.stride):
                    # 切片获取输入和目标
                    input_slice = all_tokens_array[i:i + self.context_length]
                    target_slice = all_tokens_array[i + 1:i + self.context_length + 1]
                    
                    if len(input_slice) == self.context_length and len(target_slice) == self.context_length:
                        input_arrays.append(input_slice)
                        target_arrays.append(target_slice)
                    
                    pbar.update(1)
                    
                    # 定期检查内存使用
                    if len(input_arrays) % 1000 == 0:
                        self._track_memory()
            
            # 将所有样本合并为一个列表
            for i in range(len(input_arrays)):
                self.samples.append({
                    'input_ids': input_arrays[i],
                    'target_ids': target_arrays[i]
                })
        
        elapsed = time.time() - start_time
        logger.info(f"样本创建完成，耗时: {elapsed:.2f}秒，共创建 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 获取样本
        sample = self.samples[idx]
        
        # 转换为tensor以便训练
        input_tensor = torch.tensor(sample['input_ids'], dtype=torch.long)
        target_tensor = torch.tensor(sample['target_ids'], dtype=torch.long)
        
        return input_tensor, target_tensor
        
    def __del__(self):
        """清理资源"""
        # 如果使用了内存映射文件，在析构时删除
        if self.use_memmap and self.memmap_path and os.path.exists(self.memmap_path):
            try:
                os.unlink(self.memmap_path)
                logger.debug(f"已删除临时内存映射文件: {self.memmap_path}")
            except:
                pass

# ================= 模型定义 =================

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, dropout=0.1, max_len=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.max_len = max_len
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, src):
        # src shape: [batch_size, seq_len]
        
        # 将输入转换为嵌入向量
        src = self.embedding(src) * math.sqrt(self.d_model)
        
        # 添加位置编码
        src = self.pos_encoder(src)
        
        # 通过Transformer编码器
        output = self.transformer_encoder(src)
        
        # 生成预测
        output = self.output_layer(output)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ================= 训练部分 =================

# 添加夜间模式检测
def is_night_mode():
    """检查当前是否应该使用夜间模式（晚上9点到早上8点）"""
    current_hour = datetime.now().hour
    return current_hour >= 21 or current_hour < 8

# 添加设置CPU限制的函数
def set_cpu_limit(limit_cores=None):
    """限制CPU核心使用"""
    if limit_cores is None:
        # 自动设置，夜间模式使用较少核心
        if is_night_mode():
            limit_cores = max(2, psutil.cpu_count(logical=False) // 2)
        else:
            limit_cores = psutil.cpu_count(logical=False)
    
    # 尝试设置进程亲和性
    try:
        p = psutil.Process()
        if limit_cores < psutil.cpu_count():
            # 设置CPU亲和性，使用前N个CPU核心
            p.cpu_affinity(list(range(limit_cores)))
        return True
    except Exception as e:
        log_warning(f"无法设置CPU限制: {str(e)}")
        return False

# 添加GPU内存限制函数
def limit_gpu_memory(percent=None):
    """限制GPU内存使用百分比"""
    if not torch.cuda.is_available():
        return False
    
    if percent is None:
        # 自动设置，夜间模式使用较少内存
        percent = 50 if is_night_mode() else 90
    
    try:
        # 设置内存限制
        total_memory = torch.cuda.get_device_properties(0).total_memory
        limit = int(total_memory * percent / 100)
        torch.cuda.set_per_process_memory_fraction(percent / 100)
        return True
    except Exception as e:
        log_warning(f"无法设置GPU内存限制: {str(e)}")
        return False

# 添加查找最新检查点的函数
def find_latest_checkpoint(model_save_dir):
    """查找最新的检查点文件"""
    checkpoint_files = glob.glob(os.path.join(model_save_dir, f"checkpoint_epoch*_v{VERSION}.pt"))
    if not checkpoint_files:
        return None
    
    # 按文件修改时间排序
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    return checkpoint_files[0]

def save_checkpoint(epoch, model, optimizer, scheduler, scaler, stats, save_dir, name=None, is_best=False):
    """保存训练检查点
    
    Args:
        epoch: 当前训练轮次
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        scaler: 梯度缩放器(AMP)
        stats: 训练统计信息
        save_dir: 保存目录
        name: 检查点名称
        is_best: 是否是最佳模型
    """
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # 默认名称
    if name is None:
        name = f"checkpoint_epoch_{epoch+1}"
    
    # 构建保存路径
    checkpoint_path = save_dir / f"{name}.pt"
    
    # 保存检查点
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler': scaler.state_dict() if scaler else None,
        'stats': stats
    }
    
    # 保存检查点
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"模型检查点保存至: {checkpoint_path}")
    
    # 如果是最佳模型，额外保存一份
    if is_best:
        best_path = save_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"最佳模型保存至: {best_path}")
        
    return checkpoint_path

def train_model(
    train_file, 
    test_file=None,
    model_save_dir="model_weights",
    tokenizer_path="tokenizer.json",
    context_length=512,
    batch_size=8,
    learning_rate=5e-5,
    epochs=10,
    d_model=768,
    nhead=12,
    num_layers=12,
    dim_feedforward=3072,
    dropout=0.1,
    use_amp=True,
    checkpoint_every=1,
    accumulation_steps=4,
    device=None,
    max_grad_norm=1.0,
    weight_decay=0.01,
    resume_from=None,
    auto_resume=True,
    night_mode=True,
    save_on_interrupt=True
):
    """
    训练语言模型
    
    Args:
        train_file: 训练数据文件路径
        test_file: 测试数据文件路径，用于验证
        model_save_dir: 模型保存目录
        tokenizer_path: 分词器路径
        context_length: 上下文长度
        batch_size: 批次大小
        learning_rate: 学习率
        epochs: 训练轮数
        d_model: 模型维度
        nhead: 注意力头数
        num_layers: Transformer层数
        dim_feedforward: 前馈网络维度
        dropout: Dropout率
        use_amp: 是否使用混合精度训练
        checkpoint_every: 每多少轮保存一次检查点
        accumulation_steps: 梯度累积步数
        device: 训练设备
        max_grad_norm: 梯度裁剪最大范数
        weight_decay: 权重衰减系数
        resume_from: 从特定检查点恢复训练
        auto_resume: 自动恢复最近的检查点
        night_mode: 是否启用夜间低功耗模式
        save_on_interrupt: 中断时是否保存
    """
    # 避免局部变量与全局变量冲突，重新导入必要模块
    import os as os_local
    from pathlib import Path as Path_local
    
    # 打印欢迎信息
    print_section_header("灵猫墨韵 语言模型训练")
    logger.info(f"训练版本: v{VERSION}")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 打印配置信息，更简洁的格式
    print_section_header("训练配置")
    
    # 使用简洁格式打印核心配置
    config_info = [
        f"训练数据: {train_file}",
        f"测试数据: {'无' if not test_file else test_file}",
        f"模型保存目录: {model_save_dir}",
        f"上下文长度: {context_length}",
        f"批次大小: {batch_size} (有效批量大小: {batch_size * accumulation_steps})",
        f"学习率: {learning_rate}",
        f"训练轮数: {epochs}"
    ]
    
    # 简洁打印模型架构参数
    model_info = [
        f"模型维度: {d_model}",
        f"注意力头数: {nhead}",
        f"Transformer层数: {num_layers}",
        f"前馈网络维度: {dim_feedforward}",
        f"Dropout率: {dropout}"
    ]
    
    # 简洁打印训练策略参数
    train_info = [
        f"梯度累积步数: {accumulation_steps}",
        f"梯度裁剪阈值: {max_grad_norm}",
        f"权重衰减: {weight_decay}",
        f"混合精度训练: {'是' if use_amp else '否'}",
        f"夜间模式: {'是' if night_mode else '否'}",
        f"自动恢复训练: {'是' if auto_resume else '否'}"
    ]
    
    # 分组打印参数
    for info in config_info:
        logger.info(info)
    logger.info("模型结构:")
    for info in model_info:
        logger.info(f"  {info}")
    logger.info("训练策略:")
    for info in train_info:
        logger.info(f"  {info}")
    
    # 应用夜间模式设置
    if night_mode and is_night_mode():
        logger.info("🌙 检测到夜间时段，启用低功耗模式")
        set_cpu_limit()
        # 夜间模式下调整批次大小和梯度累积
        original_batch_size = batch_size
        batch_size = max(2, batch_size // 2)
        accumulation_steps = accumulation_steps * original_batch_size // batch_size
        logger.info(f"夜间模式: 批次大小调整为 {batch_size}, 梯度累积步数调整为 {accumulation_steps}")
    
    # 确保模型保存目录存在
    os_local.makedirs(model_save_dir, exist_ok=True)
    
    # 设置设备
    if device is None:
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("训练设备: mps (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"训练设备: cuda ({torch.cuda.get_device_name(0)})")
            # 在夜间模式下限制GPU内存使用
            if night_mode and is_night_mode():
                limit_gpu_memory(50)  # 夜间使用50%GPU内存
        else:
            device = torch.device("cpu")
            logger.info("训练设备: cpu")
    
    device_info = f"训练设备: {device}"
    if device.type == 'cuda':
        device_info += f" ({torch.cuda.get_device_name(0)})"
        log_info(f"GPU内存: 总计 {format_memory_size(torch.cuda.get_device_properties(0).total_memory)}")
    elif device.type == 'mps':
        device_info += " (Apple Silicon GPU)"
    log_info(device_info)
    
    # 加载分词器
    print_section_header("加载分词器")
    vocab_size = 50000  # 默认词汇表大小
    
    if os_local.path.exists(tokenizer_path):
        try:
            from tokenizer import ClassicalTokenizer
            tokenizer = ClassicalTokenizer()
            tokenizer.load(tokenizer_path)
            logger.info("分词器加载成功，词汇量：{}".format(len(tokenizer.token_to_id)))
            vocab_size = len(tokenizer.token_to_id)
        except Exception as e:
            error_msg = f"加载分词器时出错: {str(e)}，将使用默认字符级分词"
            logger.warning(error_msg)
            tokenizer = None
    else:
        logger.warning(f"分词器文件 {tokenizer_path} 不存在，将使用默认字符级分词")
        tokenizer = None
    
    # 加载数据集
    print_section_header("加载数据集")
    logger.info(f"加载训练数据: {train_file}")
    
    # 创建数据集
    train_dataset = LMDataset(train_file, context_length, tokenizer, stride=context_length//2)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 使用0以避免多进程问题
        pin_memory=True if device != "cpu" else False,
    )
    
    # 可选的测试数据集
    test_loader = None
    if test_file and os_local.path.exists(test_file):
        logger.info(f"加载测试数据: {test_file}")
        test_dataset = LMDataset(
            test_file, context_length, tokenizer, stride=context_length
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device != "cpu" else False,
        )
    
    # 创建模型
    print_section_header("创建模型")
    logger.info("初始化模型架构...")
    
    # 创建模型
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    
    # 计算模型参数数量
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数数量: {param_count:,}")
    
    # 检查是否从指定检查点恢复
    if resume_from:
        if os_local.path.exists(resume_from):
            logger.info(f"从检查点恢复训练: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            initial_epoch = checkpoint.get('epoch', 0)
            best_loss = checkpoint.get('best_loss', float('inf'))
            logger.info(f"从第 {initial_epoch + 1} 轮继续训练")
        else:
            logger.warning(f"指定的检查点 {resume_from} 不存在，将从头开始训练")
            initial_epoch = 0
            best_loss = float('inf')
    # 自动查找最新检查点恢复
    elif auto_resume:
        latest_checkpoint = find_latest_checkpoint(model_save_dir)
        if latest_checkpoint:
            logger.info(f"找到最新检查点: {latest_checkpoint}")
            try:
                checkpoint = torch.load(latest_checkpoint, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                initial_epoch = checkpoint.get('epoch', 0)
                best_loss = checkpoint.get('best_loss', float('inf'))
                logger.info(f"从第 {initial_epoch + 1} 轮继续训练")
            except Exception as e:
                logger.warning(f"加载检查点出错: {str(e)}，将从头开始训练")
                initial_epoch = 0
                best_loss = float('inf')
        else:
            logger.info("未找到有效检查点，将从头开始训练")
            initial_epoch = 0
            best_loss = float('inf')
    else:
        initial_epoch = 0
        best_loss = float('inf')
    
    # 设置优化器
    logger.info("设置优化器和学习率调度器...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 如果从检查点恢复，尝试加载优化器状态
    if 'optimizer_state_dict' in locals().get('checkpoint', {}):
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("已恢复优化器状态")
        except:
            logger.warning("无法恢复优化器状态，将使用新的优化器")
    
    # 设置学习率调度器
    def lr_lambda(current_step):
        # 线性预热后的余弦退火
        warmup_steps = 1000
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, 10000 - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 设置混合精度训练
    scaler = None
    if use_amp:
        # 仅在CUDA设备上启用混合精度训练
        if device.type == 'cuda':
            logger.info("混合精度训练已启用，将提升训练速度")
            scaler = GradScaler()
        else:
            logger.info("当前设备不支持混合精度训练，已自动禁用")
            use_amp = False
    
    # 训练统计
    stats = {
        'losses': [],
        'learning_rates': [],
        'epoch_times': [],
        'gpu_memory_usage': [] if torch.cuda.is_available() else None
    }
    
    # 记录初始学习率
    stats['learning_rates'].append(scheduler.get_last_lr()[0])
    
    # 开始训练
    print_section_header("开始训练")
    logger.info(f"共 {epochs} 轮训练，从第 {initial_epoch+1} 轮开始")
    
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(initial_epoch, epochs):
        # 打印当前轮次信息
        print_section_header(f"第 {epoch+1}/{epochs} 轮训练")
        
        # 切换到训练模式
        model.train()
        
        # 训练循环变量
        total_loss = 0.0
        num_batches = len(train_loader)
        batch_times = []
        
        # 使用简化的进度条
        progress_bar = tqdm(train_loader, 
                          desc=f"第 {epoch+1}/{epochs} 轮训练",
                          ncols=100, 
                          leave=True,
                          dynamic_ncols=True,
                          position=0,
                          ascii=False,
                          smoothing=0.1)
        
        # 重置优化器梯度
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # 将数据移动到指定设备
            try:
                # 处理新旧两种数据格式（兼容性处理）
                if isinstance(batch, dict):
                    # 旧格式：字典形式{'input_ids': tensor, 'target_ids': tensor}
                    input_ids = batch['input_ids'].to(device)
                    target_ids = batch['target_ids'].to(device)
                elif isinstance(batch, tuple) and len(batch) == 2:
                    # 新格式：元组形式(input_tensor, target_tensor)
                    input_ids, target_ids = batch
                    input_ids = input_ids.to(device)
                    target_ids = target_ids.to(device)
                else:
                    logger.warning(f"未知的批处理格式: {type(batch)}, 跳过此批次")
                    continue
                
                # 混合精度训练
                with autocast(enabled=use_amp):
                    # 前向计算
                    outputs = model(input_ids)  # [B, L, V]
                    
                    # 计算损失
                    loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                    loss = loss / accumulation_steps  # 梯度累积
                
                # 反向传播
                scaler.scale(loss).backward()
                
                # 梯度累积和更新
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                    # 梯度裁剪
                    if max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    # 更新参数
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    
                    # 重置累积计数器
                    steps_since_update = 0
                
                # 更新进度条信息
                total_loss += loss.item() * accumulation_steps
                current_lr = scheduler.get_last_lr()[0]
                average_loss = total_loss / (step + 1)
                
                progress_bar.set_postfix({
                    'loss': f"{average_loss:.4f}",
                    'lr': f"{current_lr:.8f}"
                })
                
                # 更新训练统计
                current_step = epoch * len(train_loader) + step
                
                # 记录训练统计
                stats['losses'].append(average_loss)
                stats['learning_rates'].append(current_lr)
                stats['steps'].append(current_step)
                
                # 记录GPU内存使用
                if torch.cuda.is_available():
                    stats['gpu_memory_usage'].append(torch.cuda.memory_allocated(0))
                
            except Exception as e:
                logger.error(f"训练中出错 (step {step}): {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        # 计算epoch平均损失
        epoch_loss = total_loss / len(train_loader)
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        stats['epoch_times'].append(epoch_time)
        
        # 输出epoch统计信息
        logger.info(f"Epoch {epoch+1}/{epochs} 完成: 损失 = {epoch_loss:.4f}, 耗时 = {format_time(epoch_time)}")
        
        # 在测试集上评估
        if test_loader is not None:
            test_loss = evaluate_model(model, test_loader, criterion, device, use_amp)
            logger.info(f"测试集损失: {test_loss:.4f}")
            stats['test_losses'].append(test_loss)
        
        # 保存检查点
        if (epoch + 1) % checkpoint_every == 0 or epoch == epochs - 1:
            save_checkpoint(
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                stats,
                model_save_dir,
                name=f"model_epoch_{epoch+1}",
                is_best=(epoch == epochs - 1)
            )
        
        # 更新epoch开始时间
        epoch_start_time = time.time()
    
    # 训练结束，绘制图表和总结
    print_section_header("训练完成")
    
    # 绘制训练统计图表
    log_info("生成训练统计图表...")
    plot_training_stats(stats, model_save_dir)
    
    # 保存最终模型
    final_model_path = os_local.path.join(model_save_dir, f"final_model_v{VERSION}.pt")
    torch.save(model.state_dict(), final_model_path)
    log_success(f"✅ 训练完成！最终模型已保存至: {final_model_path}")
    
    # 打印训练总结
    training_summary = [
        f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"总训练轮数: {epoch + 1}",
        f"最佳测试损失: {best_loss:.4f}",
        f"总训练样本数: {len(train_dataset)}"
    ]
    
    log_info("训练总结:")
    for summary in training_summary:
        logger.info(summary)
    
    return model

def evaluate_model(model, test_loader, criterion, device, use_amp=True):
    """在测试集上评估模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="评估中", colour="blue", 
                           ncols=100, leave=False)
        
        for step, batch in enumerate(progress_bar):
            try:
                # 处理新旧两种数据格式（兼容性处理）
                if isinstance(batch, dict):
                    # 旧格式：字典形式{'input_ids': tensor, 'target_ids': tensor}
                    input_ids = batch['input_ids'].to(device)
                    target_ids = batch['target_ids'].to(device)
                elif isinstance(batch, tuple) and len(batch) == 2:
                    # 新格式：元组形式(input_tensor, target_tensor)
                    input_ids, target_ids = batch
                    input_ids = input_ids.to(device)
                    target_ids = target_ids.to(device)
                else:
                    logger.warning(f"未知的批处理格式: {type(batch)}, 跳过此批次")
                    continue
                
                # 混合精度评估
                with autocast(enabled=use_amp):
                    # 前向计算
                    outputs = model(input_ids)
                    
                    # 计算损失
                    loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                
                total_loss += loss.item()
                average_loss = total_loss / (step + 1)
                
                progress_bar.set_postfix({
                    'loss': f"{average_loss:.4f}"
                })
                
            except Exception as e:
                logger.error(f"评估中出错 (step {step}): {str(e)}")
                continue
    
    return total_loss / len(test_loader)

if __name__ == "__main__":
    # 设置日志系统
    logger = setup_logger()  # 确保只初始化一次
    
    # 简化欢迎信息
    logger.info(f"灵猫墨韵语言模型训练系统 v{VERSION}")
    logger.info("自动配置模式：将使用最优参数和可用设备")
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="灵猫语言模型训练")
    
    # 数据参数
    parser.add_argument("--train_file", type=str, default="dataset/train_data_train.jsonl", help="训练数据文件路径")
    parser.add_argument("--test_file", type=str, default=None, help="测试数据文件路径")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.json", help="分词器路径")
    parser.add_argument("--context_length", type=int, default=512, help="上下文长度")
    
    # 模型参数
    parser.add_argument("--d_model", type=int, default=768, help="模型维度")
    parser.add_argument("--nhead", type=int, default=12, help="注意力头数")
    parser.add_argument("--num_layers", type=int, default=12, help="Transformer层数")
    parser.add_argument("--dim_feedforward", type=int, default=3072, help="前馈网络维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--checkpoint_every", type=int, default=1, help="每多少轮保存一次检查点")
    parser.add_argument("--model_save_dir", type=str, default="model_weights", help="模型保存目录")
    parser.add_argument("--use_amp", dest='use_amp', action='store_true', help="使用混合精度训练")
    parser.add_argument("--no_use_amp", dest='use_amp', action='store_false', help="禁用混合精度训练")
    parser.set_defaults(use_amp=True)
    parser.add_argument("--no_cuda", action="store_true", help="禁用CUDA")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪最大范数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减系数")
    
    # 恢复训练相关参数
    parser.add_argument("--resume_from", type=str, default=None, help="从指定检查点恢复训练")
    parser.add_argument("--auto_resume", action="store_true", default=True, help="自动从最新检查点恢复训练")
    parser.add_argument("--no_auto_resume", dest='auto_resume', action='store_false', help="禁用自动恢复训练")
    
    # 性能与电源管理参数
    parser.add_argument("--night_mode", action="store_true", default=True, help="启用夜间低功耗模式")
    parser.add_argument("--no_night_mode", dest='night_mode', action='store_false', help="禁用夜间低功耗模式")
    parser.add_argument("--save_on_interrupt", action="store_true", default=True, help="中断时保存检查点")
    parser.add_argument("--no_save_on_interrupt", dest='save_on_interrupt', action='store_false', help="禁用中断时保存检查点")
    parser.add_argument("--clean-before-run", action="store_true", default=False, help="在开始训练前清理旧的日志和图表")
    
    args = parser.parse_args()
    
    # --- 开始前的清理工作 --- #
    if args.clean_before_run:
        logger.info("执行开始前的清理任务 (--clean-before-run)...")
        # 1. 清理训练日志目录
        train_log_dir = Path("logs/train_model")
        if train_log_dir.exists():
            try:
                shutil.rmtree(train_log_dir)
                logger.info(f"已清理训练日志目录: {train_log_dir}")
                # 清理后重新创建日志目录，避免 setup_logger 失败
                train_log_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"清理训练日志目录 {train_log_dir} 时出错: {e}")
        
        # 2. 清理旧的统计图表
        stats_dir = Path(args.model_save_dir)
        cleaned_plots = 0
        if stats_dir.exists():
            for plot_file in stats_dir.glob("training_stats_*.png"):
                try:
                    plot_file.unlink()
                    cleaned_plots += 1
                except Exception as e:
                    logger.warning(f"删除图表 {plot_file} 时出错: {e}")
            for plot_file in stats_dir.glob("training_stats_*.pdf"):
                try:
                    plot_file.unlink()
                    cleaned_plots += 1
                except Exception as e:
                    logger.warning(f"删除图表 {plot_file} 时出错: {e}")
            if cleaned_plots > 0:
                logger.info(f"已清理 {cleaned_plots} 个旧的统计图表文件于 {stats_dir}")
        logger.info("开始前的清理任务完成。")

    # 设置设备
    if args.no_cuda:
        device = torch.device("cpu")
    else:
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    # 训练模型
    train_model(
        train_file=args.train_file,
        test_file=args.test_file,
        model_save_dir=args.model_save_dir,
        tokenizer_path=args.tokenizer_path,
        context_length=args.context_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        use_amp=args.use_amp,
        checkpoint_every=args.checkpoint_every,
        accumulation_steps=args.accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        device=device,
        resume_from=args.resume_from,
        auto_resume=args.auto_resume,
        night_mode=args.night_mode,
        save_on_interrupt=args.save_on_interrupt
    )