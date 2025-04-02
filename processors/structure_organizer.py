#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
灵猫墨韵古典文学数据处理系统 - 结构组织工具
"""

import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from collections import defaultdict


class StructureOrganizer:
    """古典文学结构组织工具"""
    
    def __init__(self, config):
        """
        初始化结构组织工具
        
        Args:
            config: 配置字典
        """
        self.logger = logging.getLogger("LingmaoMoyun.StructureOrganizer")
        self.config = config
        
        # 朝代顺序
        self.dynasty_order = [
            "先秦", "秦", "汉", "三国", "晋", "南北朝", 
            "隋", "唐", "五代", "宋", "金", "元", "明", "清", "近代", "现代"
        ]
        
        # 文体分类
        self.genre_mapping = {
            "诗": ["诗", "律诗", "绝句", "古诗"],
            "词": ["词", "词牌", "词调"],
            "曲": ["曲", "散曲", "套曲", "杂剧"],
            "文": ["文", "散文", "赋", "骈文", "序", "记", "传", "议", "书", "说"],
            "小说": ["小说", "传奇", "笔记", "章回"],
            "其他": []
        }
        
        self.logger.info("结构组织工具初始化完成")
    
    def organize_data(self, data: Union[List[Dict], Dict[str, List[Dict]]]) -> Dict[str, List[Dict]]:
        """
        组织数据
        
        Args:
            data: 原始数据列表或字典
            
        Returns:
            组织后的数据字典
        """
        self.logger.info("开始组织数据")
        
        # 初始化结果字典
        organized = {
            "poems": [],    # 诗词
            "chapters": [], # 章节
            "other": []     # 其他
        }
        
        # 处理不同类型的输入数据
        if isinstance(data, dict):
            # 如果输入是字典，假设它已经按类别组织
            for category, items in data.items():
                if not isinstance(items, list):
                    self.logger.warning(f"类别 {category} 的数据不是列表，跳过")
                    continue
                    
                if category in organized:
                    organized[category].extend(items)
                else:
                    # 尝试将未知类别映射到已知类别
                    if category in ["poem", "poetry", "ci", "诗", "词"]:
                        organized["poems"].extend(items)
                    elif category in ["chapter", "book", "article", "章节", "书籍"]:
                        organized["chapters"].extend(items)
                    else:
                        organized["other"].extend(items)
        elif isinstance(data, list):
            # 如果输入是列表，按项目类型分类
            for item in data:
                if not isinstance(item, dict):
                    self.logger.warning(f"跳过非字典数据项: {type(item)}")
                    continue
                    
                # 判断是否为诗词
                if self._is_poem(item):
                    organized["poems"].append(item)
                # 判断是否为章节
                elif self._is_chapter(item):
                    organized["chapters"].append(item)
                # 其他
                else:
                    organized["other"].append(item)
        else:
            self.logger.error(f"不支持的数据类型: {type(data)}")
        
        self.logger.info(f"组织完成: 诗词 {len(organized['poems'])} 首, "
                        f"章节 {len(organized['chapters'])} 个, "
                        f"其他 {len(organized['other'])} 项")
        
        return organized
    
    def _is_poem(self, item: Dict) -> bool:
        """判断是否为诗词"""
        # 类型检查
        if not isinstance(item, dict):
            return False
            
        # 检查类型字段
        if "type" in item:
            item_type = item["type"]
            if isinstance(item_type, str) and item_type.lower() in ["poem", "poetry", "ci", "诗", "词"]:
                return True
        
        # 判断依据：包含paragraphs字段且为列表，或者包含content字段且较短
        if "paragraphs" in item:
            paragraphs = item.get("paragraphs")
            if isinstance(paragraphs, list):
                return True
            elif isinstance(paragraphs, str):
                # 如果paragraphs是字符串，尝试按行分割
                return len(paragraphs.split("\n")) <= 20
        
        if "content" in item:
            content = item.get("content")
            if not isinstance(content, str):
                return False
                
            # 假设内容少于500字符的是诗词
            if len(content) < 500:
                return True
                
            # 检查行数和行长
            lines = content.split("\n")
            if len(lines) <= 20 and all(len(line) < 30 for line in lines if line.strip()):
                return True
        
        # 根据标题判断
        title = item.get("title", "")
        if isinstance(title, str) and any(keyword in title for keyword in ["诗", "词", "曲", "赋"]):
            return True
        
        return False
    
    def _is_chapter(self, item: Dict) -> bool:
        """判断是否为章节"""
        # 类型检查
        if not isinstance(item, dict):
            return False
            
        # 检查类型字段
        if "type" in item:
            item_type = item["type"]
            if isinstance(item_type, str) and item_type.lower() in ["chapter", "book", "article", "章节", "书籍"]:
                return True
        
        # 判断依据：包含content字段且较长，或者包含chapter字段
        if "chapter" in item or (isinstance(item.get("title", ""), str) and "章" in item.get("title", "")):
            return True
        
        if "content" in item:
            content = item.get("content")
            if not isinstance(content, str):
                return False
                
            # 假设内容大于500字符的是章节
            return len(content) >= 500
        
        return False
    
    def organize_by_dynasty(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        """
        按朝代组织数据
        
        Args:
            data: 原始数据列表
            
        Returns:
            按朝代组织的数据字典
        """
        self.logger.info("开始按朝代组织数据")
        
        # 初始化结果字典
        organized = defaultdict(list)
        
        # 未知朝代
        unknown_dynasty = "未知朝代"
        
        for item in data:
            dynasty = item.get("dynasty", unknown_dynasty)
            if not dynasty:
                dynasty = unknown_dynasty
                
            organized[dynasty].append(item)
        
        self.logger.info(f"按朝代组织完成，共 {len(organized)} 个朝代")
        
        # 按朝代顺序排序
        sorted_organized = {}
        
        # 先添加已知朝代
        for dynasty in self.dynasty_order:
            if dynasty in organized:
                sorted_organized[dynasty] = organized[dynasty]
                
        # 再添加其他朝代
        for dynasty, items in organized.items():
            if dynasty not in sorted_organized:
                sorted_organized[dynasty] = items
        
        return sorted_organized
    
    def organize_by_author(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        """
        按作者组织数据
        
        Args:
            data: 原始数据列表
            
        Returns:
            按作者组织的数据字典
        """
        self.logger.info("开始按作者组织数据")
        
        # 初始化结果字典
        organized = defaultdict(list)
        
        # 未知作者
        unknown_author = "佚名"
        
        for item in data:
            author = item.get("author", unknown_author)
            if not author:
                author = unknown_author
                
            organized[author].append(item)
        
        self.logger.info(f"按作者组织完成，共 {len(organized)} 位作者")
        
        return dict(organized)
    
    def organize_by_genre(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        """
        按文体组织数据
        
        Args:
            data: 原始数据列表
            
        Returns:
            按文体组织的数据字典
        """
        self.logger.info("开始按文体组织数据")
        
        # 初始化结果字典
        organized = defaultdict(list)
        
        # 未知文体
        unknown_genre = "其他"
        
        for item in data:
            genre = self._detect_genre(item)
            if not genre:
                genre = unknown_genre
                
            organized[genre].append(item)
        
        self.logger.info(f"按文体组织完成，共 {len(organized)} 种文体")
        
        return dict(organized)
    
    def _detect_genre(self, item: Dict) -> str:
        """
        检测文体
        
        Args:
            item: 数据项
            
        Returns:
            文体名称
        """
        # 如果有明确的文体字段
        if "genre" in item and item["genre"]:
            return item["genre"]
        
        # 从标题推断
        title = item.get("title", "")
        
        for genre, keywords in self.genre_mapping.items():
            for keyword in keywords:
                if keyword in title:
                    return genre
        
        # 从内容长度和结构推断
        if "paragraphs" in item and isinstance(item.get("paragraphs"), list):
            paragraphs = item["paragraphs"]
            
            # 短小的段落，可能是诗
            if len(paragraphs) <= 8 and all(len(p) < 30 for p in paragraphs):
                return "诗"
            
            # 较长的段落，可能是词
            if len(paragraphs) <= 10 and any(len(p) > 30 for p in paragraphs):
                return "词"
        
        # 默认返回其他
        return "其他"
    
    def create_metadata_index(self, data: List[Dict]) -> Dict[str, Dict]:
        """
        创建元数据索引
        
        Args:
            data: 原始数据列表
            
        Returns:
            元数据索引字典
        """
        self.logger.info("开始创建元数据索引")
        
        # 初始化索引
        index = {
            "title": {},      # 标题索引
            "author": {},     # 作者索引
            "dynasty": {},    # 朝代索引
            "genre": {}       # 文体索引
        }
        
        # 构建索引
        for i, item in enumerate(data):
            # 索引标题
            if "title" in item and item["title"]:
                title = item["title"]
                if title not in index["title"]:
                    index["title"][title] = []
                index["title"][title].append(i)
            
            # 索引作者
            if "author" in item and item["author"]:
                author = item["author"]
                if author not in index["author"]:
                    index["author"][author] = []
                index["author"][author].append(i)
            
            # 索引朝代
            if "dynasty" in item and item["dynasty"]:
                dynasty = item["dynasty"]
                if dynasty not in index["dynasty"]:
                    index["dynasty"][dynasty] = []
                index["dynasty"][dynasty].append(i)
            
            # 索引文体
            genre = self._detect_genre(item)
            if genre not in index["genre"]:
                index["genre"][genre] = []
            index["genre"][genre].append(i)
        
        self.logger.info(f"元数据索引创建完成: {len(index['title'])} 个标题, "
                        f"{len(index['author'])} 位作者, "
                        f"{len(index['dynasty'])} 个朝代, "
                        f"{len(index['genre'])} 种文体")
        
        return index
    
    def match_across_files(self, data: List[Dict]) -> List[Dict]:
        """
        跨文件匹配数据
        
        Args:
            data: 原始数据列表
            
        Returns:
            匹配后的数据列表
        """
        self.logger.info("开始跨文件匹配数据")
        
        # 创建索引
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
        
        # 第二遍：匹配数据
        for item in data:
            # 如果缺少朝代信息，尝试从同标题或同作者的作品中获取
            if ("dynasty" not in item or not item["dynasty"]) and "title" in item:
                title = item["title"]
                if title in title_index:
                    for other_item in title_index[title]:
                        if other_item != item and "dynasty" in other_item and other_item["dynasty"]:
                            item["dynasty"] = other_item["dynasty"]
                            self.logger.debug(f"为 {title} 匹配朝代: {item['dynasty']}")
                            break
            
            # 如果缺少作者信息，尝试从同标题的作品中获取
            if ("author" not in item or not item["author"]) and "title" in item:
                title = item["title"]
                if title in title_index:
                    for other_item in title_index[title]:
                        if other_item != item and "author" in other_item and other_item["author"]:
                            item["author"] = other_item["author"]
                            self.logger.debug(f"为 {title} 匹配作者: {item['author']}")
                            break
            
            # 如果缺少注释信息，尝试从同标题的作品中获取
            if ("notes" not in item or not item["notes"]) and "title" in item:
                title = item["title"]
                if title in title_index:
                    for other_item in title_index[title]:
                        if other_item != item and "notes" in other_item and other_item["notes"]:
                            item["notes"] = other_item["notes"]
                            self.logger.debug(f"为 {title} 匹配注释")
                            break
            
            # 如果缺少翻译信息，尝试从同标题的作品中获取
            if ("translation" not in item or not item["translation"]) and "title" in item:
                title = item["title"]
                if title in title_index:
                    for other_item in title_index[title]:
                        if other_item != item and "translation" in other_item and other_item["translation"]:
                            item["translation"] = other_item["translation"]
                            self.logger.debug(f"为 {title} 匹配翻译")
                            break
        
        self.logger.info("跨文件匹配完成")
        return data


# 简单测试
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 测试配置
    test_config = {}
    
    # 创建组织器
    organizer = StructureOrganizer(test_config)
    
    # 测试数据
    test_data = [
        {
            "title": "静夜思",
            "author": "李白",
            "dynasty": "唐",
            "paragraphs": ["床前明月光，", "疑是地上霜。", "举头望明月，", "低头思故乡。"]
        },
        {
            "title": "长恨歌",
            "author": "白居易",
            "dynasty": "唐",
            "content": "汉皇重色思倾国，御宇多年求不得..."
        },
        {
            "title": "红楼梦",
            "chapter": "第一回",
            "content": "此开卷第一回也。作者自云：曾历过一番梦幻之后，故将真事隐去，而借通灵之说，撰此石头记一书..."
        }
    ]
    
    # 测试组织
    organized = organizer.organize_data(test_data)
    print(f"组织结果: {organized.keys()}")
    
    # 测试按朝代组织
    by_dynasty = organizer.organize_by_dynasty(test_data)
    print(f"按朝代组织: {by_dynasty.keys()}")
