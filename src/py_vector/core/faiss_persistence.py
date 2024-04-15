import json
import logging
import os
import pickle
import shutil
from datetime import datetime
from typing import Dict, Any, List

import faiss
import numpy as np


class FAISSPersistence:
    """FAISS索引持久化类，负责保存和加载FAISS索引、元数据和配置信息"""

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.index_file = os.path.join(base_path, "faiss_index.bin")
        self.metadata_file = os.path.join(base_path, "metadata.pkl")
        self.config_file = os.path.join(base_path, "config.json")

        os.makedirs(base_path, exist_ok=True)

    def save_index(self, index: faiss.Index, metadata: Dict[str, Any], config: Dict[str, Any]):
        """保存索引、元数据和配置"""
        try:
            # 保存 FAISS 索引
            faiss.write_index(index, self.index_file)

            # 保存元数据（文档信息、向量ID映射等）
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)

            # 保存配置信息（维度、模型名称等）
            config['saved_at'] = datetime.now().isoformat()
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            print(f"索引保存成功: {self.index_file}")

        except Exception as e:
            raise Exception(f"保存失败: {str(e)}")

    def load_index(self) -> tuple:
        """加载索引、元数据和配置"""
        try:
            # 检查文件是否存在
            if not all(os.path.exists(f) for f in [self.index_file, self.metadata_file, self.config_file]):
                return None, None, None

            # 加载 FAISS 索引
            index = faiss.read_index(self.index_file)

            # 加载元数据
            with open(self.metadata_file, 'rb') as f:
                metadata = pickle.load(f)

            # 加载配置
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            logging.info(f"✅ 索引加载成功: {index.ntotal} 个向量")
            return index, metadata, config

        except Exception as e:
            raise Exception(f"加载失败: {str(e)}")


class IncrementalFAISS:
    """增量FAISS索引类，支持索引的增量更新和备份"""

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.backup_path = os.path.join(base_path, "backups")
        self.persistence = FAISSPersistence(base_path)

        os.makedirs(self.backup_path, exist_ok=True)

    def save_with_backup(self, index: faiss.Index, metadata: Dict[str, Any], config: Dict[str, Any]):
        """保存并创建备份"""
        try:
            # 创建备份
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(self.backup_path, f"backup_{timestamp}")

            # 如果存在旧文件，先备份
            if os.path.exists(self.persistence.index_file):
                os.makedirs(backup_dir, exist_ok=True)
                shutil.copy2(self.persistence.index_file, backup_dir)
                shutil.copy2(self.persistence.metadata_file, backup_dir)
                shutil.copy2(self.persistence.config_file, backup_dir)
                print(f"备份创建: {backup_dir}")

            # 保存新数据
            self.persistence.save_index(index, metadata, config)

            # 清理旧备份（保留最近5个）
            self._cleanup_old_backups(keep_count=5)

        except Exception as e:
            raise Exception(f"备份保存失败: {str(e)}")

    def _cleanup_old_backups(self, keep_count: int = 5):
        """清理旧备份文件"""
        try:
            backups = [d for d in os.listdir(self.backup_path)
                       if d.startswith("backup_") and os.path.isdir(os.path.join(self.backup_path, d))]
            backups.sort(reverse=True)

            for backup in backups[keep_count:]:
                backup_path = os.path.join(self.backup_path, backup)
                shutil.rmtree(backup_path)
                print(f"删除旧备份: {backup}")

        except Exception as e:
            print(f"清理备份失败: {e}")


class ShardedFAISS:
    """分片FAISS索引类，支持将向量分片存储以提高性能和可扩展性"""

    def __init__(self, base_path: str, shard_size: int = 100000):
        self.base_path = base_path
        self.shard_size = shard_size
        self.shards = []

        os.makedirs(base_path, exist_ok=True)

    def add_to_shard(self, vectors: np.ndarray, metadata: List[Dict]):
        """添加向量到分片"""
        current_shard = len(self.shards) - 1 if self.shards else -1

        # 检查是否需要新分片
        if current_shard < 0 or len(self.shards[current_shard]['metadata']) >= self.shard_size:
            self._create_new_shard()
            current_shard = len(self.shards) - 1

        # 添加到当前分片
        shard = self.shards[current_shard]
        shard['index'].add(vectors)
        shard['metadata'].extend(metadata)

    def _create_new_shard(self):
        """创建新分片"""
        shard_id = len(self.shards)
        dimension = 128  # 根据实际情况设置

        shard = {
            'id': shard_id,
            'index': faiss.IndexFlatL2(dimension),
            'metadata': [],
            'file_path': os.path.join(self.base_path, f"shard_{shard_id}.faiss")
        }

        self.shards.append(shard)

    def save_all_shards(self):
        """保存所有分片"""
        for shard in self.shards:
            # 保存索引
            faiss.write_index(shard['index'], shard['file_path'])

            # 保存元数据
            metadata_file = shard['file_path'].replace('.faiss', '_metadata.pkl')
            with open(metadata_file, 'wb') as f:
                pickle.dump(shard['metadata'], f)

    def load_all_shards(self):
        """加载所有分片"""
        shard_files = [f for f in os.listdir(self.base_path) if f.startswith('shard_') and f.endswith('.faiss')]

        for shard_file in sorted(shard_files):
            shard_path = os.path.join(self.base_path, shard_file)
            metadata_path = shard_path.replace('.faiss', '_metadata.pkl')

            # 加载索引
            index = faiss.read_index(shard_path)

            # 加载元数据
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            shard = {
                'id': len(self.shards),
                'index': index,
                'metadata': metadata,
                'file_path': shard_path
            }

            self.shards.append(shard)


# 基本使用
def example_basic_persistence():
    # 创建数据
    dimension = 128
    data = np.random.random((1000, dimension)).astype('float32')

    # 创建索引
    index = faiss.IndexFlatL2(dimension)
    index.add(data)

    # 准备元数据
    metadata = {
        'documents': [{'id': i, 'text': f'document_{i}'} for i in range(1000)],
        'total_count': 1000
    }

    config = {
        'dimension': dimension,
        'index_type': 'IndexFlatL2',
        'model_name': 'text-embedding-ada-002'
    }

    # 保存
    persistence = FAISSPersistence('./faiss_data')
    persistence.save_index(index, metadata, config)

    # 加载
    loaded_index, loaded_metadata, loaded_config = persistence.load_index()
    print(f"加载的索引大小: {loaded_index.ntotal}")


# 增量保存示例
def example_incremental_save():
    incremental = IncrementalFAISS('./faiss_data')

    # 模拟数据更新
    for i in range(5):
        dimension = 128
        data = np.random.random((100, dimension)).astype('float32')
        index = faiss.IndexFlatL2(dimension)
        index.add(data)

        metadata = {'batch': i, 'count': 100}
        config = {'dimension': dimension, 'batch': i}

        incremental.save_with_backup(index, metadata, config)
        print(f"保存批次 {i}")
