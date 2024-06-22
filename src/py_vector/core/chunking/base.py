from abc import ABC, abstractmethod


class Chunker(ABC):
    """文本分块策略基类"""

    @abstractmethod
    def chunk(self, text: str, chunk_size: int, overlap: int = 0) -> list[str]:
        """将文本分割成块

        Args:
            text: 输入文本
            chunk_size: 每块最大字符数
            overlap: 相邻块之间的重叠字符数

        Returns:
            文本块列表
        """
        ...
