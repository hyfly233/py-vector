import tempfile
from pathlib import Path

import aiofiles
import pytest

from py_vector.core.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """文档处理器测试类"""

    @pytest.fixture
    def processor(self):
        """创建文档处理器实例"""
        return DocumentProcessor()

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def sample_texts(self):
        """示例文本内容"""
        return {
            'simple': "这是一个简单的测试文档。",
            'long': "这是一个很长的文档。" + "测试内容。" * 100,
            'multi_paragraph': """第一段内容。

第二段内容，包含更多信息。

第三段内容。""",
            'unicode': "这是包含中文、English和éñglish的文档。",
            'empty': "",
            'whitespace': "   \n\t   \n   "
        }

    @pytest.mark.asyncio
    async def test_save_temp_file(self, processor, temp_dir):
        """测试保存临时文件"""
        # 修复：确保content是bytes类型
        content = "测试文件内容".encode('utf-8')  # 转换为bytes
        filename = "test.txt"

        # 临时修改处理器的临时目录
        original_temp_dir = processor.temp_dir
        processor.temp_dir = temp_dir

        try:
            file_path = await processor.save_temp_file(content, filename)

            assert file_path.exists()
            assert file_path.name.endswith("_test.txt")
            assert file_path.parent == temp_dir

            # 验证文件内容
            async with aiofiles.open(file_path, 'rb') as f:
                saved_content = await f.read()
                assert saved_content == content

        finally:
            processor.temp_dir = original_temp_dir

    @pytest.mark.asyncio
    async def test_save_temp_file_text_mode(self, processor, temp_dir):
        """测试保存临时文件（文本模式验证）"""
        content_str = "测试文件内容"
        content_bytes = content_str.encode('utf-8')
        filename = "test.txt"

        original_temp_dir = processor.temp_dir
        processor.temp_dir = temp_dir

        try:
            file_path = await processor.save_temp_file(content_bytes, filename)

            assert file_path.exists()
            assert file_path.name.endswith("_test.txt")
            assert file_path.parent == temp_dir

            # 验证文件内容（文本模式读取）
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                saved_content = await f.read()
                assert saved_content == content_str

        finally:
            processor.temp_dir = original_temp_dir

    def test_get_supported_types(self, processor):
        """测试获取支持的文件类型"""
        supported_types = processor.get_supported_types()

        assert isinstance(supported_types, list)
        assert len(supported_types) > 0
        assert '.txt' in supported_types

    def test_is_supported_file(self, processor):
        """测试文件类型支持检查"""
        # 支持的文件类型
        assert processor.is_supported_file("test.txt") == True

        # 不支持的文件类型
        assert processor.is_supported_file("image.jpg") == False
        assert processor.is_supported_file("video.mp4") == False

        # 边界情况
        assert processor.is_supported_file("") == False
        assert processor.is_supported_file("noextension") == False

    def create_temp_txt_file(self, temp_dir: Path, content: str, filename: str = "test.txt") -> Path:
        """创建临时TXT文件"""
        file_path = temp_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path

    @pytest.mark.asyncio
    async def test_process_document(self, processor, temp_dir, sample_texts):
        """测试主文档处理方法"""
        file_path = self.create_temp_txt_file(temp_dir, sample_texts['multi_paragraph'])

        result = await processor.process_document(file_path)

        assert result['status'] == 'success'
        assert len(result['chunks']) >= 1
        assert result['file_name'] == file_path.name
        assert result['file_size'] > 0
        assert result['processing_time'] > 0
        assert len(result['document_hash']) == 64  # SHA256 hash length
