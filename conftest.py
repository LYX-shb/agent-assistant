"""pytest 配置：把项目根目录加入 sys.path，保证 app 包可被导入"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
