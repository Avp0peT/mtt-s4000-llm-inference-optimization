"""测试包。

加 ``__init__.py`` 是为了让 ``python -m unittest discover -t .`` 能正常发现用例
（否则 discover 会报 "Start directory is not importable"）。
各测试文件自行把 ``src/`` 与 ``benchmark/`` 插入 sys.path，
因此这里不做任何全局路径处理。
"""
