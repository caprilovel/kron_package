from setuptools import setup, find_packages
# 使用 python setup.py sdist bdist_wheel 生成安装包
# 使用 pip install --upgrade dist/global_utils-0.0.1-py3-none-any.whl 安装
# 修改版本号来进行更换版本
setup(
    name='kpd',
    version='0.1.0',
    description='Kronecker Product Decomposition',
    author='Capri',
) 