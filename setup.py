from setuptools import setup, find_packages

setup(
    name='sr4mdl',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/yuzhTHU/SR4MDL',
    license='MIT',
    author='Zihan yu',
    author_email='yuzh19@tsinghua.org.cn',
    install_requires=[
        'numpy==1.26',
        'torch==2.3.1',
        'sympy==1.12',
        'matplotlib',
        'pandas',
        'scipy',
        'tqdm',
        'pyyaml',
        'psutil',
        'setproctitle',
        'scikit-learn',
        # 'pmlb @ git+https://github.com/EpistasisLab/pmlb.git',
    ],
)
