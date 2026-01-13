"""Setup configuration for AI Mobile SLM App."""
from setuptools import setup, find_packages
import os

# Read long description from README
long_description = ""
if os.path.exists('README.md'):
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='ai-mobile-slm-app',
    version='0.1.0',
    author='Hernane Bini',
    author_email='hernane@example.com',
    description='AI-Powered Mobile App with Small Language Models (SLM) - Edge AI optimization and offline-first architecture',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Hbini/ai-mobile-slm-app',
    project_urls={
        'Bug Tracker': 'https://github.com/Hbini/ai-mobile-slm-app/issues',
        'Documentation': 'https://github.com/Hbini/ai-mobile-slm-app/wiki',
        'Source Code': 'https://github.com/Hbini/ai-mobile-slm-app',
    },
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.1.0',
        'torchvision>=0.16.0',
        'onnx>=1.14.0',
        'onnxruntime>=1.16.0',
        'fastapi>=0.104.0',
        'uvicorn>=0.24.0',
        'pydantic>=2.5.0',
        'numpy>=1.24.0',
        'pandas>=2.1.0',
        'sqlalchemy>=2.0.20',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'pytest-asyncio>=0.21.0',
            'black>=23.12.0',
            'flake8>=6.1.0',
            'mypy>=1.7.0',
            'isort>=5.13.0',
        ],
        'docs': [
            'sphinx>=7.2.0',
            'sphinx-rtd-theme>=2.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'ai-mobile-slm-app=src.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
