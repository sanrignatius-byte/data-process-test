"""Setup script for the multimodal contrastive learning data pipeline."""

from setuptools import setup, find_packages

setup(
    name="mineru-contrastive-data",
    version="1.0.0",
    description="Multimodal Contrastive Learning Data Factory using MinerU",
    author="Data Process Team",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
        "Pillow>=10.0.0",
        "openai>=1.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "embeddings": ["sentence-transformers>=2.2.0"],
        "anthropic": ["anthropic>=0.18.0"],
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "isort>=5.12.0"],
    },
    entry_points={
        "console_scripts": [
            "run-pipeline=scripts.run_pipeline:main",
        ],
    },
)
