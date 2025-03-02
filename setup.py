from setuptools import setup

setup(
    name='vllm_mps',
    author="jiangkuaixue",
    license="Apache 2.0",
    description=("vLLM MPS backend"),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=["vllm_mps"],
    python_requires=">=3.9",
    extras_require={},
    entry_points={'vllm.platform_plugins': ["mps = vllm_mps:register"]})
