from setuptools import find_packages, setup

setup(
    name="gg_bench",
    version="1.0",
    packages=find_packages(include=["gg_bench", "gg_bench.*"]),
    install_requires=open("requirements.txt").read().splitlines(),
    entry_points={
        "console_scripts": [
            "gg-bench=gg_bench.cli:main",
        ],
    },
)
