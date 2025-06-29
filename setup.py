from setuptools import setup, find_packages
setup(
    name="PPI-link-prediction-with-optimized-gcn-and-gan",
    version="0.1",
    packages=find_packages(),
    install_requires=[line.strip() for line in open("requirements.txt")],
    python_requires='>=3.8',
    description="Protein-Protein Interaction prediction using optimized GCN and GAN",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/milagjurovska/PPI-link-prediction-with-optimized-gcn-and-gan",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)