from setuptools import setup

meta = {}
with open("./veni/meta.py") as fp:
    exec(fp.read(), meta)

# Package meta-data.
IMPORTNAME = meta['__title__']
PIPNAME = meta['__packagename__']
DESCRIPTION = 'Python package for Deep Learning using forward automatic differentiation.'
URL = 'https://github.com/DSSC-projects/veni'
AUTHOR = meta['__author__']
VERSION = meta['__version__']
KEYWORDS = 'forward-AD'

REQUIRED = [
    'jax', 'jaxlib', 'torch', 'numpy',
]

EXTRAS = {
    'docs': ['sphinx', 'sphinx_rtd_theme'],
}

LDESCRIPTION = (
    ""
)

setup(
    name=PIPNAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LDESCRIPTION,
    author=AUTHOR,
    author_email=MAIL,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords=KEYWORDS,
    url=URL,
    license='MIT',
    packages=["veni"],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    zip_safe=False,
)
