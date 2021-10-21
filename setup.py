#!/usr/bin/env python

import os
import re

from setuptools import setup, find_packages


PACKAGE_NAME = "crowdnalysis"
HERE = os.path.dirname(__file__)
PACKAGE_DIR = "src"

CLASSIFIERS = """
License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)
Operating System :: OS Independent
Intended Audience :: Science/Research
Topic :: Scientific/Engineering :: Information Analysis
Programming Language :: Python :: 3
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
"""

HOMEPAGE = "https://github.com/Crowd4SDG/crowdnalysis"

PROJECT_URLS = {
    "Tutorial": "https://github.com/Crowd4SDG/crowdnalysis/blob/master/nb/tutorial.ipynb",
    "Bug Tracker": "https://github.com/Crowd4SDG/crowdnalysis/issues"
}

KEYWORDS = "citizen-science crowdsourcing annotation-aggregation annotator-model prospective-analysis"


def readme() -> str:
    with open(os.path.join(HERE, 'README.md'), 'r') as f:
        content = f.read()
    return content


def requirements() -> str:
    with open(os.path.join(HERE, "requirements.txt"), "r") as f:
        content = f.read()
    return content.splitlines()


def get_version() -> str:
    version_file = open(os.path.join(HERE, PACKAGE_DIR, PACKAGE_NAME, "_version.py"))
    version_contents = version_file.read()
    return re.search("__version__ = \"(.*?)\"", version_contents).group(1)


setup(
    name=PACKAGE_NAME,
    version=get_version(),
    author="IIIA-CSIC",
    description="Library to help analyze crowdsourcing results",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url=HOMEPAGE,
    project_urls=PROJECT_URLS,
    install_requires=requirements(),
    classifiers=CLASSIFIERS.strip().split('\n'),
    keywords=KEYWORDS,
    package_dir={"": PACKAGE_DIR},
    packages=find_packages(where=PACKAGE_DIR),
    package_data={PACKAGE_NAME: ['cmdstan/cmdstan/*.stan']},
    python_requires=">=3.7"
)
