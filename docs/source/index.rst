.. temfpy documentation master file, created by
   sphinx-quickstart on Tue May 19 22:17:31 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to temfpy's documentation!
==================================

.. image:: https://readthedocs.org/projects/temfpy/badge/?version=latest
    :target: https://temfpy.readthedocs.io/en/latest

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT

.. image:: https://github.com/OpenSourceEconomics/temfpy/workflows/CI/badge.svg
    :target: https://github.com/OpenSourceEconomics/temfpy/actions?query=branch%3Amaster

.. image:: https://app.codacy.com/project/badge/Grade/b9d3b1fd4e2a461aa47e212e80f6a0eb
    :target: https://www.codacy.com/gh/OpenSourceEconomics/temfpy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=OpenSourceEconomics/temfpy&amp;utm_campaign=Badge_Grade

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://codecov.io/gh/OpenSourceEconomics/temfpy/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/OpenSourceEconomics/temfpy

.. image:: https://img.shields.io/badge/zulip-join_chat-brightgreen.svg
    :target: https://ose.zulipchat.com



``temfpy`` is an open source package providing test models and functions for common numerical components in computational economic models

With ``conda`` available on your path, installing and testing ``temfpy`` is as simple as typing

.. code-block:: bash

    $ conda install -c opensourceeconomics temfpy
    $ python -c "import temfpy; temfpy.test()"



Supported by
------------

.. image:: ../_static/images/ose-logo.jpg
	  :target: https://github.com/OpenSourceEconomics

.. toctree::
   :hidden:
   :maxdepth: 2

   installation
   optimization
   uncertainty_quantification
   resources

   how_to/investigate_robustness_starting_values.ipynb
