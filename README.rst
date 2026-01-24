=========
atp-utils
=========


.. image:: https://img.shields.io/pypi/v/atp-utils.svg
        :target: https://pypi.python.org/pypi/atp-utils


Module for working with ATP data files, running ATP from a Python
environment, and extracting results.


* Free software: MIT license


Features
--------

* Drive ATP simulations from Python
* Extract steady-state results from LIS files
* Read PL4 binary output files
* Parse LIS file sections (simulation extrema, statistic switching)
* Symmetrical component transformations


Installation
------------

Install from PyPI::

    pip install atp-utils

Or install in development mode::

    pip install -e .[dev]


Usage
-----

.. code-block:: python

    import atp_utils
    from atp_utils import run_ATP, read_pl4
    from atp_utils import sequence_components
    from atp_utils.lis_parsing import get_output_variables


CLI
---

Extract steady-state results to CSV::

    make-ss-csv ATP_FILE [CSV_FILE] [BIN_FILE] [--busfile BUSFILE]


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
