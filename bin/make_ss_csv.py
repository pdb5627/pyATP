#! /usr/bin/env python
#  -*- coding: utf-8 -*-
'''make_ss_csv.py

Outputs steady-state results from ATP LIS file in csv format for easier post-
processing.

'''

import pyATP
import sys
import argparse
import numpy as np
import json


def main(argv=None):

    #parser = argparse.ArgumentParser(usage=__doc__)
    parser = argparse.ArgumentParser()

    parser.add_argument('--busfile', required=False,
                        help='JSON file with a list of buses and branches '
                             'to be extracted as three-phase buses or '
                             'branches. '
                             'Phase voltages and sequence component voltages '
                             'are added to the output file. Bus names should '
                             'not include phase letters; these will be added '
                             'automatically.')
    """Example JSON file for bus and branch names:
    {
    "buses": [
        "BUS1",
        "BUS2",
        "BUS3"
    ],
    "branches": [
        ["BUS1", "BUS2"],
        ["BUS2", "BUS1"],
        ["BUS2", "BUS3"]
    ]
    }"""
    parser.add_argument('ATP_FILE',
                        help='ATP file to be processed. Name of the LIS file is '
                             'inferred from the ATP file name.')
    parser.add_argument('CSV_FILE',
                        nargs='?',
                        default=None,
                        help='Name to output csv file to. Any existing file will '
                             'be overwritten.')
    parser.add_argument('BIN_FILE',
                        nargs='?',
                        default=None,
                        help='Name to output binary (Python pickle format) '
                             'file to. Any existing file will be overwritten.')

    if argv is None:
        argv = sys.argv


    # Set up a logger so any errors can go to file to facilitate debugging
    import logging, os
    from logging.config import dictConfig

    # By default, log to the same directory the program is run from    
    if os.path.exists(os.path.dirname(argv[0])):
        logfile = os.path.join(os.path.dirname(argv[0]), 'make_ss_output.log')
    else:
        logfile = 'make_ss_output.log'

    logging_config = {
        'version': 1,
        'formatters': {
            'file': {'format':
                  '%(asctime)s ' + os.environ['USERNAME'] + ' %(levelname)-8s %(message)s'},
            'console': {'format':
                  '%(levelname)-8s %(message)s'}
            },

        'handlers': {
            'file': {'class': 'logging.FileHandler',
                'filename': logfile,
                'formatter': 'file',
                'level': 'INFO'},
            'console': {'class': 'logging.StreamHandler',
                'formatter': 'console',
                'level': 'INFO'}
            },
        'loggers': {
            'root' : {'handlers': ['file', 'console'],
                'level': 'DEBUG'}
            }
    }

    dictConfig(logging_config)

    logger = logging.getLogger('root')

    try:
        logger.info('Running %s.' % argv[0])
        logger.info('Logging to file %s.' % os.path.abspath(logfile))

        args = parser.parse_args()

        ATP_file = args.ATP_FILE
        LIS_file = pyATP.lis_filename(ATP_file)

        if args.busfile is not None:
            try:
                with open(args.busfile) as f:
                    bus_branch = json.load(f)
                    buses = bus_branch['buses']
                    branches = bus_branch['branches']
            except IOError:
                logger.warning('Bus file %s does not exist. Continuing without '
                               'bus file.' % args.busfile)
                buses = None
                branches = None
        else:
            buses = None
            branches = None

        logger.info('Reading data from %s...' % LIS_file)
        SS_file = pyATP.output_ss_file(LIS_file, args.CSV_FILE, args.BIN_FILE,
                                       buses, branches, RMS_scale=True)
        logger.info('Steady state data saved to %s.' % SS_file)
        if buses is not None:
            ph_voltages, seq_voltages, neg_seq_imbalance = pyATP.process_SS_bus_voltages(LIS_file, buses)
            n_max = np.argmax(neg_seq_imbalance)
            logger.info('Maximum negative-sequence unbalance factor: %.4f at %s' %
                (neg_seq_imbalance[n_max], buses[n_max]))
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as e:
        logger.error('Program error', exc_info=True)

    logger.info('DONE.')

if __name__ =='__main__':
    sys.exit(main())
