__version__ = '1.1.2'

import logging
lgr = logging.getLogger('remodnav')

import sys
import numpy as np

from .clf import (
    EyegazeClassifier,
    events2bids_events_tsv,
)


def pred(data):

    # import argparse
    # import inspect

    # kwargs = {}
    # for func in (EyegazeClassifier.__init__, EyegazeClassifier.preproc):
    #     # pull kwargs and their defaults out of the function definitions
    #     argspec = inspect.getfullargspec(func)
    #     kwargs.update(zip(argspec.args[::-1], argspec.defaults[::-1]))

    # parser = argparse.ArgumentParser(
    #     prog='remodnav',
    #     description='{}'.format(
    #         EyegazeClassifier.__doc__,
    #     ),
    #     formatter_class=argparse.RawDescriptionHelpFormatter,
    # )
    # parser.add_argument(
    #     'infile', metavar='<datafile>',
    #     help="""Data file with eye gaze recordings to process. The first two
    #     columns in this file must contain X and Y coordinates, while each line
    #     is a timepoint (no header). The file is read with NumPy's recfromcsv
    #     and may be compressed.""")
    # parser.add_argument(
    #     'outfile', metavar='<eventfile>',
    #     help="""Output file name. This file will contain information on all
    #     detected eye movement events in BIDS events.tsv format.""")
    # parser.add_argument(
    #     'px2deg', type=float, metavar='<PX2DEG>',
    #     help="""Factor to convert pixel coordinates to visual degrees, i.e.
    #     the visual angle of a single pixel. Pixels are assumed to be square.
    #     This will typically be a rather small value.""")
    # parser.add_argument(
    #     'sampling_rate', type=float, metavar='<SAMPLING RATE>',
    #     help="""Sampling rate of the data in Hertz. Only data with dense
    #     regular sampling are supported.""")
    # parser.add_argument(
    #     '--log-level', choices=('debug', 'info', 'warn', 'error'),
    #     metavar='level', default='warn',
    #     help="""debug|info|warn|error. 'info' and 'debug' levels enable output
    #     of increasing levels of detail on the algorithm steps and decision
    #     making. Default: warn""")

    # for argname, default in sorted(kwargs.items(), key=lambda x: x[0]):
    #     parser.add_argument(
    #         '--{}'.format(argname.replace('_', '-')),
    #         dest=argname,
    #         metavar='<float>' if argname != 'savgol-polyord' else '<int>',
    #         type=float if argname != 'savgol-polyord' else int,
    #         default=default,
    #         help=help[argname] + ' [default: {}]'.format(default))

    # args = parser.parse_args(args[1:])

    # logging.basicConfig(
    #     format='%(levelname)s:%(message)s',
    #     level=getattr(logging, args.log_level.upper()))

    # data = np.recfromcsv(
    #     args.infile,
    #     delimiter='\t',
    #     names=['x', 'y'],
    #     usecols=[0, 1])
    # lgr.info('Read %i samples', len(data))

    # clf = EyegazeClassifier(
    #     **{k: getattr(args, k) for k in (
    #         'px2deg', 'sampling_rate', 'velthresh_startvelocity',
    #         'min_intersaccade_duration', 'min_saccade_duration',
    #         'min_pursuit_duration', 'pursuit_velthresh',
    #         'max_initial_saccade_freq', 'saccade_context_window_length',
    #         'max_pso_duration', 'min_fixation_duration', 'lowpass_cutoff_freq',
    #         'noise_factor')}
    # )

    px2deg = 0.001
    sampling_rate = 150
    clf = EyegazeClassifier(px2deg, sampling_rate, )

    pp = clf.preproc(data)
    # pp = clf.preproc(
    #     data,
    #     **{k: getattr(args, k) for k in (
    #         'min_blink_duration', 'dilate_nan', 'median_filter_length',
    #         'savgol_length', 'savgol_polyord', 'max_vel')}
    # )

    events = clf(pp, classify_isp=True, sort_events=False)

    # events2bids_events_tsv(events, args.outfile)

    return events
