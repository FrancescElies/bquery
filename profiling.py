#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import numpy as np
import bcolz as bz
import bquery as bq
import tempfile
import shutil
import os
import pstats, cProfile

import pyximport
pyximport.install()

N = int(1e7)
rootdir = tempfile.mkdtemp()
os.rmdir(rootdir)
ra = np.fromiter(((str(i % 100), i * 2.) for i in xrange(N)), dtype='S3,i8')
ct = bz.ctable(ra, rootdir=rootdir)
ct = bq.open(rootdir)

cProfile.runctx("ct.groupby(['f0'], ['f1'])", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats('time', 'calls').print_stats(20)

shutil.rmtree(rootdir)
