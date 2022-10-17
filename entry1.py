#!/usr/bin/env python
import subprocess
import os

file = '/home/hlcv_team014/ablation1/VideoCompression_WithoutMaskGenerator/entry2.sh'
os.chmod(file, 0o0777)
subprocess.call(['sh', file])
