"""
 * @Author: Yong Guo 
 * @Date: 2018-08-19 17:21:46 
 * @Last Modified by:   Yong Guo 
 * @Last Modified time: 2018-08-19 17:21:46 
 """

from logger import *
import random
import datetime
from os import makedirs
from os.path import exists, join

file_path = '/home/young/tflog/%s/logs'%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

if not exists(file_path):
    makedirs(file_path)

my_logger = Logger(file_path)

run_count = 0

for i in range(10):
    my_logger.scalar_summary("a", random.randint(1, 10000), run_count)
    my_logger.scalar_summary("b", float(random.randint(1, 10000))/10000, run_count)
    run_count += 1