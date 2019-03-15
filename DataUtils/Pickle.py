# @Author : bamtercelboo
# @Datetime : 2018/8/23 12:26
# @File : Pickle.py
# @Last Modify Time : 2018/8/23 12:26
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Pickle.py
    FUNCTION : None
"""

# Introduce python packages
import sys
import os
import time
import pickle
# import _pickle as pickle

# Introduce missing packages in here


class Pickle(object):
    """
     Pickle
    """
    def __init__(self):
        print("Pickle")
        self.obj_count = 0

    @staticmethod
    def save(obj, path, mode="wb"):
        """
        :param obj:  obj dict to dump
        :param path: save path
        :param mode:  file mode
        """
        start_time = time.time()
        print("Save Obj To {}".format(path))
        # print("obj", obj)
        assert isinstance(obj, dict), "The type of obj must be a dict type."
        if os.path.exists(path):
            os.remove(path)
        pkl_file = open(path, mode=mode)
        pickle.dump(obj, pkl_file)
        pkl_file.close()
        end_time = time.time()
        print("Save Obj Cost {:.4f}.".format(end_time - start_time))

    @staticmethod
    def load(path, mode="rb"):
        """
        :param path:  pkl path
        :param mode: file mode
        :return: data dict
        """
        start_time = time.time()
        print("Load Obj From {}".format(path))
        if os.path.exists(path) is False:
            print("Path {} illegal.".format(path))
        pkl_file = open(path, mode=mode)
        data = pickle.load(pkl_file)
        pkl_file.close()
        end_time = time.time()
        print("Load Obj Cost {:.4f}.".format(end_time - start_time))
        return data


pcl = Pickle




