import os
import os.path as osp
import subprocess
import tempfile
from collections import OrderedDict

import cv2
from cv2 import (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS,
                 CAP_PROP_FRAME_COUNT, CAP_PROP_FOURCC,
                 CAP_PROP_POS_FRAMES)
from utils.helpers.file_helper import FileHelper

from lib.utils.tools.progressbar import track_progress


class Cache(object):

    def __init__(self, capacity):
        self._cache = OrderedDict()
        self._capacity = int(capacity)
        if capacity <= 0:
            raise ValueError('capacity must be a positive integer')

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return len(self._cache)

    def put(self, key, val):
        if key in self._cache:
            return
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = val

    def get(self, key, default=None):
        val = self._cache[key] if key in self._cache else default
        return val


class VideoReader(object):
    """"""

    def __init__(self, filename, cache_capacity=10):
        FileHelper.check_file_exist(filename, 'Video file not found: ' + filename)
        self._vcap = cv2.VideoCapture(filename)
        assert cache_capacity > 0
        self._cache = Cache(cache_capacity)
        self._position = 0
                        
        self._width = int(self._vcap.get(CAP_PROP_FRAME_WIDTH))
        self._height = int(self._vcap.get(CAP_PROP_FRAME_HEIGHT))
        self._fps = int(round(self._vcap.get(CAP_PROP_FPS)))
        self._frame_cnt = int(self._vcap.get(CAP_PROP_FRAME_COUNT))
        self._fourcc = self._vcap.get(CAP_PROP_FOURCC)

    @property
    def vcap(self):
        """"""
        return self._vcap

    @property
    def opened(self):
        """"""
        return self._vcap.isOpened()

    @property
    def width(self):
        """"""
        return self._width

    @property
    def height(self):
        """"""
        return self._height

    @property
    def resolution(self):
        """"""
        return (self._width, self._height)

    @property
    def fps(self):
        """"""
        return self._fps

    @property
    def frame_cnt(self):
        """"""
        return self._frame_cnt

    @property
    def fourcc(self):
        """"""
        return self._fourcc

    @property
    def position(self):
        """"""
        return self._position

    def _get_real_position(self):
        return int(round(self._vcap.get(CAP_PROP_POS_FRAMES)))

    def _set_real_position(self, frame_id):
        self._vcap.set(CAP_PROP_POS_FRAMES, frame_id)
        pos = self._get_real_position()
        for _ in range(frame_id - pos):
            self._vcap.read()
        self._position = frame_id

    def read(self):
        """"""
                              
        if self._cache:
            img = self._cache.get(self._position)
            if img is not None:
                ret = True
            else:
                if self._position != self._get_real_position():
                    self._set_real_position(self._position)
                ret, img = self._vcap.read()
                if ret:
                    self._cache.put(self._position, img)
        else:
            ret, img = self._vcap.read()
        if ret:
            self._position += 1
        return img

    def get_frame(self, frame_id):
        """"""
        if frame_id < 0 or frame_id >= self._frame_cnt:
            raise IndexError(
                '"frame_id" must be between 0 and {}'.format(self._frame_cnt -
                                                             1))
        if frame_id == self._position:
            return self.read()
        if self._cache:
            img = self._cache.get(frame_id)
            if img is not None:
                self._position = frame_id + 1
                return img
        self._set_real_position(frame_id)
        ret, img = self._vcap.read()
        if ret:
            if self._cache:
                self._cache.put(self._position, img)
            self._position += 1
        return img

    def current_frame(self):
        """"""
        if self._position == 0:
            return None
        return self._cache.get(self._position - 1)

    def cvt2frames(self,
                   frame_dir,
                   file_start=0,
                   filename_tmpl='{:06d}.jpg',
                   start=0,
                   max_num=0,
                   show_progress=True):
        """"""
        FileHelper.make_dirs(frame_dir)
        if max_num == 0:
            task_num = self.frame_cnt - start
        else:
            task_num = min(self.frame_cnt - start, max_num)
        if task_num <= 0:
            raise ValueError('start must be less than total frame number')
        if start > 0:
            self._set_real_position(start)

        def write_frame(file_idx):
            img = self.read()
            filename = osp.join(frame_dir, filename_tmpl.format(file_idx))
            cv2.imwrite(filename, img)

        if show_progress:
            track_progress(write_frame, range(file_start, file_start + task_num))

        else:
            for i in range(task_num):
                img = self.read()
                if img is None:
                    break
                filename = osp.join(frame_dir,
                                    filename_tmpl.format(i + file_start))
                cv2.imwrite(filename, img)

    def __len__(self):
        return self.frame_cnt

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self.get_frame(i)
                for i in range(*index.indices(self.frame_cnt))
            ]
                                   
        if index < 0:
            index += self.frame_cnt
            if index < 0:
                raise IndexError('index out of range')
        return self.get_frame(index)

    def __iter__(self):
        self._set_real_position(0)
        return self

    def __next__(self):
        img = self.read()
        if img is not None:
            return img
        else:
            raise StopIteration

    next = __next__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._vcap.release()




class VideoHelper(object):

    @staticmethod
    def convert_video(in_file, out_file, print_cmd=False, pre_options='', **kwargs):
        """"""
        options = []
        for k, v in kwargs.items():
            if isinstance(v, bool):
                if v:
                    options.append('-{}'.format(k))
            elif k == 'log_level':
                assert v in [
                    'quiet', 'panic', 'fatal', 'error', 'warning', 'info',
                    'verbose', 'debug', 'trace'
                ]
                options.append('-loglevel {}'.format(v))
            else:
                options.append('-{} {}'.format(k, v))
        cmd = 'ffmpeg -y {} -i {} {} {}'.format(pre_options, in_file,
                                                ' '.join(options), out_file)
        if print_cmd:
            print(cmd)
        subprocess.call(cmd, shell=True)

    @staticmethod
    def resize_video(in_file,
                     out_file,
                     size=None,
                     ratio=None,
                     keep_ar=False,
                     log_level='info',
                     print_cmd=False,
                     **kwargs):
        """"""
        if size is None and ratio is None:
            raise ValueError('expected size or ratio must be specified')
        elif size is not None and ratio is not None:
            raise ValueError('size and ratio cannot be specified at the same time')
        options = {'log_level': log_level}
        if size:
            if not keep_ar:
                options['vf'] = 'scale={}:{}'.format(size[0], size[1])
            else:
                options['vf'] = ('scale=w={}:h={}:force_original_aspect_ratio'
                                 '=decrease'.format(size[0], size[1]))
        else:
            if not isinstance(ratio, tuple):
                ratio = (ratio, ratio)
            options['vf'] = 'scale="trunc(iw*{}):trunc(ih*{})"'.format(ratio[0], ratio[1])

        VideoHelper.convert_video(in_file, out_file, print_cmd, **options)

    @staticmethod
    def cut_video(in_file,
                  out_file,
                  start=None,
                  end=None,
                  vcodec=None,
                  acodec=None,
                  log_level='info',
                  print_cmd=False,
                  **kwargs):
        """"""
        options = {'log_level': log_level}
        if vcodec is None:
            options['vcodec'] = 'copy'
        if acodec is None:
            options['acodec'] = 'copy'
        if start:
            options['ss'] = start
        else:
            start = 0
        if end:
            options['t'] = end - start

        VideoHelper.convert_video(in_file, out_file, print_cmd, **options)

    @staticmethod
    def concat_video(video_list,
                     out_file,
                     vcodec=None,
                     acodec=None,
                     log_level='info',
                     print_cmd=False,
                     **kwargs):
        """"""
        _, tmp_filename = tempfile.mkstemp(suffix='.txt', text=True)
        with open(tmp_filename, 'w') as f:
            for filename in video_list:
                f.write('file {}\n'.format(osp.abspath(filename)))
        options = {'log_level': log_level}
        if vcodec is None:
            options['vcodec'] = 'copy'
        if acodec is None:
            options['acodec'] = 'copy'

        VideoHelper.convert_video(tmp_filename, out_file, print_cmd, pre_options='-f concat -safe 0', **options)
        os.remove(tmp_filename)

    @staticmethod
    def frames2video(frame_dir,
                     video_file,
                     fps=30,
                     fourcc='XVID',
                     filename_tmpl='{:06d}.jpg',
                     start=0,
                     end=0,
                     show_progress=True):
        """"""

        first_file = osp.join(frame_dir, filename_tmpl.format(start))
        img = cv2.imread(first_file)
        height, width = img.shape[:2]
        resolution = (width, height)
        vwriter = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*fourcc), fps,
                                  resolution)

        def write_frame(file_idx):
            filename = osp.join(frame_dir, filename_tmpl.format(file_idx))
            img = cv2.imread(filename)
            vwriter.write(img)

        if show_progress:
            track_progress(write_frame, range(start, end))
        else:
            for i in range(start, end):
                filename = osp.join(frame_dir, filename_tmpl.format(i))
                img = cv2.imread(filename)
                vwriter.write(img)
        vwriter.release()

