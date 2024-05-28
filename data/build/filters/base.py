from abc import ABC, abstractmethod
from collections import defaultdict
import logging
import os
import shutil
from tqdm import tqdm
from utils.os import list_visible, write_lines

logger = logging.getLogger(__name__)


class BaseDatasetFilter(ABC):
    def __init__(self, path, config):
        self.path = path
        self.config = config
        self.abbrev = None

    @abstractmethod
    def __call__(self, samples):
        raise NotImplementedError

    def _print_stats(self, kept, total):
        """
        Uses logger to print the number of samples kept by filter and the original number of samples.
        :param kept: `int`. Number of samples kept by filter.
        :param total: `int`. Original number of samples supplied to filter.
        :return: None.
        """
        logging.info(f'{self.__class__.__name__} kept {kept} / {total} samples')

    @abstractmethod
    def _update_files(self, keep_old=True):
        raise NotImplementedError


class YouTubeContentFilter(BaseDatasetFilter):

    def __init__(self, path, config):
        super().__init__(path, config)
        self.content_dir = None
        self.msg = None
        self.abbrev = None

    def __call__(self, videoids):
        """
        Applies content-related filter to YouTube videos with IDs `videoids`, as specified by criteria in
        `self._criteria`. Note that if files beyond those covered in `videoids` are found in `self.content_dir`,
        they will be removed from `self.content_dir` and moved to the `old/` directory.
        :param videoids: `list` of YouTube video IDs.
        :return: `list` of filtered YouTube video IDs.
        """
        videoids = set(videoids)
        self._download_content(videoids)
        logging.info(self.msg)
        self.to_remove = []

        files = list_visible(self.content_dir, 'f')
        id2files = defaultdict(list)
        for file in files:
            id2files[file[:file.find('.')]].append(file)
        extra_ids = set(id2files.keys()).difference(set(videoids))
        for videoid in extra_ids:
            self.to_remove.extend(id2files[videoid])
        files = []
        for videoid in videoids:
            files.extend(id2files[videoid])

        for file in tqdm(files):
            content = self._read_content(os.path.join(self.content_dir, file))
            if not self._criteria(content):
                self.to_remove.append(file)

        self._update_files()
        self._print_stats(len(self.videoids), len(videoids))
        return self.videoids

    @abstractmethod
    def _criteria(self, content):
        """
        Intended to determine whether the YouTube subtitle text satisfies certain criteria, and intended to return
        `True` if satisfied, `False` otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def _download_content(self, videoids):
        raise NotImplementedError

    @abstractmethod
    def _read_content(self, path):
        raise NotImplementedError

    def _update_files(self, keep_old=True):
        content_type = os.path.basename(self.content_dir)
        videoids_file = os.path.join(self.path, 'videoids.txt')
        if keep_old:
            old_dir = os.path.join(self.path, 'old')
            old_content_dir = os.path.join(old_dir, content_type)
            os.makedirs(old_content_dir, exist_ok=True)
            videoids_kept = os.path.join(self.path, f'videoids-pre{self.abbrev}.txt')
            os.rename(videoids_file, videoids_kept)
            shutil.move(videoids_kept, old_dir)

        # logging.info('Removing the following files: ' + str(self.to_remove))
        for file in self.to_remove:
            path = os.path.join(self.content_dir, file)
            if keep_old:
                shutil.move(path, old_content_dir)
            else:
                os.remove(path)

        self.videoids = {file[:file.find('.')] for file in list_visible(self.content_dir, 'f')}
        write_lines(self.videoids, videoids_file)
