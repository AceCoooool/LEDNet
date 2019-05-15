"""Base dataset methods."""
import os
from torch.utils import data


class ClassProperty(object):
    """Readonly @ClassProperty descriptor for internal usage."""

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class SimpleDataset(data.Dataset):
    """Simple Dataset wrapper for lists and arrays.

    Parameters
    ----------
    data : dataset-like object
        Any object that implements `len()` and `[]`.
    """

    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class _LazyTransformDataset(data.Dataset):
    """Lazily transformed dataset."""

    def __init__(self, data, fn):
        super(_LazyTransformDataset, self).__init__()
        self._data = data
        self._fn = fn

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[idx]
        if isinstance(item, tuple):
            return self._fn(*item)
        return self._fn(item)

    def transform(self, fn):
        self._fn = fn


class VisionDataset(data.Dataset):
    """Base Dataset with directory checker.

    Parameters
    ----------
    root : str
        The root path of xxx.names, by default is '~/.mxnet/datasets/foo', where
        `foo` is the name of the dataset.
    """

    def __init__(self, root):
        super(VisionDataset, self).__init__()
        if not os.path.isdir(os.path.expanduser(root)):
            helper_msg = "{} is not a valid dir. Did you forget to initialize \
                         datasets described in: \
                         `http://gluon-cv.mxnet.io/build/examples_datasets/index.html`? \
                         You need to initialize each dataset only once.".format(root)
            raise OSError(helper_msg)

    @property
    def classes(self):
        raise NotImplementedError

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.classes)

    def transform(self, fn, lazy=True):
        """Returns a new dataset with each sample transformed by the
        transformer function `fn`.

        Parameters
        ----------
        fn : callable
            A transformer function that takes a sample as input and
            returns the transformed sample.
        lazy : bool, default True
            If False, transforms all samples at once. Otherwise,
            transforms each sample on demand. Note that if `fn`
            is stochastic, you must set lazy to True or you will
            get the same result on all epochs.

        Returns
        -------
        Dataset
            The transformed dataset.
        """
        trans = _LazyTransformDataset(self, fn)
        if lazy:
            return trans
        return SimpleDataset([i for i in trans])
