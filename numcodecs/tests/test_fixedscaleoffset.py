import itertools

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from numcodecs.fixedscaleoffset import FixedScaleOffset
from numcodecs.tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode,
    check_repr,
)

arrays = [
    np.linspace(1000, 1001, 1000, dtype='<f8'),
    np.random.normal(loc=1000, scale=1, size=1000).astype('<f8'),
    np.linspace(1000, 1001, 1000, dtype='<f8').reshape(100, 10),
    np.linspace(1000, 1001, 1000, dtype='<f8').reshape(100, 10, order='F'),
    np.linspace(1000, 1001, 1000, dtype='<f8').reshape(10, 10, 10),
]


codecs = [
    FixedScaleOffset(offset=1000, scale=10, dtype='<f8', astype='<i1'),
    FixedScaleOffset(offset=1000, scale=10**2, dtype='<f8', astype='<i2'),
    FixedScaleOffset(offset=1000, scale=10**6, dtype='<f8', astype='<i4'),
    FixedScaleOffset(offset=1000, scale=10**12, dtype='<f8', astype='<i8'),
    FixedScaleOffset(offset=1000, scale=10**12, dtype='<f8'),
]

nan_arrays = [
    np.array([np.nan, 0.0111, 0.0222, 0.0333, 0.0444, 0.0555, 0.0666, 0.0777, 0.0888, 0.1], dtype='f4'),
    np.array([np.nan, 0.0111, 0.0222, 0.0333, 0.0444, 0.0555, 0.0666, 0.0777, 0.0888, 0.1], dtype='f4') + 10,
]

nan_codecs = [
    FixedScaleOffset(offset=0, scale=100, dtype='<f4', astype='<i2', fill_value=np.iinfo('<i2').min),
    FixedScaleOffset(offset=10, scale=100, dtype='<f4', astype='<i2', fill_value=np.iinfo('<i2').min),
]


def test_encode_decode():
    for arr, codec in itertools.product(arrays, codecs):
        precision = int(np.log10(codec.scale))
        check_encode_decode(arr, codec, precision=precision)
    
    for arr, codec in zip(nan_arrays, nan_codecs):
        precision = int(np.log10(codec.scale))
        check_encode_decode(arr, codec, precision=precision)


def test_encode():
    dtype = '<f8'
    astype = '|u1'
    codec = FixedScaleOffset(scale=10, offset=1000, dtype=dtype, astype=astype)
    arr = np.linspace(1000, 1001, 10, dtype=dtype)
    expect = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10], dtype=astype)
    actual = codec.encode(arr)
    assert_array_equal(expect, actual)
    assert np.dtype(astype) == actual.dtype


def test_encode_nan():
    dtype = '<f4'
    astype = '<i2'
    codec = FixedScaleOffset(
        offset=0, scale=100, dtype=dtype, astype=astype, fill_value=np.iinfo(astype).min
    )
    arr = np.array(
        [np.nan, 0.0111, 0.0222, 0.0333, 0.0444, 0.0555, 0.0666, 0.0777, 0.0888, 0.1],
        dtype=dtype
    )
    expect = np.array([-32768, 1, 2, 3, 4, 6, 7, 8, 9, 10], dtype=astype)
    actual = codec.encode(arr)
    assert_array_equal(expect, actual)
    assert np.dtype(astype) == actual.dtype


def test_config():
    for codec in codecs + nan_codecs:
        check_config(codec)


def test_repr():
    stmt = "FixedScaleOffset(scale=10, offset=100, dtype='<f8', astype='<i4')"
    check_repr(stmt)
    stmt = "FixedScaleOffset(scale=100, offset=0, dtype='<f4', astype='<i2', fill_value=-32768)"
    check_repr(stmt)


def test_backwards_compatibility():
    precision = [int(np.log10(codec.scale)) for codec in codecs]
    check_backwards_compatibility(FixedScaleOffset.codec_id, arrays, codecs, precision=precision)
    precision = [int(np.log10(codec.scale)) for codec in nan_codecs]
    check_backwards_compatibility(FixedScaleOffset.codec_id, nan_arrays, nan_codecs, precision=precision, prefix='fill_value')

def test_errors():
    with pytest.raises(ValueError):
        FixedScaleOffset(dtype=object, astype='i4', scale=10, offset=100)
    with pytest.raises(ValueError):
        FixedScaleOffset(dtype='f8', astype=object, scale=10, offset=100)
    with pytest.raises(TypeError):
        FixedScaleOffset(
            offset=0, scale=100, dtype='i4', astype='i2', fill_value=np.iinfo('i2').min
        )
    with pytest.raises(TypeError):
        FixedScaleOffset(offset=0, scale=100, dtype='f4', astype='i2', fill_value='bad')
    with pytest.raises(TypeError):
        FixedScaleOffset(offset=0, scale=100, dtype='f4', astype='i2', fill_value=3.1)
    with pytest.raises(NotImplementedError):
        FixedScaleOffset(offset=0, scale=100, dtype='f8', astype='f4', fill_value=3.1)
    with pytest.raises(ValueError):
        FixedScaleOffset(offset=0, scale=100, dtype='f8', astype='i1', fill_value=3000)
