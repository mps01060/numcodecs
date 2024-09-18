import numpy as np

from .abc import Codec
from .compat import ensure_ndarray, ndarray_copy


class FixedScaleOffset(Codec):
    """Simplified version of the scale-offset filter available in HDF5.
    Applies the transformation `(x - offset) * scale` to all chunks. Results
    are rounded to the nearest integer but are not packed according to the
    minimum number of bits.

    Parameters
    ----------
    offset : float
        Value to subtract from data.
    scale : float
        Value to multiply by data.
    dtype : dtype
        Data type to use for decoded data.
    astype : dtype, optional
        Data type to use for encoded data.
    fill_value : integer, optional
        A value used to represent NaNs during encoding when `astype` is an integer
        data type. This allows round-tripping NaN values by encoding them as an 
        integer and decoding them back to NaN. Similar to the `add_offset` and 
        `scale_factor` in netCDF4, `fill_value` ensures NaNs can be preserved 
        during the transformation. It is only relevant when `astype` is an integer 
        dtype and ignored for float types. If not provided, NaNs are not encoded.

    Notes
    -----
    If `astype` is an integer data type, please ensure that it is sufficiently
    large to store encoded values. No checks are made and data may become corrupted
    due to integer overflow if `astype` is too small.

    When `fill_value` is provided and `astype` is an integer dtype, NaNs are 
    encoded as this value and are decoded back to NaNs during the reverse 
    transformation. This is not implemented for astype==float, because `fill_value`
    is not required as NaNs are natively supported by floats.

    Examples
    --------
    >>> import numcodecs
    >>> import numpy as np
    >>> x = np.linspace(1000, 1001, 10, dtype='f8')
    >>> x
    array([1000.        , 1000.11111111, 1000.22222222, 1000.33333333,
           1000.44444444, 1000.55555556, 1000.66666667, 1000.77777778,
           1000.88888889, 1001.        ])
    >>> codec = numcodecs.FixedScaleOffset(offset=1000, scale=10, dtype='f8', astype='u1')
    >>> y1 = codec.encode(x)
    >>> y1
    array([ 0,  1,  2,  3,  4,  6,  7,  8,  9, 10], dtype=uint8)
    >>> z1 = codec.decode(y1)
    >>> z1
    array([1000. , 1000.1, 1000.2, 1000.3, 1000.4, 1000.6, 1000.7,
           1000.8, 1000.9, 1001. ])
    >>> codec = numcodecs.FixedScaleOffset(offset=1000, scale=10**2, dtype='f8', astype='u1')
    >>> y2 = codec.encode(x)
    >>> y2
    array([ 0,  11,  22,  33,  44,  56,  67,  78,  89, 100], dtype=uint8)
    >>> z2 = codec.decode(y2)
    >>> z2
    array([1000.  , 1000.11, 1000.22, 1000.33, 1000.44, 1000.56,
           1000.67, 1000.78, 1000.89, 1001.  ])
    >>> codec = numcodecs.FixedScaleOffset(offset=1000, scale=10**3, dtype='f8', astype='u2')
    >>> y3 = codec.encode(x)
    >>> y3
    array([ 0,  111,  222,  333,  444,  556,  667,  778,  889, 1000], dtype=uint16)
    >>> z3 = codec.decode(y3)
    >>> z3
    array([1000.   , 1000.111, 1000.222, 1000.333, 1000.444, 1000.556,
           1000.667, 1000.778, 1000.889, 1001.   ])
    >>> x_nans = np.linspace(0, 0.1, 10, dtype='f4')
    >>> x_nans[0] = np.nan
    >>> x_nans
    array([       nan, 0.01111111, 0.02222222, 0.03333334, 0.04444445,
           0.05555556, 0.06666667, 0.07777778, 0.08888889, 0.1       ], dtype=float32)
    >>> codec = numcodecs.FixedScaleOffset(offset=0, scale=100, dtype='f4', astype='i2', fill_value=-32768)
    >>> y4 = codec.encode(x_nans)
    >>> y4
    array([-32768, 1, 2, 3, 4, 6, 7, 8, 9, 10], dtype=int16)
    >>> z4 = codec.decode(y4)
    >>> z4
    array([ nan, 0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08, 0.09, 0.1 ], dtype=float32)

    See Also
    --------
    numcodecs.quantize.Quantize

    """

    codec_id = 'fixedscaleoffset'

    def __init__(self, offset, scale, dtype, astype=None, fill_value=None):
        self.offset = offset
        self.scale = scale
        self.dtype = np.dtype(dtype)
        if astype is None:
            self.astype = self.dtype
        else:
            self.astype = np.dtype(astype)
        if self.dtype == np.dtype(object) or self.astype == np.dtype(object):
            raise ValueError('object arrays are not supported')
        if fill_value is not None:
            if np.issubdtype(self.astype, np.floating):
                raise NotImplementedError(
                    'Encoding floats to floats does not require a fill_value '
                    'since floats natively support NaNs.'
                )
            if not np.issubdtype(self.dtype, np.floating):
                raise TypeError(
                    f'fill_value requires a floating-point input dtype, but got dtype "{self.dtype}".'
                )
            if not isinstance(fill_value, (int, np.integer)):
                raise TypeError('fill_value must be an integer value')
            if not np.can_cast(fill_value, self.astype, casting='safe'):
                raise ValueError(
                    f'fill_value "{fill_value}"" cannot be safely cast to output dtype "{self.astype}"'
                )
            # Convert NumPy integer to Python native types for JSON serialization compatibility
            if isinstance(fill_value, np.integer):
                fill_value = int(fill_value)
        self.fill_value = fill_value

    def encode(self, buf):
        # normalise input
        arr = ensure_ndarray(buf).view(self.dtype)

        # flatten to simplify implementation
        arr = arr.reshape(-1, order='A')

        # compute scale offset
        enc = (arr - self.offset) * self.scale

        # change nans to fill_value
        if self.fill_value is not None:
            enc[np.isnan(enc)] = self.fill_value

        # round to nearest integer
        enc = np.around(enc)

        # convert dtype
        enc = enc.astype(self.astype, copy=False)

        return enc

    def decode(self, buf, out=None):
        # interpret buffer as numpy array
        enc = ensure_ndarray(buf).view(self.astype)

        # flatten to simplify implementation
        enc = enc.reshape(-1, order='A')

        # decode scale offset
        dec = (enc / self.scale) + self.offset

        # convert fill_values to nans
        if self.fill_value is not None:
            dec[enc==self.fill_value] = np.nan

        # convert dtype
        dec = dec.astype(self.dtype, copy=False)

        # handle output
        return ndarray_copy(dec, out)

    def get_config(self):
        # override to handle encoding dtypes
        return dict(
            id=self.codec_id,
            scale=self.scale,
            offset=self.offset,
            dtype=self.dtype.str,
            fill_value=self.fill_value,
            astype=self.astype.str,
        )

    def __repr__(self):
        r = f'{type(self).__name__}(scale={self.scale}, offset={self.offset}, dtype={self.dtype.str!r}'
        if self.astype != self.dtype:
            r += f', astype={self.astype.str!r}'
        if self.fill_value is not None:
            r += f', fill_value={self.fill_value}'
        r += ')'
        return r
