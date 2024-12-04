import numpy as np

def lowPassFilter(_W_ati_unfiltered, order, bw, ts, buffer):
    """First or second order low-pass filter. 

    Tustin discretization rule.

    Modified from https://github.com/GielVanvelk/Thesis/blob/36fe7ecb367e6944d7afad3c01af594ab6807832/src/ati_iface/src/ati_iface-component.cpp#L206
    
    Parameters
    ----------
    _W_ati_unfiltered: numpy.ndarray
        The unfiltered input. An ndarray with shape: `(6)`.
    order: int, default is 2.
    bw: float, default is 2 * PI *50.
    ts: float, default is 0.03.
    buffer: numpy.ndarray
        An ndarray with shape: `(6, 6)`. Initialized with all zeros.

    Returns
    -------
    numpy.ndarray
        The filtered result. An ndarray with shape: `(6)`.
    buffer: numpy.ndarray
        An ndarray with shape: `(6, 6)`. Need to recursively updated.
    """
    if not isinstance(_W_ati_unfiltered, np.ndarray):
        _W_ati_unfiltered = np.array(_W_ati_unfiltered)
    if not isinstance(buffer, np.ndarray):
        buffer = np.array(buffer)

    assert _W_ati_unfiltered.shape == (6,), \
        '\'_W_ati_unfiltered\' should be size of (6,), but got size of {}'.format(_W_ati_unfiltered.shape)
    assert buffer.shape == (6, 6), \
        '\'buffer\' should be size of (6, 6), but got size of {}'.format(buffer.shape)

    if order == 1:
        b_0 = bw * ts
        b_1 = bw * ts
        b_2 = 0
        a_0 = ts * bw + 2
        a_1 = ts * bw - 2
        a_2 = 0
    elif order == 2:
        b_0 = bw * bw * ts * ts
        b_1 = 2 * bw * bw * ts * ts
        b_2 = bw * bw * ts * ts
        a_0 = 4 * ts * bw + bw * bw * ts * ts + 4
        a_1 = -8 + 2 * bw * bw * ts * ts
        a_2 = -4 * ts * bw + bw * bw * ts * ts + 4
    else:
        print('[Warning] lowPassFilter: Wrong order.')
        return _W_ati_unfiltered

    _W_ati_filtered = np.zeros(6)
    for i in range(6):
        buffer[2, i] = buffer[1, i]
        buffer[1, i] = buffer[0, i]
        buffer[0, i] = _W_ati_unfiltered[i]
        buffer[5, i] = buffer[4, i]
        buffer[4, i] = buffer[3, i]
        buffer[3, i] = (b_0 * buffer[0, i] + b_1 * buffer[1, i] + b_2 * buffer[2, i] - a_1 * buffer[4, i] - a_2 * buffer[5, i]) / a_0
        _W_ati_filtered[i] = buffer[3, i]

    return _W_ati_filtered, buffer


def avgSmooth(_W_ati_unfiltered, buffer):
    """Average past three frame as smoothing

    Parameters
    ----------
    _W_ati_unfiltered: numpy.ndarray
        The unfiltered input. An ndarray with shape: `(6)`.
    buffer: numpy.ndarray
        An ndarray with shape: `(6, 6)`. Initialized with all zeros.

    Returns
    -------
    numpy.ndarray
        The filtered result. An ndarray with shape: `(6)`.
    buffer: numpy.ndarray
        An ndarray with shape: `(6, 6)`. Need to recursively updated.
    """
    if not isinstance(_W_ati_unfiltered, np.ndarray):
        _W_ati_unfiltered = np.array(_W_ati_unfiltered)
    if not isinstance(buffer, np.ndarray):
        buffer = np.array(buffer)

    assert _W_ati_unfiltered.shape == (6,), \
        '\'_W_ati_unfiltered\' should be size of (6,), but got size of {}'.format(_W_ati_unfiltered.shape)
    assert buffer.shape == (6, 6), \
        '\'buffer\' should be size of (6, 6), but got size of {}'.format(buffer.shape)

    _W_ati_filtered = np.zeros(6)
    for i in range(6):
        buffer[2, i] = buffer[1, i]
        buffer[1, i] = buffer[0, i]
        buffer[0, i] = _W_ati_unfiltered[i]
        _W_ati_filtered[i] = (buffer[2, i] + buffer[1, i] + buffer[0, i])/3.0

    return _W_ati_filtered, buffer