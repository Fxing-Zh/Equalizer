# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 14:14:48 2022

@author: heng.zhang

"""

import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, sqrt, tau, pi


# IIRFilter

class IIRFilter:
    
    def __init__(self, order: int) -> None:
        self.order = order
        # a_{0} ... a_{k}
        self.a_coeffs = [1.0] + [0.0] * order
        # b_{0} ... b_{k}
        self.b_coeffs = [1.0] + [0.0] * order
        
        # x[n-1] ... x[n-k]
        self.input_history = [0.0] * order
        # y[n-1] ... y[n-k]
        self.output_history = [0.0] * order

    def set_coefficients(self, a_coeffs: list[float], b_coeffs: list[float]) -> None:
        if len(a_coeffs) < self.order:
            a_coeffs = [1.0] + a_coeffs
        if len(a_coeffs) != self.order + 1:
            raise ValueError(
                f"预期 a_coeffs to 有 {self.order + 1} elements for {self.order}"
                f"-order filter, got {len(a_coeffs)}"
            )
        if len(b_coeffs) != self.order + 1:
            raise ValueError(
                f"Expected b_coeffs to have {self.order + 1} elements for {self.order}"
                f"-order filter, got {len(a_coeffs)}"
            )
        self.a_coeffs = a_coeffs
        self.b_coeffs = b_coeffs

    def process(self, sample: float) -> float:
        
        result = 0.0
        # 从索引 1 开始，最后执行索引 0
        for i in range(1, self.order + 1):
            result += (self.b_coeffs[i] * self.input_history[i-1]-self.a_coeffs[i] * self.output_history[i-1])
        result = (result + self.b_coeffs[0] * sample) / self.a_coeffs[0]
        self.input_history[1:] = self.input_history[:-1]
        self.output_history[1:] = self.output_history[:-1]
        self.input_history[0] = sample
        self.output_history[0] = result
        return result

# samples

class FilterType:

    def init_buffer(filt: IIRFilter):
        
        # x[n-1] ... x[n-k]
        filt.input_history = [0.0] * len(filt.input_history)
        # y[n-1] ... y[n-k]
        filt.output_history = [0.0] * len(filt.output_history)
    
    # LPF:    H(s) = 1 / (s^2 + s/Q + 1)
    def make_lowpass(
        filt: IIRFilter, frequency: int, samplerate: int, q_factor: float = 1 / sqrt(2)
    ) -> IIRFilter:
    
        w0 = tau * frequency / samplerate
        _sin = sin(w0)
        _cos = cos(w0)
        alpha = _sin / (2 * q_factor)
    
        b0 = (1 - _cos) / 2
        b1 = 1 - _cos
    
        a0 = 1 + alpha
        a1 = -2 * _cos
        a2 = 1 - alpha
    
        # filt = IIRFilter(2)
        filt.set_coefficients([a0, a1, a2], [b0, b1, b0])
        # return filt
    
    # HPF:   H(s) = s^2 / (s^2 + s/Q + 1)
    def make_highpass(
        filt: IIRFilter, frequency: int, samplerate: int, q_factor: float = 1 / sqrt(2)
    ) -> IIRFilter:
    
        w0 = tau * frequency / samplerate
        _sin = sin(w0)
        _cos = cos(w0)
        alpha = _sin / (2 * q_factor)
    
        b0 = (1 + _cos) / 2
        b1 = -1 - _cos
    
        a0 = 1 + alpha
        a1 = -2 * _cos
        a2 = 1 - alpha
    
        # filt = IIRFilter(2)
        filt.set_coefficients([a0, a1, a2], [b0, b1, b0])
        # return filt
    
    # BPF:   H(s) = (s/Q) / (s^2 + s/Q + 1)
    def make_bandpass(
        filt: IIRFilter, frequency: int, samplerate: int, q_factor: float = 1 / sqrt(2)
    ) -> IIRFilter:
    
        w0 = tau * frequency / samplerate
        _sin = sin(w0)
        _cos = cos(w0)
        alpha = _sin / (2 * q_factor)
    
        b0 = alpha  #_sin / 2
        b1 = 0
        b2 = -b0
    
        a0 = 1 + alpha
        a1 = -2 * _cos
        a2 = 1 - alpha
    
        # filt = IIRFilter(2)
        filt.set_coefficients([a0, a1, a2], [b0, b1, b2])
        # return filt
    
    # notch:      H(s) = (s^2 + 1) / (s^2 + s/Q + 1)
    def make_notch(
        filt: IIRFilter, frequency: int, samplerate: int, q_factor: float = 1 / sqrt(2)
    ) -> IIRFilter:
        
        w0 = tau * frequency / samplerate
        _sin = sin(w0)
        _cos = cos(w0)
        alpha = _sin / (2 * q_factor)
        
        b0 =   1
        b1 =  -2 * _cos
        b2 =   1
        
        a0 =   1 + alpha
        a1 =  -2 * _cos
        a2 =   1 - alpha
        
        # filt = IIRFilter(2)
        filt.set_coefficients([a0, a1, a2], [b0, b1, b2])
        # return filt
        
    # APF:   H(s) = (s^2 - s/Q + 1) / (s^2 + s/Q + 1)
    def make_allpass(
        filt: IIRFilter, frequency: int, samplerate: int, q_factor: float = 1 / sqrt(2)
    ) -> IIRFilter:
    
        w0 = tau * frequency / samplerate
        _sin = sin(w0)
        _cos = cos(w0)
        alpha = _sin / (2 * q_factor)
    
        b0 = 1 - alpha
        b1 = -2 * _cos
        b2 = 1 + alpha
    
        # filt = IIRFilter(2)
        filt.set_coefficients([b2, b1, b0], [b0, b1, b2])
        # return filt
    
    # peakingEQ:  H(s) = (s^2 + s*(A/Q) + 1) / (s^2 + s/(A*Q) + 1)
    def make_peak(
        filt: IIRFilter, frequency: int, samplerate: int, gain_db: float, q_factor: float = 1 / sqrt(2)
    ) -> IIRFilter:
    
        w0 = tau * frequency / samplerate
        _sin = sin(w0)
        _cos = cos(w0)
        alpha = _sin / (2 * q_factor)
        big_a = 10 ** (gain_db / 40)
    
        b0 = 1 + alpha * big_a
        b1 = -2 * _cos
        b2 = 1 - alpha * big_a
        a0 = 1 + alpha / big_a
        a1 = -2 * _cos
        a2 = 1 - alpha / big_a
    
        # filt = IIRFilter(2)
        # filt.set_coefficients([round(a0, 6), round(a1, 6), round(a2, 6)], [round(b0, 6), round(b1, 6), round(b2, 6)])
        filt.set_coefficients([a0, a1, a2], [b0, b1, b2])
        # return filt
    
    # lowShelf: H(s) = A * (s^2 + (sqrt(A)/Q)*s + A)/(A*s^2 + (sqrt(A)/Q)*s + 1)
    def make_lowshelf(
        filt: IIRFilter, frequency: int, samplerate: int, gain_db: float, q_factor: float = 1 / sqrt(2)
    ) -> IIRFilter:
    
        w0 = tau * frequency / samplerate
        _sin = sin(w0)
        _cos = cos(w0)
        alpha = _sin / (2 * q_factor)
        big_a = 10 ** (gain_db / 40)
        pmc = (big_a + 1) - (big_a - 1) * _cos
        ppmc = (big_a + 1) + (big_a - 1) * _cos
        mpc = (big_a - 1) - (big_a + 1) * _cos
        pmpc = (big_a - 1) + (big_a + 1) * _cos
        aa2 = 2 * sqrt(big_a) * alpha
    
        b0 = big_a * (pmc + aa2)
        b1 = 2 * big_a * mpc
        b2 = big_a * (pmc - aa2)
        a0 = ppmc + aa2
        a1 = -2 * pmpc
        a2 = ppmc - aa2
    
        # filt = IIRFilter(2)
        filt.set_coefficients([a0, a1, a2], [b0, b1, b2])
        # return filt
    
    # highShelf: H(s) = A * (A*s^2 + (sqrt(A)/Q)*s + 1)/(s^2 + (sqrt(A)/Q)*s + A)
    def make_highshelf(
        filt: IIRFilter, frequency: int, samplerate: int, gain_db: float, q_factor: float = 1 / sqrt(2)
    ) -> IIRFilter:
    
        w0 = tau * frequency / samplerate
        _sin = sin(w0)
        _cos = cos(w0)
        alpha = _sin / (2 * q_factor)
        big_a = 10 ** (gain_db / 40)
        pmc = (big_a + 1) - (big_a - 1) * _cos
        ppmc = (big_a + 1) + (big_a - 1) * _cos
        mpc = (big_a - 1) - (big_a + 1) * _cos
        pmpc = (big_a - 1) + (big_a + 1) * _cos
        aa2 = 2 * sqrt(big_a) * alpha
    
        b0 = big_a * (ppmc + aa2)
        b1 = -2 * big_a * pmpc
        b2 = big_a * (ppmc - aa2)
        a0 = pmc + aa2
        a1 = 2 * mpc
        a2 = pmc - aa2
    
        # filt = IIRFilter(2)
        filt.set_coefficients([a0, a1, a2], [b0, b1, b2])
        # return filt

def get_bounds(
    fft_results: np.ndarray, samplerate: int
) -> tuple[int and float, int and float]:
    lowest = max([-30, np.min(fft_results[1: samplerate // 2 - 1])])
    highest = min([30, np.max(fft_results[1: samplerate // 2 - 1])])
    return lowest, highest


def show_frequency_response(filter: IIRFilter, samplerate: int) -> None:

    size = 512
    outputs = [1] + [0] * (size - 1)
    for filt in filter:
        outputs = [filt.process(item) for item in outputs]
    # outputs = [filter.process(item) for item in inputs]

    filler = [0] * (samplerate - size)  # zero-padding
    outputs += filler
    fft_out = np.abs(np.fft.fft(outputs))
    fft_db = 20 * np.log10(fft_out)

    # Frequencies on log scale from 24 to nyquist frequency
    plt.xlim(24, samplerate / 2 - 1)
    plt.xlabel("Frequency (Hz)")
    plt.xscale("log")

    # Display within reasonable bounds
    bounds = get_bounds(fft_db, samplerate)
    plt.ylim(max([-80, bounds[0]]), min([80, bounds[1]]))
    plt.ylabel("Gain (dB)")

    plt.plot(fft_db)
    plt.show()


def show_phase_response(filter: IIRFilter, samplerate: int) -> None:
    size = 512
    outputs = [1] + [0] * (size - 1)
    for filt in filter:
        outputs = [filt.process(item) for item in outputs]
    # outputs = [filter.process(item) for item in inputs]

    filler = [0] * (samplerate - size)
    outputs += filler
    fft_out = np.angle(np.fft.fft(outputs))
    plt.xlim(24, samplerate / 2 - 1)
    plt.xlabel("Frequency (Hz)")
    plt.xscale("log")

    plt.ylim(-pi, pi)
    plt.ylabel("Phase shift (Radians)")
    plt.plot(np.unwrap(fft_out, -2 * pi))
    plt.show()

if __name__ == '__main__':

    filter1 = IIRFilter(2)
    filter2 = IIRFilter(2)    

    FilterType.make_lowshelf(filter1, 2056, 48000, -14, 0.95)
    FilterType.make_lowshelf(filter2, 250, 48000, -4, 1.44)
    filt = [filter1, filter2]
    show_frequency_response(filt, 48000)
    show_phase_response(filt, 48000)
    
    