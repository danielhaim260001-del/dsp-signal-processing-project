\\final project 
\\Q1
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = "/mnt/data/201282401-proj_data.txt"
data = np.loadtxt(file_path)

# Define the sampling frequency
sampling_frequency = 60000  # Hz
time = np.arange(len(data)) / sampling_frequency  # Time in seconds

# Create the plot with English labels and blue color
plt.figure(figsize=(12, 6))
plt.plot(time, data, color='blue', label="Data")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Time Domain Representation of Data")
plt.legend()
plt.grid()

# Show the plot
plt.show()

\\Q2
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = "/mnt/data/201282401-proj_data.txt"
data = np.loadtxt(file_path)

# Define the sampling frequency
sampling_frequency = 60000  # Hz
time = np.arange(len(data)) / sampling_frequency  # Time in seconds

# Define the zoom range (e.g., first 1000 samples for better clarity)
zoom_samples = 1000
zoom_time = time[:zoom_samples]
zoom_data = data[:zoom_samples]

# Create the zoomed plot
plt.figure(figsize=(12, 6))
plt.plot(zoom_time, zoom_data, color='blue', label="Zoomed Data")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Zoomed Time Domain Representation of Data")
plt.legend()
plt.grid()

# Show the plot
plt.show()

\\Q3
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# Load the data
file_path = "/mnt/data/201282401-proj_data.txt"
data = np.loadtxt(file_path)

# Define the sampling frequency
sampling_frequency = 60000  # Hz
N = len(data)  # Number of samples
time = np.arange(len(data)) / sampling_frequency  # Time in seconds

# Compute the DFT of the signal
dft = fft(data)
frequencies_hz = fftfreq(N, d=1 / sampling_frequency)  # Frequency axis in Hz

# Magnitude for linear and dB scales
magnitude_linear = np.abs(dft)
magnitude_db = 20 * np.log10(magnitude_linear + 1e-12)  # Avoid log(0)

# Frequencies for normalized frequency (0 to 2*pi)
frequencies_norm = 2 * np.pi * frequencies_hz / sampling_frequency

# Plot 3.1 - Linear magnitude scale and frequency axis in Hz
plt.figure(figsize=(12, 6))
plt.plot(frequencies_hz[:N // 2], magnitude_linear[:N // 2], color='blue', label="Linear Magnitude")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("DFT - Linear Magnitude Scale and Frequency in Hz")
plt.grid()
plt.legend()
plt.show()

# Plot 3.2 - dB magnitude scale and frequency axis in Hz
plt.figure(figsize=(12, 6))
plt.plot(frequencies_hz[:N // 2], magnitude_db[:N // 2], color='green', label="Magnitude (dB)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("DFT - dB Magnitude Scale and Frequency in Hz")
plt.grid()
plt.legend()
plt.show()

# Plot 3.3 - dB magnitude scale and normalized frequency (0 to 2π rad/sample)
plt.figure(figsize=(12, 6))
plt.plot(frequencies_norm[:N // 2], magnitude_db[:N // 2], color='red', label="Magnitude (dB)")
plt.xlabel("Normalized Frequency (rad/sample)")
plt.ylabel("Magnitude (dB)")
plt.title("DFT - dB Magnitude Scale and Normalized Frequency (0 to 2π rad/sample)")
plt.grid()
plt.legend()
plt.show()

\\Q5
import numpy as np
import matplotlib.pyplot as plt

# פונקציה לחישוב תגובת התדר
def H_omega(omega):
    """
    חישוב תגובת התדר H(e^(j*omega)).
    omega: תדר זוויתי מנורמל (ברדיאנים לדגימה).
    """
    z = np.exp(1j * omega)  # הצבה z = e^(j*omega)
    numerator = (z - 1.1 * np.exp(1j * 0.01)) * (z - 1.1 * np.exp(-1j * 0.01))
    denominator = (z - 0.5)**2
    return numerator / denominator

# תחום התדרים (0 עד 2*pi)
omega = np.linspace(0, 2 * np.pi, 1000)

# חישוב המגניטודה
H_vals = H_omega(omega)
magnitude = np.abs(H_vals)

# גרף תגובת התדר עם מגניטודה לינארית ותדר מנורמל
plt.figure(figsize=(12, 6))
plt.plot(omega, magnitude, color='blue', label="Magnitude (Linear)")
plt.xlabel("Normalized Frequency (rad/sample)")
plt.ylabel("Magnitude")
plt.title("Frequency Response - Linear Magnitude and Normalized Frequency")
plt.grid()
plt.legend()
plt.show()


\\Q7
import numpy as np
import matplotlib.pyplot as plt

# Define poles and zeros for H(z)
zeros_H = [1.1 * np.exp(1j * 0.01), 1.1 * np.exp(-1j * 0.01)]
poles_H = [0.5, 0.5]

# Define poles and zeros for Hmin(z)
zeros_Hmin = [1 / 1.1 * np.exp(-1j * 0.01), 1 / 1.1 * np.exp(1j * 0.01)]
poles_Hmin = [0.5, 0.5]

# Define poles and zeros for Hap(z)
zeros_Hap = zeros_H
poles_Hap = zeros_Hmin

# Function to plot pole-zero map
def plot_pz_map(zeros, poles, title):
    plt.figure(figsize=(6, 6))
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.grid(True, linestyle='--', linewidth=0.5)

    # Unit circle
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='dashed')
    plt.gca().add_patch(circle)

    # Plot zeros and poles
    plt.scatter(np.real(zeros), np.imag(zeros), marker='o', facecolors='none', edgecolors='b', label="Zeros")
    plt.scatter(np.real(poles), np.imag(poles), marker='x', color='r', label="Poles")

    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.show()

# Plot pole-zero maps
plot_pz_map(zeros_H, poles_H, "Pole-Zero Map of H(z)")
plot_pz_map(zeros_Hmin, poles_Hmin, "Pole-Zero Map of Hmin(z)")
plot_pz_map(zeros_Hap, poles_Hap, "Pole-Zero Map of Hap(z)")


\\Q8
import numpy as np
import matplotlib.pyplot as plt

# פונקציה לחישוב תגובת התדר
def compute_frequency_response(zeros, poles, omega):
    """
    חישוב תגובת התדר H(e^(j*omega)) עבור קבוצת אפסים וקטבים נתונה.
    """
    z = np.exp(1j * omega)  # e^(j*omega)
    
    # חישוב המונה והמכנה
    numerator = np.ones_like(z, dtype=complex)
    denominator = np.ones_like(z, dtype=complex)
    
    for zero in zeros:
        numerator *= (z - zero)
    for pole in poles:
        denominator *= (z - pole)
    
    return numerator / denominator

# הגדרת תחום התדרים (0 עד 2*pi)
omega = np.linspace(0, 2 * np.pi, 1000)

# הגדרת האפסים והקטבים עבור Hap(z) ו- Hmin(z)
zeros_Hap = [1.1 * np.exp(1j * 0.01), 1.1 * np.exp(-1j * 0.01)]
poles_Hap = [1 / 1.1 * np.exp(-1j * 0.01), 1 / 1.1 * np.exp(1j * 0.01)]

zeros_Hmin = [1 / 1.1 * np.exp(-1j * 0.01), 1 / 1.1 * np.exp(1j * 0.01)]
poles_Hmin = [0.5, 0.5]

# חישוב תגובת התדר עבור Hap(z) ו- Hmin(z)
Hap_vals = compute_frequency_response(zeros_Hap, poles_Hap, omega)
Hmin_vals = compute_frequency_response(zeros_Hmin, poles_Hmin, omega)

# חישוב המגניטודה של התגובות
magnitude_Hap = np.abs(Hap_vals)
magnitude_Hmin = np.abs(Hmin_vals)

# גרף תגובת התדר עבור Hap(z)
plt.figure(figsize=(12, 6))
plt.plot(omega, magnitude_Hap, color='blue', label="Magnitude of Hap(z)")
plt.xlabel("Normalized Frequency (rad/sample)")
plt.ylabel("Magnitude")
plt.title("Frequency Response of Hap(z) - Linear Magnitude")
plt.grid()
plt.legend()
plt.show()

# גרף תגובת התדר עבור Hmin(z)
plt.figure(figsize=(12, 6))
plt.plot(omega, magnitude_Hmin, color='red', label="Magnitude of Hmin(z)")
plt.xlabel("Normalized Frequency (rad/sample)")
plt.ylabel("Magnitude")
plt.title("Frequency Response of Hmin(z) - Linear Magnitude")
plt.grid()
plt.legend()
plt.show()


\\Q10,11
import scipy.signal as signal

# Define the coefficients of the correction system difference equation
b = np.array([1, -1, 0.25])  # Coefficients for x[n], x[n-1], x[n-2]
a = np.array([1, -2.1999, 1.21])  # Coefficients for y[n], y[n-1], y[n-2]

# Apply the correction system (digital filter) to the input data
corrected_signal = signal.lfilter(b, a, data)

# Plot the corrected signal
plt.figure(figsize=(12, 6))
plt.plot(time, corrected_signal, label="Corrected Signal", color='blue')
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Signal After Passing Through the Correction System")
plt.legend()
plt.grid()
plt.show()


\\Q12
import numpy as np
import matplotlib.pyplot as plt

# Load the signal data from the provided file
file_path = '/mnt/data/201282401-proj_data.txt'
signal = np.loadtxt(file_path)

# Updated parameters
Fs = 60000  # Sampling frequency in Hz
N = len(signal)  # Number of samples

# Compute the FFT
freq = np.fft.fftfreq(N, 1 / Fs)  # Frequency axis
fft_values = np.fft.fft(signal)

# Compute magnitude
linear_magnitude = np.abs(fft_values)
db_magnitude = 20 * np.log10(linear_magnitude + 1e-12)  # Adding small value to avoid log(0)

# Plotting
# 12.1 Linear magnitude scale
plt.figure(figsize=(12, 6))
plt.plot(freq[:N // 2], linear_magnitude[:N // 2])  # Only plot positive frequencies
plt.title("DFT with Linear Magnitude Scale")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

# 12.2 dB magnitude scale
plt.figure(figsize=(12, 6))
plt.plot(freq[:N // 2], db_magnitude[:N // 2])  # Only plot positive frequencies
plt.title("DFT with dB Magnitude Scale")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.show()

\\Q16
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

# Filter specifications
M = 200  # Filter order
fc = 3 * 4000  # Cutoff frequency (3 times the lowest frequency in Hz)
Fs = 60000  # Sampling frequency
num_taps = M + 1  # Number of filter coefficients

# Normalize cutoff frequency
fc_normalized = fc / (Fs / 2)  # Normalize with Nyquist frequency

# Design the LPF using a Bartlett window
h = firwin(num_taps, fc_normalized, window='bartlett')

# Compute frequency response for normalized frequency (0 to 2π rad/sample)
w, h_freq = freqz(h, worN=8000)

# Plot the frequency response with normalized frequency
plt.figure(figsize=(12, 6))
plt.plot(w, np.abs(h_freq))  # Linear magnitude scale
plt.title("Frequency Response of the 200th Order LPF (Linear Magnitude)")
plt.xlabel("Normalized Frequency (rad/sample)")
plt.ylabel("Magnitude")
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'])
plt.grid(True)
plt.show()

\\Q18+19
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, firwin, freqz

# Load the signal data from the provided file
file_path = '/mnt/data/201282401-proj_data.txt'
signal = np.loadtxt(file_path)

# Filter specifications
M = 200  # Filter order
fc = 3 * 4000  # Cutoff frequency (3 times the lowest frequency in Hz)
Fs = 60000  # Sampling frequency
num_taps = M + 1  # Number of filter coefficients

# Normalize cutoff frequency
fc_normalized = fc / (Fs / 2)  # Normalize with Nyquist frequency

# Design the LPF using a Bartlett window
h = firwin(num_taps, fc_normalized, window='bartlett')

# Apply the FIR filter to the input signal
filtered_signal = lfilter(h, 1.0, signal)

# Convert sample indices to time in seconds
time = np.arange(len(filtered_signal)) / Fs  # Time array based on sampling frequency

# Plot the filtered signal in the time domain
plt.figure(figsize=(10, 8))
plt.plot(time, filtered_signal, label="Filtered Signal", color="blue")
plt.title("Filtered Signal in Time Domain")
plt.xlabel("t [sec]")
plt.ylabel("signal")
plt.grid(True)
plt.tight_layout()
plt.show()


\\Q20
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, firwin, freqz

# Load the signal data from the provided file
file_path = '/mnt/data/201282401-proj_data.txt'
signal = np.loadtxt(file_path)

# Filter specifications
M = 200  # Filter order
fc = 3 * 4000  # Cutoff frequency (3 times the lowest frequency in Hz)
Fs = 60000  # Sampling frequency
num_taps = M + 1  # Number of filter coefficients

# Normalize cutoff frequency
fc_normalized = fc / (Fs / 2)  # Normalize with Nyquist frequency

# Design the LPF using a Bartlett window
h = firwin(num_taps, fc_normalized, window='bartlett')

# Apply the FIR filter to the input signal
filtered_signal = lfilter(h, 1.0, signal)

# Compute the DFT of the filtered output signal
filtered_signal_fft = np.fft.fft(filtered_signal)
filtered_signal_magnitude = np.abs(filtered_signal_fft)
freq_axis = np.fft.fftfreq(len(filtered_signal), 1 / Fs)  # Frequency axis in Hz

# Plot the DFT with linear magnitude scale
plt.figure(figsize=(12, 6))
plt.plot(freq_axis[:len(filtered_signal)//2], filtered_signal_magnitude[:len(filtered_signal)//2])
plt.title("DFT of Filtered Output Signal (Linear Magnitude)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()






