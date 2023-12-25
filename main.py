import numpy as np
import matplotlib.pyplot as plt


def remove_cosmic_ray(spectra: np.ndarray, threshold: float):
    mean = spectra.mean(axis=0)
    std = spectra.std(axis=0)
    deviation = (spectra - mean[np.newaxis, :]) / std[np.newaxis, :]
    print('spikes detected: ', np.sum(deviation > threshold))
    mask = np.where(deviation > threshold, 0, 1)
    spectra_removed = spectra * mask
    spectra_average = spectra_removed.sum(axis=0)[np.newaxis, :] / mask.sum(axis=0)[np.newaxis, :] * (1 - mask)
    return spectra_removed + spectra_average


def generate_noise(level: float, size: int):
    return np.random.normal(0, level, size)


def Lorentzian(x: np.ndarray, x0: float, gamma: float, A: float):
    return A * gamma ** 2 / ((x - x0) ** 2 + gamma ** 2)


def generate_spike(size: int, probability: float, level: float):
    spikes = np.zeros(size)
    random_indices = np.random.random(size=size)
    print('number of spikes: ', np.sum(random_indices > (1 - probability)))
    spikes[random_indices > (1 - probability)] = 1
    return spikes * level


def generate_test_data(size: int, n: int, level_noise: float = 0.1, level_spike: float = 1, probability_spike: float = 0.003):
    spectra = np.zeros((n, size))
    for i in range(n):
        spectra[i] += generate_spike(size, probability_spike, level_spike)
        spectra[i] += generate_noise(level_noise, size)
    return spectra


def generate_test_data_from_baseline(baseline: np.ndarray, n: int, level_noise: float = 0.1, level_spike: float = 1, probability_spike: float = 0.003):
    return baseline[np.newaxis, :] + generate_test_data(baseline.shape[0], n, level_noise, level_spike, probability_spike)


def main():
    size = 1024
    accumulation = 6

    x = np.arange(size)
    y = Lorentzian(x, 512, 10, 3)

    spectra = generate_test_data_from_baseline(y, accumulation)
    spectra_removed = remove_cosmic_ray(spectra, 2.1)

    print(y.mean())
    print(spectra.mean())
    print(spectra_removed.mean())

    plt.plot(spectra.mean(axis=0), linewidth=1, color='black', label='Original')
    plt.plot(spectra_removed.mean(axis=0), linewidth=0.6, alpha=0.9, color='red', label='CRR')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
