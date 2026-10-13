// Just a small C++ experiment, not the full solver.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    const double dx = 1.0 / n, diff = 0.01, dt = 0.0001;
    const double pi = std::acos(-1.0);
    std::vector<double> phi(n), next(n);
    for (int i = 0; i < n; ++i)
        phi[i] = std::sin(2 * pi * (i + 0.5) * dx);
    for (int i = 0; i < n; ++i) {
        int l = (i + n - 1) % n, r = (i + 1) % n;
        next[i] = phi[i] + dt * diff * (phi[r] - 2 * phi[i] + phi[l]) / (dx * dx);
    }
    std::cout << std::setprecision(16);
    for (auto v : next) std::cout << v << '\n';
}
