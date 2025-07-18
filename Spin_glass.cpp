#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <map>
#include <bitset>
#include <iomanip>
#include <numeric>
#include <fstream>

using namespace std;

const int ROWS = 2;
const int COLS = 5;
std::string NAME = "2x5_1.csv";

using Matrix = array<array<int, COLS>, ROWS>;

// Create base matrix with customizable entries
Matrix create_base_matrix() {
    return Matrix{{
        {1, 1, 1, 1, 1},  // top row with a 0 in the middle
        {1, 1, 1, 1, 1}   // bottom row all 1s
    }};
}

// Generate all possible spin configurations by flipping 1s to -1s
vector<Matrix> generate_all_configurations(const Matrix& base) {
    vector<Matrix> configs;
    vector<pair<int, int>> one_positions;

    // Find all positions with 1
    for (int i = 0; i < ROWS; ++i)
        for (int j = 0; j < COLS; ++j)
            if (base[i][j] == 1)
                one_positions.emplace_back(i, j);

    int num_ones = one_positions.size();
    int total_configs = 1 << num_ones; // 2^num_ones

    for (int mask = 0; mask < total_configs; ++mask) {
        Matrix mat = base;
        for (int k = 0; k < num_ones; ++k) {
            int flip = (mask >> k) & 1;
            if (flip == 1)
                mat[one_positions[k].first][one_positions[k].second] = -1;
        }
        configs.push_back(mat);
    }
    return configs;
}

// Compute energy-like value (S) for a configuration
int compute_energy(const Matrix& mat) {
    int S = 0;
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            if (i == 0 && j < COLS - 1) {
                S += mat[i][j] * mat[i][j + 1];
                S += mat[i][j] * mat[i + 1][j];
                S += mat[i][j] * mat[i + 1][j + 1];
                S += mat[i + 1][j] * mat[i + 1][j + 1];
            } else if (i == 0 && j == COLS - 1) {
                S += mat[i][j] * mat[i + 1][j];
            }
        }
    }
    return S;
}

int main() {
    Matrix base = create_base_matrix();
    vector<Matrix> configs = generate_all_configurations(base);
    cout << "Total possible matrices: " << configs.size() << "\n\n";

    // Save individual configurations and their energies
    ofstream csv_file("ising_configurations.csv");
    csv_file << "Config,Energy\n";

    for (const auto& mat : configs) {
        int energy = compute_energy(mat);
        csv_file << "\"";
        for (const auto& row : mat)
            for (int val : row)
                csv_file << val << " ";
        csv_file << "\"," << energy << "\n";
    }
    csv_file.close();

    // Compute energy counts (degeneracy)
    vector<int> energies;
    map<int, int> counts;

    for (const auto& config : configs) {
        int energy = compute_energy(config);
        energies.push_back(energy);
        counts[energy]++;
    }

    int max_count_energy = energies[0];
    int max_count = 0;

    for (const auto& [energy, count] : counts) {
        if (count > max_count) {
            max_count = count;
            max_count_energy = energy;
        }
    }

    cout << "The most recurring value is " << max_count_energy << ", and it appears " << max_count << " times.\n";

    cout << "\nEnergy | Degeneracy\n";
    cout << "-------------------\n";
    for (const auto& [energy, count] : counts) {
        cout << setw(6) << energy << " | " << count << "\n";
    }

    // Save energy-degeneracy info to a second CSV
    ofstream deg_file(NAME);
    for (const auto& [energy, count] : counts) {
        deg_file << energy << "," << count << "\n";
    }
    deg_file.close();

    return 0;
}
