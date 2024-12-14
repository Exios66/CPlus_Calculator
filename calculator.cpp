#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
#include <limits>
#include <vector>
#include <stack>
#include <algorithm>
#include <numeric>
#include <map>

class ScientificCalculator {
private:
    // Constants
    const double PI = 3.14159265358979323846;
    const double E = 2.71828182845904523536;
    
    // Helper functions
    bool isNumber(const std::string& str) {
        try {
            std::stod(str);
            return true;
        } catch (...) {
            return false;
        }
    }

    double degToRad(double degrees) {
        return degrees * PI / 180.0;
    }

    double radToDeg(double radians) {
        return radians * 180.0 / PI;
    }

    // Statistical Helper Functions
    double calculateMean(const std::vector<double>& data) {
        if (data.empty()) throw std::runtime_error("Cannot calculate mean of empty dataset!");
        return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    }

    double calculateVariance(const std::vector<double>& data, bool sample = true) {
        if (data.size() < (sample ? 2 : 1)) {
            throw std::runtime_error("Insufficient data for variance calculation!");
        }
        double mean = calculateMean(data);
        double sum = 0.0;
        for (double value : data) {
            sum += pow(value - mean, 2);
        }
        return sum / (data.size() - (sample ? 1 : 0));
    }

    double calculateStandardDeviation(const std::vector<double>& data, bool sample = true) {
        return sqrt(calculateVariance(data, sample));
    }

    double calculateZScore(double value, double mean, double stdDev) {
        if (stdDev == 0) throw std::runtime_error("Standard deviation cannot be zero!");
        return (value - mean) / stdDev;
    }

public:
    // Basic Operations
    double add(double a, double b) { return a + b; }
    double subtract(double a, double b) { return a - b; }
    double multiply(double a, double b) { return a * b; }
    double divide(double a, double b) {
        if (b == 0) throw std::runtime_error("Division by zero!");
        return a / b;
    }

    // Power and Root Operations
    double power(double base, double exponent) { return pow(base, exponent); }
    double squareRoot(double n) {
        if (n < 0) throw std::runtime_error("Cannot calculate square root of negative number!");
        return sqrt(n);
    }
    double cubeRoot(double n) { return cbrt(n); }

    // Trigonometric Functions (in degrees)
    double sine(double angle) { return sin(degToRad(angle)); }
    double cosine(double angle) { return cos(degToRad(angle)); }
    double tangent(double angle) { 
        if (fmod(angle + 90.0, 180.0) == 0) throw std::runtime_error("Tangent undefined at this angle!");
        return tan(degToRad(angle)); 
    }

    // Inverse Trigonometric Functions
    double arcSine(double value) {
        if (value < -1 || value > 1) throw std::runtime_error("Arc sine input must be between -1 and 1!");
        return radToDeg(asin(value));
    }
    double arcCosine(double value) {
        if (value < -1 || value > 1) throw std::runtime_error("Arc cosine input must be between -1 and 1!");
        return radToDeg(acos(value));
    }
    double arcTangent(double value) { return radToDeg(atan(value)); }

    // Logarithmic Functions
    double naturalLog(double n) {
        if (n <= 0) throw std::runtime_error("Cannot calculate logarithm of non-positive number!");
        return log(n);
    }
    double log10(double n) {
        if (n <= 0) throw std::runtime_error("Cannot calculate logarithm of non-positive number!");
        return std::log10(n);
    }

    // Exponential Function
    double exp(double n) { return std::exp(n); }

    // Additional Functions
    double absolute(double n) { return std::abs(n); }
    double factorial(double n) {
        if (n < 0 || floor(n) != n) throw std::runtime_error("Factorial only defined for non-negative integers!");
        if (n == 0 || n == 1) return 1;
        return n * factorial(n - 1);
    }

    // Statistical Functions
    std::vector<double> descriptiveStats(const std::vector<double>& data) {
        if (data.empty()) throw std::runtime_error("Empty dataset!");
        
        std::vector<double> sortedData = data;
        std::sort(sortedData.begin(), sortedData.end());
        
        double mean = calculateMean(data);
        double median = (sortedData.size() % 2 == 0) 
            ? (sortedData[sortedData.size()/2 - 1] + sortedData[sortedData.size()/2]) / 2
            : sortedData[sortedData.size()/2];
        double stdDev = calculateStandardDeviation(data);
        double min = sortedData.front();
        double max = sortedData.back();
        double range = max - min;
        
        // Return vector contains: {mean, median, stdDev, min, max, range}
        return {mean, median, stdDev, min, max, range};
    }

    double correlation(const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size()) throw std::runtime_error("Datasets must be of equal size!");
        if (x.empty()) throw std::runtime_error("Empty datasets!");

        double meanX = calculateMean(x);
        double meanY = calculateMean(y);
        double stdDevX = calculateStandardDeviation(x);
        double stdDevY = calculateStandardDeviation(y);

        double sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            sum += (x[i] - meanX) * (y[i] - meanY);
        }

        return sum / ((x.size() - 1) * stdDevX * stdDevY);
    }

    // Probability Distributions
    double normalPDF(double x, double mean, double stdDev) {
        return (1.0 / (stdDev * sqrt(2.0 * PI))) * 
               exp(-0.5 * pow((x - mean) / stdDev, 2));
    }

    double normalCDF(double x, double mean, double stdDev) {
        return 0.5 * (1.0 + erf((x - mean) / (stdDev * sqrt(2.0))));
    }

    // Psychometric Functions
    double cronbachAlpha(const std::vector<std::vector<double>>& items) {
        if (items.empty() || items[0].empty()) throw std::runtime_error("Empty dataset!");
        
        size_t n = items.size();    // number of items
        size_t k = items[0].size(); // number of respondents
        
        // Calculate variance of sum scores
        std::vector<double> sumScores(k, 0.0);
        for (size_t i = 0; i < n; ++i) {
            if (items[i].size() != k) throw std::runtime_error("Inconsistent number of responses!");
            for (size_t j = 0; j < k; ++j) {
                sumScores[j] += items[i][j];
            }
        }
        double varSum = calculateVariance(sumScores, false);
        
        // Calculate sum of item variances
        double sumVar = 0.0;
        for (const auto& item : items) {
            sumVar += calculateVariance(item, false);
        }
        
        return (n / (n - 1.0)) * (1.0 - (sumVar / varSum));
    }

    double itemDifficulty(const std::vector<double>& responses) {
        if (responses.empty()) throw std::runtime_error("Empty response set!");
        double sum = std::accumulate(responses.begin(), responses.end(), 0.0);
        return sum / responses.size();
    }

    double itemDiscrimination(const std::vector<double>& responses, const std::vector<double>& totalScores) {
        return correlation(responses, totalScores);
    }

    std::vector<double> standardizeScores(const std::vector<double>& rawScores, double desiredMean = 100, double desiredStdDev = 15) {
        double mean = calculateMean(rawScores);
        double stdDev = calculateStandardDeviation(rawScores);
        
        std::vector<double> standardizedScores;
        standardizedScores.reserve(rawScores.size());
        
        for (double score : rawScores) {
            double zScore = calculateZScore(score, mean, stdDev);
            standardizedScores.push_back(zScore * desiredStdDev + desiredMean);
        }
        
        return standardizedScores;
    }
};

void displayMenu() {
    std::cout << "\n=== Scientific Calculator ===" << std::endl;
    std::cout << "Basic Operations:" << std::endl;
    std::cout << "1. Addition (+)" << std::endl;
    std::cout << "2. Subtraction (-)" << std::endl;
    std::cout << "3. Multiplication (*)" << std::endl;
    std::cout << "4. Division (/)" << std::endl;
    std::cout << "\nAdvanced Mathematical Functions:" << std::endl;
    std::cout << "5. Power (^)" << std::endl;
    std::cout << "6. Square Root (√)" << std::endl;
    std::cout << "7. Cube Root (∛)" << std::endl;
    std::cout << "8. Sine (sin)" << std::endl;
    std::cout << "9. Cosine (cos)" << std::endl;
    std::cout << "10. Tangent (tan)" << std::endl;
    std::cout << "11. Arc Sine (asin)" << std::endl;
    std::cout << "12. Arc Cosine (acos)" << std::endl;
    std::cout << "13. Arc Tangent (atan)" << std::endl;
    std::cout << "14. Natural Logarithm (ln)" << std::endl;
    std::cout << "15. Logarithm base 10 (log)" << std::endl;
    std::cout << "16. Exponential (e^x)" << std::endl;
    std::cout << "17. Absolute Value (|x|)" << std::endl;
    std::cout << "18. Factorial (n!)" << std::endl;
    std::cout << "\nStatistical and Psychometric Analysis:" << std::endl;
    std::cout << "19. Descriptive Statistics" << std::endl;
    std::cout << "20. Correlation Analysis" << std::endl;
    std::cout << "21. Normal Distribution (PDF)" << std::endl;
    std::cout << "22. Normal Distribution (CDF)" << std::endl;
    std::cout << "23. Cronbach's Alpha" << std::endl;
    std::cout << "24. Item Analysis" << std::endl;
    std::cout << "25. Score Standardization" << std::endl;
    std::cout << "\n0. Exit" << std::endl;
    std::cout << "Enter your choice: ";
}

void displayStatisticalMenu() {
    std::cout << "\n=== Statistical and Psychometric Analysis ===" << std::endl;
    std::cout << "19. Descriptive Statistics" << std::endl;
    std::cout << "20. Correlation Analysis" << std::endl;
    std::cout << "21. Normal Distribution (PDF)" << std::endl;
    std::cout << "22. Normal Distribution (CDF)" << std::endl;
    std::cout << "23. Cronbach's Alpha" << std::endl;
    std::cout << "24. Item Analysis" << std::endl;
    std::cout << "25. Score Standardization" << std::endl;
}

std::vector<double> getDatasetFromUser(const std::string& prompt) {
    std::vector<double> data;
    std::string input;
    std::cout << prompt << " (Enter one number per line, empty line to finish):" << std::endl;
    
    while (true) {
        std::getline(std::cin, input);
        if (input.empty()) break;
        
        try {
            data.push_back(std::stod(input));
        } catch (...) {
            std::cout << "Invalid input! Please enter a valid number." << std::endl;
        }
    }
    return data;
}

int main() {
    ScientificCalculator calc;
    int choice;
    double num1, num2, result;

    while (true) {
        displayMenu();
        std::cin >> choice;

        if (choice == 0) {
            std::cout << "Thank you for using the Scientific Calculator!" << std::endl;
            break;
        }

        try {
            switch (choice) {
                case 1: // Addition
                    std::cout << "Enter first number: ";
                    std::cin >> num1;
                    std::cout << "Enter second number: ";
                    std::cin >> num2;
                    result = calc.add(num1, num2);
                    break;

                case 2: // Subtraction
                    std::cout << "Enter first number: ";
                    std::cin >> num1;
                    std::cout << "Enter second number: ";
                    std::cin >> num2;
                    result = calc.subtract(num1, num2);
                    break;

                case 3: // Multiplication
                    std::cout << "Enter first number: ";
                    std::cin >> num1;
                    std::cout << "Enter second number: ";
                    std::cin >> num2;
                    result = calc.multiply(num1, num2);
                    break;

                case 4: // Division
                    std::cout << "Enter numerator: ";
                    std::cin >> num1;
                    std::cout << "Enter denominator: ";
                    std::cin >> num2;
                    result = calc.divide(num1, num2);
                    break;

                case 5: // Power
                    std::cout << "Enter base: ";
                    std::cin >> num1;
                    std::cout << "Enter exponent: ";
                    std::cin >> num2;
                    result = calc.power(num1, num2);
                    break;

                case 6: // Square Root
                    std::cout << "Enter number: ";
                    std::cin >> num1;
                    result = calc.squareRoot(num1);
                    break;

                case 7: // Cube Root
                    std::cout << "Enter number: ";
                    std::cin >> num1;
                    result = calc.cubeRoot(num1);
                    break;

                case 8: // Sine
                    std::cout << "Enter angle in degrees: ";
                    std::cin >> num1;
                    result = calc.sine(num1);
                    break;

                case 9: // Cosine
                    std::cout << "Enter angle in degrees: ";
                    std::cin >> num1;
                    result = calc.cosine(num1);
                    break;

                case 10: // Tangent
                    std::cout << "Enter angle in degrees: ";
                    std::cin >> num1;
                    result = calc.tangent(num1);
                    break;

                case 11: // Arc Sine
                    std::cout << "Enter value (-1 to 1): ";
                    std::cin >> num1;
                    result = calc.arcSine(num1);
                    break;

                case 12: // Arc Cosine
                    std::cout << "Enter value (-1 to 1): ";
                    std::cin >> num1;
                    result = calc.arcCosine(num1);
                    break;

                case 13: // Arc Tangent
                    std::cout << "Enter value: ";
                    std::cin >> num1;
                    result = calc.arcTangent(num1);
                    break;

                case 14: // Natural Logarithm
                    std::cout << "Enter number: ";
                    std::cin >> num1;
                    result = calc.naturalLog(num1);
                    break;

                case 15: // Log base 10
                    std::cout << "Enter number: ";
                    std::cin >> num1;
                    result = calc.log10(num1);
                    break;

                case 16: // Exponential
                    std::cout << "Enter power of e: ";
                    std::cin >> num1;
                    result = calc.exp(num1);
                    break;

                case 17: // Absolute Value
                    std::cout << "Enter number: ";
                    std::cin >> num1;
                    result = calc.absolute(num1);
                    break;

                case 18: // Factorial
                    std::cout << "Enter non-negative integer: ";
                    std::cin >> num1;
                    result = calc.factorial(num1);
                    break;

                case 19: { // Descriptive Statistics
                    std::vector<double> data = getDatasetFromUser("Enter dataset");
                    auto stats = calc.descriptiveStats(data);
                    std::cout << "Mean: " << stats[0] << std::endl;
                    std::cout << "Median: " << stats[1] << std::endl;
                    std::cout << "Standard Deviation: " << stats[2] << std::endl;
                    std::cout << "Minimum: " << stats[3] << std::endl;
                    std::cout << "Maximum: " << stats[4] << std::endl;
                    std::cout << "Range: " << stats[5] << std::endl;
                    break;
                }
                case 20: { // Correlation
                    std::vector<double> x = getDatasetFromUser("Enter first dataset (X)");
                    std::vector<double> y = getDatasetFromUser("Enter second dataset (Y)");
                    result = calc.correlation(x, y);
                    break;
                }
                case 21: { // Normal PDF
                    std::cout << "Enter x value: ";
                    std::cin >> num1;
                    std::cout << "Enter mean: ";
                    std::cin >> num2;
                    double stdDev;
                    std::cout << "Enter standard deviation: ";
                    std::cin >> stdDev;
                    result = calc.normalPDF(num1, num2, stdDev);
                    break;
                }
                case 22: { // Normal CDF
                    std::cout << "Enter x value: ";
                    std::cin >> num1;
                    std::cout << "Enter mean: ";
                    std::cin >> num2;
                    double stdDev;
                    std::cout << "Enter standard deviation: ";
                    std::cin >> stdDev;
                    result = calc.normalCDF(num1, num2, stdDev);
                    break;
                }
                case 23: { // Cronbach's Alpha
                    std::cout << "Enter number of items: ";
                    int numItems;
                    std::cin >> numItems;
                    std::vector<std::vector<double>> items;
                    for (int i = 0; i < numItems; ++i) {
                        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                        items.push_back(getDatasetFromUser("Enter responses for item " + std::to_string(i + 1)));
                    }
                    result = calc.cronbachAlpha(items);
                    break;
                }
                case 24: { // Item Analysis
                    std::vector<double> responses = getDatasetFromUser("Enter item responses (0/1)");
                    std::vector<double> totalScores = getDatasetFromUser("Enter total scores");
                    double difficulty = calc.itemDifficulty(responses);
                    double discrimination = calc.itemDiscrimination(responses, totalScores);
                    std::cout << "Item Difficulty: " << difficulty << std::endl;
                    std::cout << "Item Discrimination: " << discrimination << std::endl;
                    result = difficulty; // Just to maintain the result variable
                    break;
                }
                case 25: { // Score Standardization
                    std::vector<double> rawScores = getDatasetFromUser("Enter raw scores");
                    std::cout << "Enter desired mean (default 100): ";
                    double desiredMean;
                    std::cin >> desiredMean;
                    std::cout << "Enter desired standard deviation (default 15): ";
                    double desiredStdDev;
                    std::cin >> desiredStdDev;
                    auto standardizedScores = calc.standardizeScores(rawScores, desiredMean, desiredStdDev);
                    std::cout << "Standardized scores:" << std::endl;
                    for (double score : standardizedScores) {
                        std::cout << score << std::endl;
                    }
                    result = standardizedScores[0]; // Just to maintain the result variable
                    break;
                }

                default:
                    std::cout << "Invalid choice! Please try again." << std::endl;
                    continue;
            }

            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Result: " << result << std::endl;

        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }

        // Clear input buffer
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    return 0;
} 