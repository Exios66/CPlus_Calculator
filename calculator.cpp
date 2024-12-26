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
#include <random>

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

    void validateDataset(const std::vector<double>& data, const std::string& name = "Dataset") {
        if (data.empty()) {
            throw std::runtime_error(name + " cannot be empty!");
        }
    }

    void validateMatrixDimensions(const std::vector<std::vector<double>>& matrix) {
        if (matrix.empty() || matrix[0].empty()) {
            throw std::runtime_error("Matrix cannot be empty!");
        }
        size_t rows = matrix.size();
        size_t cols = matrix[0].size();
        for (const auto& row : matrix) {
            if (row.size() != cols) {
                throw std::runtime_error("Matrix has inconsistent dimensions!");
            }
        }
    }

    void validateProbability(double p, const std::string& name = "Probability") {
        if (p < 0 || p > 1) {
            throw std::runtime_error(name + " must be between 0 and 1!");
        }
    }

    void validatePositive(double value, const std::string& name = "Value") {
        if (value <= 0) {
            throw std::runtime_error(name + " must be positive!");
        }
    }

    void validateNonNegative(double value, const std::string& name = "Value") {
        if (value < 0) {
            throw std::runtime_error(name + " cannot be negative!");
        }
    }

    // Enhanced eigenvalue calculation using QR algorithm
    std::vector<double> calculateEigenvalues(const std::vector<std::vector<double>>& matrix) {
        validateMatrixDimensions(matrix);
        
        // Copy matrix for QR decomposition
        std::vector<std::vector<double>> A = matrix;
        size_t n = A.size();
        std::vector<double> eigenvalues(n);
        
        const int MAX_ITERATIONS = 1000;
        const double TOLERANCE = 1e-10;
        
        for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
            // QR decomposition
            std::vector<std::vector<double>> Q(n, std::vector<double>(n));
            std::vector<std::vector<double>> R(n, std::vector<double>(n));
            
            // Modified Gram-Schmidt process
            for (size_t j = 0; j < n; ++j) {
                std::vector<double> v(n);
                for (size_t i = 0; i < n; ++i) {
                    v[i] = A[i][j];
                }
                
                for (size_t i = 0; i < j; ++i) {
                    double dot = 0;
                    for (size_t k = 0; k < n; ++k) {
                        dot += Q[k][i] * v[k];
                    }
                    R[i][j] = dot;
                    for (size_t k = 0; k < n; ++k) {
                        v[k] -= dot * Q[k][i];
                    }
                }
                
                double norm = 0;
                for (size_t i = 0; i < n; ++i) {
                    norm += v[i] * v[i];
                }
                norm = sqrt(norm);
                
                if (norm > TOLERANCE) {
                    R[j][j] = norm;
                    for (size_t i = 0; i < n; ++i) {
                        Q[i][j] = v[i] / norm;
                    }
                }
            }
            
            // Matrix multiplication A = R * Q
            std::vector<std::vector<double>> newA(n, std::vector<double>(n, 0.0));
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    for (size_t k = 0; k < n; ++k) {
                        newA[i][j] += R[i][k] * Q[k][j];
                    }
                }
            }
            
            // Check convergence
            bool converged = true;
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    if (std::abs(A[i][j] - newA[i][j]) > TOLERANCE) {
                        converged = false;
                        break;
                    }
                }
                if (!converged) break;
            }
            
            A = newA;
            
            if (converged) {
                for (size_t i = 0; i < n; ++i) {
                    eigenvalues[i] = A[i][i];
                }
                break;
            }
        }
        
        std::sort(eigenvalues.begin(), eigenvalues.end(), std::greater<double>());
        return eigenvalues;
    }

    // Enhanced parallel analysis with better random number generation
    std::vector<std::vector<double>> generateRandomData(size_t numVariables, size_t numObservations) {
        std::vector<std::vector<double>> randomData(numVariables, std::vector<double>(numObservations));
        
        // Use hardware random device for better seed
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::normal_distribution<double> dist(0.0, 1.0);
        
        // Generate random data with proper correlation structure
        for (size_t i = 0; i < numVariables; ++i) {
            for (size_t j = 0; j < numObservations; ++j) {
                randomData[i][j] = dist(gen);
            }
        }
        
        return randomData;
    }

    std::vector<std::vector<double>> calculateCorrelationMatrix(const std::vector<std::vector<double>>& data) {
        size_t n = data.size();
        std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i; j < n; ++j) {
                double r = (i == j) ? 1.0 : correlation(data[i], data[j]);
                matrix[i][j] = matrix[j][i] = r;
            }
        }
        
        return matrix;
    }

    // Custom gamma function implementation
    double gamma(double x) {
        if (x <= 0) throw std::runtime_error("Gamma function undefined for non-positive numbers");
        
        // Lanczos approximation
        const double g = 7;
        const double p[] = {
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7
        };
        
        if (x < 0.5) {
            return M_PI / (std::sin(M_PI * x) * gamma(1 - x));
        }
        
        x -= 1;
        double a = p[0];
        for (int i = 1; i < 9; i++) {
            a += p[i] / (x + i);
        }
        
        double t = x + g + 0.5;
        return sqrt(2 * M_PI) * pow(t, x + 0.5) * exp(-t) * a;
    }
    
    double incompleteGamma(double s, double x) {
        const double EPSILON = 1e-10;
        const int MAXITER = 1000;
        
        if (x < 0) return 0.0;
        if (x == 0) return 0.0;
        if (s <= 0) throw std::runtime_error("Invalid shape parameter in incomplete gamma");
        
        double sum = 1.0 / s;
        double term = sum;
        
        for (int i = 1; i < MAXITER; i++) {
            term *= x / (s + i);
            sum += term;
            if (std::abs(term) < EPSILON) break;
        }
        
        return pow(x, s) * exp(-x) * sum;
    }
    
    double regularizedGammaP(double s, double x) {
        if (x < 0 || s <= 0) return 0.0;
        return incompleteGamma(s, x) / gamma(s);
    }

    double chiSquareCDF(double x, size_t df) {
        if (x < 0) return 0.0;
        return regularizedGammaP(df/2.0, x/2.0);
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

    // Advanced Psychometric Functions
    double kuderRichardson20(const std::vector<double>& responses) {
        if (responses.empty()) throw std::runtime_error("Empty response set!");
        
        size_t n = responses.size();
        double p = itemDifficulty(responses);
        double q = 1.0 - p;
        double pq = p * q;
        double variance = calculateVariance(responses, false);
        
        return (n / (n - 1.0)) * (1.0 - (pq / variance));
    }

    double spearmanBrownProphecy(double reliability, double lengthFactor) {
        if (reliability < 0 || reliability > 1) 
            throw std::runtime_error("Reliability must be between 0 and 1!");
        if (lengthFactor <= 0) 
            throw std::runtime_error("Length factor must be positive!");
        
        return (lengthFactor * reliability) / (1.0 + (lengthFactor - 1.0) * reliability);
    }

    std::vector<double> itemTotalCorrelations(const std::vector<std::vector<double>>& items) {
        if (items.empty() || items[0].empty()) 
            throw std::runtime_error("Empty dataset!");
        
        size_t numItems = items.size();
        size_t numResponses = items[0].size();
        std::vector<double> correlations;
        correlations.reserve(numItems);
        
        // Calculate total scores
        std::vector<double> totalScores(numResponses, 0.0);
        for (const auto& item : items) {
            if (item.size() != numResponses) 
                throw std::runtime_error("Inconsistent number of responses!");
            for (size_t i = 0; i < numResponses; ++i) {
                totalScores[i] += item[i];
            }
        }
        
        // Calculate correlation for each item
        for (const auto& item : items) {
            correlations.push_back(correlation(item, totalScores));
        }
        
        return correlations;
    }

    std::vector<double> standardError(const std::vector<double>& scores, double reliability) {
        if (scores.empty()) throw std::runtime_error("Empty score set!");
        if (reliability < 0 || reliability > 1) 
            throw std::runtime_error("Reliability must be between 0 and 1!");
        
        double stdDev = calculateStandardDeviation(scores);
        double sem = stdDev * sqrt(1 - reliability);
        
        std::vector<double> confidenceIntervals;
        confidenceIntervals.reserve(scores.size() * 2);
        
        // Calculate 95% confidence intervals (±1.96 SEM)
        for (double score : scores) {
            confidenceIntervals.push_back(score - 1.96 * sem);  // Lower bound
            confidenceIntervals.push_back(score + 1.96 * sem);  // Upper bound
        }
        
        return confidenceIntervals;
    }

    std::vector<double> itemInformationFunction(const std::vector<double>& abilities, 
                                              double difficulty, 
                                              double discrimination) {
        std::vector<double> information;
        information.reserve(abilities.size());
        
        for (double theta : abilities) {
            double p = 1.0 / (1.0 + exp(-discrimination * (theta - difficulty)));
            double q = 1.0 - p;
            information.push_back(discrimination * discrimination * p * q);
        }
        
        return information;
    }

    std::vector<double> testInformationFunction(const std::vector<double>& abilities,
                                              const std::vector<double>& difficulties,
                                              const std::vector<double>& discriminations) {
        if (difficulties.size() != discriminations.size())
            throw std::runtime_error("Number of difficulties must match number of discriminations!");
        
        std::vector<double> totalInformation(abilities.size(), 0.0);
        
        for (size_t i = 0; i < difficulties.size(); ++i) {
            auto itemInfo = itemInformationFunction(abilities, difficulties[i], discriminations[i]);
            for (size_t j = 0; j < abilities.size(); ++j) {
                totalInformation[j] += itemInfo[j];
            }
        }
        
        return totalInformation;
    }

    std::vector<double> differentialItemFunctioning(const std::vector<double>& group1Responses,
                                                  const std::vector<double>& group2Responses) {
        if (group1Responses.empty() || group2Responses.empty())
            throw std::runtime_error("Empty response sets!");
        
        double p1 = itemDifficulty(group1Responses);
        double p2 = itemDifficulty(group2Responses);
        
        double n1 = group1Responses.size();
        double n2 = group2Responses.size();
        
        double se = sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2));
        double zScore = (p1 - p2) / se;
        double pValue = 2 * (1 - normalCDF(abs(zScore), 0, 1));
        
        return {p1 - p2, zScore, pValue};
    }

    std::vector<double> parallelAnalysis(const std::vector<std::vector<double>>& data,
                                       int numFactors,
                                       int numIterations = 1000) {
        if (data.empty() || data[0].empty())
            throw std::runtime_error("Empty dataset!");
        
        size_t numVariables = data.size();
        size_t numObservations = data[0].size();
        
        // Calculate correlation matrix of actual data
        std::vector<std::vector<double>> correlationMatrix(numVariables, std::vector<double>(numVariables));
        for (size_t i = 0; i < numVariables; ++i) {
            for (size_t j = i; j < numVariables; ++j) {
                double r = (i == j) ? 1.0 : correlation(data[i], data[j]);
                correlationMatrix[i][j] = correlationMatrix[j][i] = r;
            }
        }
        
        // Calculate eigenvalues of actual data
        std::vector<double> actualEigenvalues = calculateEigenvalues(correlationMatrix);
        std::vector<double> randomEigenvalues(numFactors, 0.0);
        
        // Generate random data and calculate mean eigenvalues
        for (int iter = 0; iter < numIterations; ++iter) {
            auto randomData = generateRandomData(numVariables, numObservations);
            auto randomCorr = calculateCorrelationMatrix(randomData);
            auto eigenvalues = calculateEigenvalues(randomCorr);
            
            for (int i = 0; i < numFactors; ++i) {
                randomEigenvalues[i] += eigenvalues[i] / numIterations;
            }
        }
        
        return {actualEigenvalues.begin(), actualEigenvalues.begin() + numFactors};
    }

    // Enhanced Item Response Theory functions
    std::vector<double> itemResponseProbabilities(const std::vector<double>& abilities,
                                                double difficulty,
                                                double discrimination,
                                                double guessing = 0.0) {
        validateDataset(abilities, "Abilities");
        validateProbability(guessing, "Guessing parameter");
        
        std::vector<double> probabilities;
        probabilities.reserve(abilities.size());
        
        for (double theta : abilities) {
            double z = discrimination * (theta - difficulty);
            double p = guessing + (1.0 - guessing) / (1.0 + exp(-z));
            probabilities.push_back(p);
        }
        
        return probabilities;
    }

    // Enhanced DIF analysis with effect size
    struct DIFResult {
        double difference;
        double zScore;
        double pValue;
        double effectSize;
    };

    DIFResult differentialItemFunctioningEnhanced(const std::vector<double>& group1Responses,
                                                const std::vector<double>& group2Responses) {
        validateDataset(group1Responses, "Group 1 responses");
        validateDataset(group2Responses, "Group 2 responses");
        
        double p1 = itemDifficulty(group1Responses);
        double p2 = itemDifficulty(group2Responses);
        
        validateProbability(p1, "Group 1 difficulty");
        validateProbability(p2, "Group 2 difficulty");
        
        double n1 = group1Responses.size();
        double n2 = group2Responses.size();
        
        double se = sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2));
        double zScore = (p1 - p2) / se;
        double pValue = 2.0 * (1.0 - normalCDF(std::abs(zScore), 0, 1));
        
        // Calculate effect size (phi coefficient)
        double effectSize = (p1 - p2) / sqrt(p1 * (1 - p1) * p2 * (1 - p2));
        
        return {p1 - p2, zScore, pValue, effectSize};
    }

    // Enhanced reliability analysis
    struct ReliabilityMetrics {
        double cronbachAlpha;
        double standardError;
        double confidenceInterval[2];
    };

    ReliabilityMetrics calculateReliabilityMetrics(const std::vector<std::vector<double>>& items) {
        validateMatrixDimensions(items);
        
        double alpha = cronbachAlpha(items);
        
        // Calculate standard error of measurement
        std::vector<double> totalScores;
        size_t numRespondents = items[0].size();
        totalScores.reserve(numRespondents);
        
        for (size_t i = 0; i < numRespondents; ++i) {
            double sum = 0.0;
            for (const auto& item : items) {
                sum += item[i];
            }
            totalScores.push_back(sum);
        }
        
        double stdDev = calculateStandardDeviation(totalScores);
        double sem = stdDev * sqrt(1 - alpha);
        
        // Calculate 95% confidence interval for alpha
        double z = 1.96; // 95% confidence level
        double se_alpha = sqrt((1 - alpha * alpha) / (numRespondents - 2));
        
        return {
            alpha,
            sem,
            {alpha - z * se_alpha, alpha + z * se_alpha}
        };
    }

    // Enhanced factor analysis
    struct FactorAnalysisResult {
        std::vector<double> eigenvalues;
        double kmo;
        double bartlett_chi_square;
        double bartlett_p_value;
    };

    FactorAnalysisResult factorAnalysis(const std::vector<std::vector<double>>& data) {
        validateMatrixDimensions(data);
        
        size_t n = data.size();
        auto corrMatrix = calculateCorrelationMatrix(data);
        
        // Calculate KMO (Kaiser-Meyer-Olkin) measure
        double kmo = calculateKMO(corrMatrix);
        
        // Calculate Bartlett's test
        size_t df = (n * (n - 1)) / 2;
        double det = calculateDeterminant(corrMatrix);
        double chi_square = -(data[0].size() - 1 - (2 * n + 5) / 6) * log(det);
        double p_value = 1.0 - chiSquareCDF(chi_square, df);
        
        return {
            calculateEigenvalues(corrMatrix),
            kmo,
            chi_square,
            p_value
        };
    }

private:
    double calculateKMO(const std::vector<std::vector<double>>& corrMatrix) {
        size_t n = corrMatrix.size();
        double sum_r2 = 0.0;
        double sum_p2 = 0.0;
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                double r = corrMatrix[i][j];
                sum_r2 += r * r;
                
                // Calculate partial correlations
                std::vector<double> partial_r;
                for (size_t k = 0; k < n; ++k) {
                    if (k != i && k != j) {
                        double p = (corrMatrix[i][j] - corrMatrix[i][k] * corrMatrix[j][k]) /
                                 sqrt((1 - corrMatrix[i][k] * corrMatrix[i][k]) *
                                      (1 - corrMatrix[j][k] * corrMatrix[j][k]));
                        partial_r.push_back(p);
                    }
                }
                
                for (double p : partial_r) {
                    sum_p2 += p * p;
                }
            }
        }
        
        return sum_r2 / (sum_r2 + sum_p2);
    }

    double calculateDeterminant(const std::vector<std::vector<double>>& matrix) {
        size_t n = matrix.size();
        if (n == 1) return matrix[0][0];
        if (n == 2) return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        
        double det = 0.0;
        for (size_t j = 0; j < n; ++j) {
            std::vector<std::vector<double>> submatrix(n - 1, std::vector<double>(n - 1));
            for (size_t i = 1; i < n; ++i) {
                size_t k = 0;
                for (size_t l = 0; l < n; ++l) {
                    if (l != j) {
                        submatrix[i-1][k] = matrix[i][l];
                        ++k;
                    }
                }
            }
            det += (j % 2 == 0 ? 1 : -1) * matrix[0][j] * calculateDeterminant(submatrix);
        }
        return det;
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
    
    std::cout << "\nStatistical Analysis:" << std::endl;
    std::cout << "19. Descriptive Statistics" << std::endl;
    std::cout << "20. Correlation Analysis" << std::endl;
    std::cout << "21. Normal Distribution (PDF)" << std::endl;
    std::cout << "22. Normal Distribution (CDF)" << std::endl;
    
    std::cout << "\nPsychometric Analysis:" << std::endl;
    std::cout << "23. Reliability Analysis" << std::endl;
    std::cout << "24. Item Response Theory" << std::endl;
    std::cout << "25. Factor Analysis" << std::endl;
    std::cout << "26. Differential Item Functioning" << std::endl;
    std::cout << "27. Item-Total Correlations" << std::endl;
    std::cout << "28. Standard Error & Confidence Intervals" << std::endl;
    std::cout << "29. Parallel Analysis" << std::endl;
    std::cout << "30. Score Standardization" << std::endl;
    
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
                case 23: { // Reliability Analysis
                    std::cout << "Enter number of items: ";
                    int numItems;
                    std::cin >> numItems;
                    std::vector<std::vector<double>> items;
                    
                    for (int i = 0; i < numItems; ++i) {
                        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                        items.push_back(getDatasetFromUser("Enter responses for item " + std::to_string(i + 1)));
                    }
                    
                    auto metrics = calc.calculateReliabilityMetrics(items);
                    std::cout << "Cronbach's Alpha: " << metrics.cronbachAlpha << std::endl;
                    std::cout << "Standard Error of Measurement: " << metrics.standardError << std::endl;
                    std::cout << "95% Confidence Interval: [" << metrics.confidenceInterval[0] 
                              << ", " << metrics.confidenceInterval[1] << "]" << std::endl;
                    result = metrics.cronbachAlpha;
                    break;
                }
                
                case 24: { // Item Response Theory
                    auto abilities = getDatasetFromUser("Enter ability values (θ)");
                    
                    std::cout << "Enter item difficulty (b): ";
                    double difficulty;
                    std::cin >> difficulty;
                    
                    std::cout << "Enter item discrimination (a): ";
                    double discrimination;
                    std::cin >> discrimination;
                    
                    std::cout << "Enter guessing parameter (c) [0-1]: ";
                    double guessing;
                    std::cin >> guessing;
                    
                    auto probs = calc.itemResponseProbabilities(abilities, difficulty, discrimination, guessing);
                    std::cout << "\nItem Response Probabilities:" << std::endl;
                    for (size_t i = 0; i < abilities.size(); ++i) {
                        std::cout << "θ = " << abilities[i] << ": " << probs[i] << std::endl;
                    }
                    result = probs[0];
                    break;
                }
                
                case 25: { // Factor Analysis
                    std::cout << "Enter number of variables: ";
                    int numVars;
                    std::cin >> numVars;
                    std::vector<std::vector<double>> data;
                    
                    for (int i = 0; i < numVars; ++i) {
                        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                        data.push_back(getDatasetFromUser("Enter data for variable " + std::to_string(i + 1)));
                    }
                    
                    auto results = calc.factorAnalysis(data);
                    std::cout << "\nFactor Analysis Results:" << std::endl;
                    std::cout << "KMO Measure: " << results.kmo << std::endl;
                    std::cout << "Bartlett's Test: χ² = " << results.bartlett_chi_square 
                              << " (p = " << results.bartlett_p_value << ")" << std::endl;
                    std::cout << "\nEigenvalues:" << std::endl;
                    for (size_t i = 0; i < results.eigenvalues.size(); ++i) {
                        std::cout << "Factor " << (i + 1) << ": " << results.eigenvalues[i] << std::endl;
                    }
                    result = results.kmo;
                    break;
                }
                
                case 26: { // Differential Item Functioning
                    auto group1 = getDatasetFromUser("Enter responses for Group 1 (0/1)");
                    auto group2 = getDatasetFromUser("Enter responses for Group 2 (0/1)");
                    
                    auto difResults = calc.differentialItemFunctioningEnhanced(group1, group2);
                    std::cout << "\nDIF Analysis Results:" << std::endl;
                    std::cout << "Difficulty Difference: " << difResults.difference << std::endl;
                    std::cout << "Z-Score: " << difResults.zScore << std::endl;
                    std::cout << "P-Value: " << difResults.pValue << std::endl;
                    std::cout << "Effect Size: " << difResults.effectSize << std::endl;
                    result = difResults.difference;
                    break;
                }
                
                case 27: { // Item-Total Correlations
                    std::cout << "Enter number of items: ";
                    int numItems;
                    std::cin >> numItems;
                    std::vector<std::vector<double>> items;
                    
                    for (int i = 0; i < numItems; ++i) {
                        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                        items.push_back(getDatasetFromUser("Enter responses for item " + std::to_string(i + 1)));
                    }
                    
                    auto correlations = calc.itemTotalCorrelations(items);
                    std::cout << "\nItem-Total Correlations:" << std::endl;
                    for (size_t i = 0; i < correlations.size(); ++i) {
                        std::cout << "Item " << (i + 1) << ": " << correlations[i] << std::endl;
                    }
                    result = correlations[0];
                    break;
                }
                
                case 28: { // Standard Error & Confidence Intervals
                    auto scores = getDatasetFromUser("Enter test scores");
                    std::cout << "Enter reliability coefficient (0-1): ";
                    double reliability;
                    std::cin >> reliability;
                    
                    auto intervals = calc.standardError(scores, reliability);
                    std::cout << "\n95% Confidence Intervals:" << std::endl;
                    for (size_t i = 0; i < scores.size(); ++i) {
                        std::cout << "Score " << scores[i] << ": ["
                                 << intervals[i*2] << ", " << intervals[i*2+1] << "]" << std::endl;
                    }
                    result = intervals[0];
                    break;
                }
                
                case 29: { // Parallel Analysis
                    std::cout << "Enter number of variables: ";
                    int numVars;
                    std::cin >> numVars;
                    std::vector<std::vector<double>> data;
                    
                    for (int i = 0; i < numVars; ++i) {
                        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                        data.push_back(getDatasetFromUser("Enter data for variable " + std::to_string(i + 1)));
                    }
                    
                    std::cout << "Enter number of factors to extract: ";
                    int numFactors;
                    std::cin >> numFactors;
                    
                    auto eigenvalues = calc.parallelAnalysis(data, numFactors);
                    std::cout << "\nParallel Analysis Results:" << std::endl;
                    for (size_t i = 0; i < eigenvalues.size(); ++i) {
                        std::cout << "Factor " << (i + 1) << " eigenvalue: " << eigenvalues[i] << std::endl;
                    }
                    result = eigenvalues[0];
                    break;
                }
                
                case 30: { // Score Standardization
                    auto rawScores = getDatasetFromUser("Enter raw scores");
                    std::cout << "Enter desired mean (default 100): ";
                    double desiredMean;
                    std::cin >> desiredMean;
                    std::cout << "Enter desired standard deviation (default 15): ";
                    double desiredStdDev;
                    std::cin >> desiredStdDev;
                    
                    auto standardizedScores = calc.standardizeScores(rawScores, desiredMean, desiredStdDev);
                    std::cout << "\nStandardized Scores:" << std::endl;
                    for (size_t i = 0; i < standardizedScores.size(); ++i) {
                        std::cout << "Raw Score " << rawScores[i] << " → " 
                                 << standardizedScores[i] << std::endl;
                    }
                    result = standardizedScores[0];
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