#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
#include <limits>
#include <vector>
#include <stack>

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
};

void displayMenu() {
    std::cout << "\n=== Scientific Calculator ===" << std::endl;
    std::cout << "1. Addition (+)" << std::endl;
    std::cout << "2. Subtraction (-)" << std::endl;
    std::cout << "3. Multiplication (*)" << std::endl;
    std::cout << "4. Division (/)" << std::endl;
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
    std::cout << "0. Exit" << std::endl;
    std::cout << "Enter your choice: ";
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