#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include "MultiLayerPerceptron.h"
#include "RandomNumberGenerator.h"

double sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}

double sigmoidDerivative(double x) {
	double e = std::exp(x);
	double e1 = e + 1.0;
	return e / (e1 * e1);
}

double identity(double x) {
	return x;
}

double one(double) {
	return 1.0;
}

int main() {
	mlp::MultiLayerPerceptron<double> network(1, {
		std::make_tuple(20, sigmoid, sigmoidDerivative),
		std::make_tuple(1, identity, one),
	});
	mlp::RandomNumberGenerator<double, std::mt19937_64> generator(-8.0, 8.0);
	network.generateParameters(generator);
	double input;
	double output;
	for (int i = -10; i < 10; i++) {
		input = i;
		network.test(&input, &input + 1, &output);
		std::cout << i << ':' << output << std::endl;
	}
	return 0;
}
