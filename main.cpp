#include <cmath>
#include <iostream>
#include <random>
#include "MultiLayerPerceptron.h"
#include "RandomNumberGenerator.h"

double sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}

double identity(double x) {
	return x;
}

int main() {
	mlp::MultiLayerPerceptron<double> network(1, {
		std::make_pair(20, sigmoid),
		std::make_pair(1, identity),
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
