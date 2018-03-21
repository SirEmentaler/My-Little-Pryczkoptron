#include <iostream>
#include <random>
#include "IdentityFunction.h"
#include "LogisticFunction.h"
#include "MultiLayerPerceptron.h"
#include "RandomNumberGenerator.h"

int main() {
	mlp::MultiLayerPerceptron<double> network(1, {
		{20, std::make_shared<mlp::LogisticFunction<double>>()},
		{1, std::make_shared<mlp::IdentityFunction<double>>()},
	});
	mlp::RandomNumberGenerator<double, std::mt19937_64> generator(-8.0, 8.0);
	network.generateParameters(generator);
	double input;
	double output;
	for (int i = -10; i < 10; i++) {
		input = i;
		network.test(&input, &output);
		std::cout << i << ':' << output << std::endl;
	}
	return 0;
}
