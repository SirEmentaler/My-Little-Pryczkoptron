#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include "IdentityFunction.h"
#include "LogisticFunction.h"
#include "MultiLayerPerceptron.h"
#include "RandomNumberGenerator.h"

int main() {
	mlp::MultiLayerPerceptron<double> network(1, {
		{64, std::make_shared<mlp::LogisticFunction<double>>()},
		{1, std::make_shared<mlp::IdentityFunction<double>>()},
	});
	mlp::RandomNumberGenerator<double, std::mt19937_64> generator(-1.0, 1.0);
	network.generateParameters(generator);
	std::mt19937_64 shuffler;
	std::vector<std::pair<double, double>> input;
	std::ifstream in("approximation_train_1.txt");
	for (std::pair<double, double> p; in >> p.first >> p.second;) {
		input.push_back(p);
	}
	const int epochCount = 10000;
	const int epochCountPercent = epochCount / 100;
	for (int i = 0; i < epochCount; i++) {
		if (i % epochCountPercent == 0) {
			double error = 0.0;
			for (const auto& p : input) {
				double output;
				network.test(&p.first, &output);
				double diff = output - p.second;
				error += diff * diff;
			}
			std::cout << (i / epochCountPercent) << '%' << " - error is " << error << std::endl;
		}
		std::shuffle(input.begin(), input.end(), shuffler);
		for (const auto& p : input) {
			network.train(&p.first, &p.second);
			network.apply(1e-2);
		}
	}
	std::cout << "100%" << std::endl;
	std::ofstream out("approximation_results_1.txt");
	out << "Argument\tExpected\tObtained\n";
	std::sort(input.begin(), input.end());
	for (const auto& p : input) {
		double output;
		network.test(&p.first, &output);
		out << p.first << '\t' << p.second << '\t' << output << '\n';
	}
	return 0;
}
