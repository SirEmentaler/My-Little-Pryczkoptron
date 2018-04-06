#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include "IdentityFunction.h"
#include "HyperbolicTangent.h"
#include "MultiLayerPerceptron.h"
#include "RandomNumberGenerator.h"

int main() {
	mlp::MultiLayerPerceptron<double> network(1, {
		{17, mlp::HyperbolicTangent<double>()},
		{1, mlp::IdentityFunction<double>()},
	});
	mlp::RandomNumberGenerator<double, std::mt19937_64> generator(-1.0, 1.0);
	network.generateWeights(generator);
	{
		std::vector<std::pair<double, double>> input;
		std::ifstream in("approximation_train_1.txt");
		for (std::pair<double, double> p; in >> p.first >> p.second;) {
			input.push_back(p);
		}
		const int epochCount = 1000;
		const int epochCountPercent = epochCount / 100;
		std::mt19937_64 shuffler;
		for (int i = 0; i < epochCount; i++) {
			double error = 0.0;
			std::shuffle(input.begin(), input.end(), shuffler);
			for (const auto& p : input) {
				error += network.train(&p.first, &p.second);
			}
			network.apply(1e-4, 0.99);
			if (i % epochCountPercent == 0) {
				std::cout << "Learning... " <<(i / epochCountPercent) << '%' << " - error is " << error << std::endl;
			}
		}
		std::cout << "Learning complete" << std::endl;
	}
	{
		std::vector<std::pair<double, double>> input;
		std::ifstream in("approximation_test.txt");
		for (std::pair<double, double> p; in >> p.first >> p.second;) {
			input.push_back(p);
		}
		std::ofstream out("approximation_results_2.txt");
		out << "Argument\tExpected\tObtained\n";
		double error = 0.0;
		std::sort(input.begin(), input.end());
		std::cout << "Running tests..." << std::endl;
		for (const auto& p : input) {
			double output;
			network.test(&p.first, &output);
			error += (output - p.second) * (output - p.second);
			out << p.first << '\t' << p.second << '\t' << output << '\n';
		}
		std::cout << "Done" << std::endl;
		std::cout << "Average test error was " << (error / input.size()) << std::endl;
	}
	return 0;
}
