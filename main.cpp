/*#include <fstream>
#include <utility>
#include <vector>
#include "IdentityFunction.h"
#include "HyperbolicTangent.h"
#include "MultiLayerPerceptron.h"
#include "PerceptronTrainer.h"

int main() {
	mlp::MultiLayerPerceptron<double> network(1, {
		{17, mlp::HyperbolicTangent<double>()},
		{1, mlp::IdentityFunction<double>()},
	});
	mlp::PerceptronTrainer<double> trainer(1, 1);
	trainer.setMaxEpochs(10000);
	trainer.setErrorThreshold(0.06);
	trainer.setInitialWeightRange(0.25);
	trainer.setLearningRate(1e-4);
	trainer.setMomentum(0.9);
	std::ifstream trainingData("approximation_train_1.txt");
	for (double input, output; trainingData >> input >> output;) {
		trainer.addTest(&input, &output);
	}
	trainer.train(network);
	std::vector<std::pair<double, double>> data;
	std::ifstream testData("approximation_test.txt");
	for (double input, output; testData >> input >> output;) {
		data.emplace_back(input, output);
	}
	std::ofstream out("approximation_results_1.txt");
	out << "Argument\tExpected\tObtained\n";
	for (const auto& inout : data) {
		double output;
		network.test(&inout.first, &output);
		out << inout.first << '\t' << inout.second << '\t' << output << '\n';
	}
	return 0;
}*/

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include "IdentityFunction.h"
#include "LogisticFunction.h"
#include "MultiLayerPerceptron.h"
#include "RandomNumberGenerator.h"
#include "PerceptronTrainer.h"

int main() {
	mlp::MultiLayerPerceptron<double> network(4, {
		{17, mlp::LogisticFunction<double>()},
		{3, mlp::LogisticFunction<double>()},
	});
	mlp::PerceptronTrainer<double> trainer(4, 3);
	trainer.setMaxEpochs(2000);
	trainer.setErrorThreshold(0.01);
	trainer.setInitialWeightRange(0.25);
	trainer.setLearningRate(1e-3);
	trainer.setMomentum(0.8);
	std::ifstream trainingData("classification_train.txt");
	for (double input[4]; trainingData >> input[0] >> input[1] >> input[2] >> input[3];) {
		double output[3] {};
		int outcome;
		trainingData >> outcome;
		output[outcome - 1] = 1.0;
		trainer.addTest(input, output);
	}
	trainer.train(network);
	std::vector<std::pair<std::array<double, 4>, int>> data;
	std::ifstream testData("classification_test.txt");
	for (std::array<double, 4> input; testData >> input[0] >> input[1] >> input[2] >> input[3];) {
		int outcome;
		testData >> outcome;
		data.emplace_back(input, outcome);
	}
	std::ofstream out("classification_results_1.txt");
	out << "Expected\tObtained\n";
	int correct = 0;
	for (const auto& inout : data) {
		double output[3];
		network.test(inout.first.begin(), output);
		int result = std::max_element(std::begin(output), std::end(output)) - output + 1;
		out << inout.second << '\t' << result << '\n';
		correct += result == inout.second;
	}
	std::cout << correct << " out of " << data.size() << " guessed" << std::endl;
	return 0;
}
