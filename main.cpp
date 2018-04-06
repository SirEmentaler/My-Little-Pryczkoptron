#include <fstream>
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
	trainer.setLearningRate(1e-4);
	trainer.setMomentum(0.9);
	std::ifstream trainingData("approximation_train_1.txt");
	for (double input, output; trainingData >> input >> output;) {
		trainer.addTest(&input, &output);
	}
	trainer.train(network);
	std::vector<std::pair<double, double>> input;
	std::ifstream in("approximation_test.txt");
	for (std::pair<double, double> p; in >> p.first >> p.second;) {
		input.push_back(p);
	}
	std::ofstream out("approximation_results_1.txt");
	out << "Argument\tExpected\tObtained\n";
	for (const auto& p : input) {
		double output;
		network.test(&p.first, &output);
		out << p.first << '\t' << p.second << '\t' << output << '\n';
	}
	return 0;
}
