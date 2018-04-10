////////////////////////////////////////////////////////////
//
// Copyright (c) 2018 Jan Filipowicz, Filip Turobos
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
////////////////////////////////////////////////////////////

#ifndef PERCEPTRON_TRAINER_H_
#define PERCEPTRON_TRAINER_H_

#include <algorithm>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>
#include "MultiLayerPerceptron.h"
#include "RandomNumberGenerator.h"

namespace mlp {

template<typename T>
class PerceptronTrainer {
public:
	/// TODO
	PerceptronTrainer(std::size_t inputSize, std::size_t outputSize);
	/// TODO
	template<class Perceptron>
	void train(Perceptron& perceptron) const;
	/// TODO
	template<class InputIt1, class InputIt2>
	void addTest(InputIt1 inFirst, InputIt2 outFirst);
	/// TODO
	void setMaxEpochs(std::size_t value) {maxEpochs = value;}
	/// TODO
	void setErrorThreshold(T value) {errorThreshold = value;}
	/// TODO
	void setInitialWeightRange(T value) {initialWeightRange = value;}
	/// TODO
	void setLearningRate(T value) {learningRate = value;}
	/// TODO
	void setMomentum(T value) {momentum = value;}
private:
	std::vector<std::pair<std::vector<T>, std::vector<T>>> dataSet;
	std::size_t inputSize;
	std::size_t outputSize;
	std::size_t maxEpochs = 0;
	T errorThreshold = T();
	T initialWeightRange = T();
	T learningRate = T();
	T momentum = T();
};

/**
	TODO
*/
template<typename T>
PerceptronTrainer<T>::PerceptronTrainer(std::size_t inputSize, std::size_t outputSize)
	: inputSize(inputSize), outputSize(outputSize) {}

/**
	TODO
*/
template<typename T>
template<class Perceptron>
void PerceptronTrainer<T>::train(Perceptron& perceptron) const {
	double scaledThreshold = errorThreshold * dataSet.size();
	RandomNumberGenerator<T, std::mt19937_64> generator(-initialWeightRange, initialWeightRange);
	perceptron.generateWeights(generator);
	for (std::size_t i = maxEpochs; i--;) {
		T error = T();
		for (const auto& test : dataSet) {
			error += perceptron.train(test.first.begin(), test.second.begin());
		}
		if (error < scaledThreshold)
			return;
		perceptron.apply(learningRate, momentum);
	}
}

/**
	TODO
*/
template<typename T>
template<class InputIt1, class InputIt2>
void PerceptronTrainer<T>::addTest(InputIt1 inFirst, InputIt2 outFirst) {
	std::vector<T> in(inputSize);
	std::copy_n(inFirst, inputSize, in.begin());
	std::vector<T> out(outputSize);
	std::copy_n(outFirst, outputSize, out.begin());
	dataSet.emplace_back(std::move(in), std::move(out));
}

}

#endif

