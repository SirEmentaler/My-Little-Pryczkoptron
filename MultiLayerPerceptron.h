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

#ifndef MULTI_LAYER_PERCEPTRON_H_
#define MULTI_LAYER_PERCEPTRON_H_

#include <algorithm>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <utility>
#include <vector>
#include "NeuronLayerSpecification.h"
#include "NeuronLayer.h"

namespace mlp {

/// Template class representing a multilayer perceptron
/**
	A multilayer perceptron is a neural network consisting of some number
	of neuron layers.

	@tparam T Must meet the requirements of `NumericType` and for objects
	          `a, b` of type `T`, the expressions `a + b` and `a * b` must
	          be well-formed and be of type assignable to T.
*/
template<typename T>
class MultiLayerPerceptron {
public:
	/// Data type the class operates on
	using ValueType = T;
	/// Constructs the perceptron from range
	template<class InputIt>
	MultiLayerPerceptron(std::size_t inputSize, InputIt first, InputIt last);
	/// Constructs the perceptron from initializer list
	MultiLayerPerceptron(std::size_t inputSize, std::initializer_list<NeuronLayerSpecification<T>> init);
	/// Obtains number of layers of the perceptron
	std::size_t size() const;
	/// Produces neural network output based on provided input data
	template<class ForwardIt, class OutputIt>
	void test(ForwardIt first, OutputIt out) const;
	/// Trains neural network based on provided input data and expected output
	template<class InputIt1, class InputIt2>
	T train(InputIt1 first, InputIt2 expected);
	/// TODO
	void apply(T rate, T momentum);
	/// Generates biases and weights of neurons
	template<class Generator>
	void generateParameters(Generator gen);
private:
	template<class InputIt>
	void construct(std::size_t inputSize, InputIt first, InputIt last);
	std::size_t inputSize;
	std::vector<NeuronLayer<T>> layers;
};

/**
	The range `[first, last)` should contain specifications of neuron layers
	to be created within the perceptron. Each element of the range should
	be an ordered pair where the first object specifies the size of a layer
	and the second object specifies its activation function. The number of layers
	created will be equal to `std::distance(first, last)`.

	@tparam    InputIt   Must meet the requirements of `InputIterator`
	                     and dereference to a tuple-like type  with at least
	                     3 elements that supports `std::get`
	@param[in] inputSize Number of inputs of the perceptron
	@param[in] first     The beginning of the layer specification range
	@param[in] last      The end of the layer specification range
*/
template<typename T>
template<class InputIt>
MultiLayerPerceptron<T>::MultiLayerPerceptron(std::size_t inputSize, InputIt first, InputIt last) {
	construct(inputSize, first, last);
}

/**
	The initializer list should contain specifications of neuron layers
	to be created within the perceptron. Each element of the range should
	be an ordered pair where the first object specifies the size of a layer
	and the second object specifies its activation function. The number of layers
	created will be equal to `init.size()`.

	@param[in] inputSize Number of inputs of the perceptron
	@param[in] init      Initializer list containing layer specifications
*/
template<typename T>
MultiLayerPerceptron<T>::MultiLayerPerceptron(std::size_t inputSize, std::initializer_list<NeuronLayerSpecification<T>> init) {
	construct(inputSize, init.begin(), init.end());
}

/**
	@returns Size of the perceptron, i.e. number of neuron layers
*/
template<typename T>
std::size_t MultiLayerPerceptron<T>::size() const {
	return layers.size();
}

/**
	Interprets the range `[first, first + inputSize)` as perceptron input and
	feeds it to the neural network. The output of the final layer is then
	placed in the range beginning at `out`.

	@tparam     ForwardIt Must meet the requirements of `ForwardIterator`
	@tparam     OutputIt  Must meet the requirements of `OutputIterator`
	@param[in]  first     The beginning of the input range
	@param[out] out       The beginning of the destination range
*/
template<typename T>
template<class ForwardIt, class OutputIt>
void MultiLayerPerceptron<T>::test(ForwardIt first, OutputIt out) const {
	if (size() == 0) {
		std::copy_n(first, inputSize, out);
	} else {
		std::vector<T> inter(layers.front().group.size());
		layers.front().group.process(first, inter.begin());
		std::transform(inter.begin(), inter.end(), inter.begin(), layers.front().activation);
		auto operation = [&](const NeuronLayer<T>& layer) {
			std::vector<T> buffer(layer.group.size());
			layer.group.process(inter.begin(), buffer.begin());
			std::transform(buffer.begin(), buffer.end(), buffer.begin(), layer.activation);
			inter = std::move(buffer);
		};
		std::for_each(std::next(layers.begin()), layers.end(), operation);
		std::copy(inter.begin(), inter.end(), out);
	}
}

/**
	Interprets the range `[first, first + inputSize)` as perceptron input and
	feeds it to the neural network. Interprets the range
	`[expected, expected + outputSize)` as expected results of the operation
	and corrects weights and biases based on it.

	@tparam    InputIt1 Must meet the requirements of `ForwardIterator`
	@tparam    InputIt2 Must meet the requirements of `InputIterator`
	@param[in] first    The beginning of the input range
	@param[in] expected The beginning of the expected output range
*/
template<typename T>
template<class InputIt1, class InputIt2>
T MultiLayerPerceptron<T>::train(InputIt1 first, InputIt2 expected) {
	std::vector<std::vector<T>> sums;
	std::vector<std::vector<T>> activeSums;
	std::vector<T> factors(inputSize);
	sums.reserve(size() + 1);
	sums.emplace_back(inputSize);
	activeSums.reserve(size());
	std::copy_n(first, inputSize, factors.begin());
	std::copy_n(factors.begin(), inputSize, sums.front().begin());
	auto operation = [&](const NeuronLayer<T>& layer) {
		activeSums.push_back(factors);
		std::vector<T> buffer(layer.group.size());
		layer.group.process(factors.begin(), buffer.begin());
		sums.push_back(buffer);
		std::transform(buffer.begin(), buffer.end(), buffer.begin(), layer.activation);
		factors = std::move(buffer);
	};
	std::for_each(layers.begin(), layers.end(), operation);
	auto sumIt = sums.rbegin();
	auto activeSumIt = activeSums.rbegin();
	std::transform(factors.begin(), factors.end(), expected, factors.begin(), std::minus<T>());
	T result = std::inner_product(factors.begin(), factors.end(), factors.begin(), T());
	auto backpropagation = [&](NeuronLayer<T>& layer) {
		auto& last = *sumIt++;
		auto& lastActive = *activeSumIt++;
		std::transform(factors.begin(), factors.end(), last.begin(), factors.begin(), [&](T factor, T sum) {
			return factor * layer.activation.derivative(sum);
		});
		std::vector<T> buffer(lastActive.size());
		layer.group.modify(factors.begin(), lastActive.begin(), buffer.begin());
		factors = std::move(buffer);
	};
	std::for_each(layers.rbegin(), layers.rend(), backpropagation);
	return result;
}

template<typename T>
void MultiLayerPerceptron<T>::apply(T rate, T momentum) {
	for (auto&& layer : layers) {
		layer.group.apply(rate, momentum);
	}
}

/**
	Fills bias and weight values of all neurons in the network with
	outputs of function `gen`.

	@tparam    Generator An invokable type with signature equivalent to
	                     `Ret f()`, such that a value of type `Ret` may
	                     be assigned to a variable of type `T`
	@param[in] gen       The generator function
*/
template<typename T>
template<class Generator>
void MultiLayerPerceptron<T>::generateParameters(Generator gen) {
	for (auto&& layer : layers) {
		layer.group.generateParameters(gen);
	}
}

template<typename T>
template<class InputIt>
void MultiLayerPerceptron<T>::construct(std::size_t inputSize, InputIt first, InputIt last) {
	this->inputSize = inputSize;
	std::for_each(first, last, [&](const NeuronLayerSpecification<T>& spec) {
		layers.emplace_back(NeuronLayer<T>{NeuronGroup<T>(spec.size, inputSize), spec.activation});
		inputSize = spec.size;
	});
}

}

#endif
