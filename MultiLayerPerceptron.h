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
#include <tuple>
#include <utility>
#include <vector>
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
	/// Layer type
	using Layer = NeuronLayer<T>;
	/// Constructs the perceptron from range
	template<class InputIt>
	MultiLayerPerceptron(std::size_t inputSize, InputIt first, InputIt last);
	/// Constructs the perceptron from initializer list
	template<class TupleType>
	MultiLayerPerceptron(std::size_t inputSize, std::initializer_list<TupleType> init);
	/// Obtains number of layers of the perceptron
	std::size_t size() const;
	/// Produces neural network output based on provided input data
	template<class BidirIt, class OutputIt>
	void test(BidirIt first, BidirIt last, OutputIt out) const;
	/// Generates biases and weights of neurons
	template<class Generator>
	void generateParameters(Generator gen);
private:
	template<class InputIt>
	void construct(std::size_t inputSize, InputIt first, InputIt last);
	std::vector<Layer> layers;
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

	@tparam    TupleType Must be tuple-like type with at least 3 elements
	                     and support `std::get`
	@param[in] inputSize Number of inputs of the perceptron
	@param[in] init      Initializer list containing layer specifications
*/
template<typename T>
template<class TupleType>
MultiLayerPerceptron<T>::MultiLayerPerceptron(std::size_t inputSize, std::initializer_list<TupleType> init) {
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
	Interprets the range `[first, last)` as perceptron input and feeds it
	to the neural network. The output of the final layer is then placed in
	the range beginning at `out`.

	The size of `[first, last)` must match exactly the input size expected
	by the perceptron, otherwise the behavior is undefined.

	@tparam     BidirIt  Must meet the requirements of `BidirectionalIterator`
	@tparam     OutputIt Must meet the requirements of `OutputIterator`
	@param[in]  first    The beginning of the input range
	@param[in]  last     The end of the input range
	@param[out] out      The beginning of the destination range
*/
template<typename T>
template<class BidirIt, class OutputIt>
void MultiLayerPerceptron<T>::test(BidirIt first, BidirIt last, OutputIt out) const {
	if (size() == 0) {
		std::copy(first, last, out);
	} else if (size() == 1) {
		layers.front().process(first, last, out);
	} else {
		std::vector<T> inter(layers.front().size());
		layers.front().process(first, last, inter.begin());
		auto operation = [&](const Layer& layer) {
			std::vector<T> buffer(layer.size());
			layer.process(inter.begin(), inter.end(), buffer.begin());
			inter = std::move(buffer);
		};
		std::for_each(std::next(layers.begin()), std::prev(layers.end()), operation);
		layers.back().process(inter.begin(), inter.end(), out);
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
	using namespace std::placeholders;
	auto generate = &Layer::template generateParameters<Generator>;
	auto operation = std::bind(generate, _1, gen);
	std::for_each(layers.begin(), layers.end(), operation);
}

template<typename T>
template<class InputIt>
void MultiLayerPerceptron<T>::construct(std::size_t inputSize, InputIt first, InputIt last) {
	std::for_each(first, last, [&](const auto& tuple) {
		std::size_t size = std::get<0>(tuple);
		layers.emplace_back(size, inputSize, std::get<1>(tuple), std::get<2>(tuple));
		inputSize = size;
	});
}

}

#endif
