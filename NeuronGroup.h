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

#ifndef NEURON_GROUP_H_
#define NEURON_GROUP_H_

#include <algorithm>
#include <cstddef>
#include <functional>
#include <vector>
#include "Neuron.h"

namespace mlp {

/// Template class representing a group of neurons
/**
	A neuron group stores some number of neurons, as well as an activation
	function used collectively for all of them.

	@tparam T Must meet the requirements of `NumericType` and for objects
	          `a, b` of type `T`, the expressions `a + b` and `a * b` must
	          be well-formed and be of type assignable to T.
*/
template<typename T>
class NeuronGroup {
public:
	/// Data type the class operates on
	using ValueType = T;
	/// Neuron type
	using Neuron = Neuron<T>;
	/// Constructs the neuron layer
	NeuronGroup(std::size_t size, std::size_t inputSize);
	/// Obtains number of neurons in the layer
	std::size_t size() const;
	/// Obtains size of layer input
	std::size_t inputSize() const;
	/// Produces output based on provided input data
	template<class ForwardIt, class OutputIt>
	void process(ForwardIt first, OutputIt out) const;
	/// Determines changes to biases and weights
	template<class InputIt, class ForwardIt1, class ForwardIt2>
	void modify(InputIt factors, ForwardIt1 args, ForwardIt2 out);
	/// Applies changes to biases and weights
	void apply(T rate, T momentum);
	/// Generates biases of neurons
	template<class Generator>
	void generateBiases(Generator gen);
	/// Generates weights of neurons
	template<class Generator>
	void generateWeights(Generator gen);
private:
	std::size_t inSize;
	std::vector<Neuron> neurons;
};

/**
	@tparam    Function   An invokable type with signature equivalent to
	                      `Ret f(Arg)`, such that a value of type `Ret` may
	                      be assigned to a variable of type `T` and `T` is
	                      implicitly convertible to `Arg`
	@param[in] size       Number of neurons in the layer
	@param[in] inputSize  Number of inputs to the layer
*/
template<typename T>
NeuronGroup<T>::NeuronGroup(std::size_t size, std::size_t inputSize)
	: inSize(inputSize), neurons(size, Neuron(inputSize)) {}

/**
	@returns Size of the layer, i.e. number of neurons it contains
*/
template<typename T>
std::size_t NeuronGroup<T>::size() const {
	return neurons.size();
}

/**
	@returns Size of the expected input
*/
template<typename T>
std::size_t NeuronGroup<T>::inputSize() const {
	return inSize;
}

/**
	Interprets the range `[first, first + inputSize)` as neuron layer input
	and forwards it to the neurons. The output is then placed in the range
	beginning at `out`.

	@tparam     ForwardIt Must meet the requirements of `ForwardIterator`
	@tparam     OutputIt  Must meet the requirements of `OutputIterator`
	@param[in]  first     The beginning of the input range
	@param[out] out       The beginning of the destination range
*/
template<typename T>
template<class ForwardIt, class OutputIt>
void NeuronGroup<T>::process(ForwardIt first, OutputIt out) const {
	using namespace std::placeholders;
	auto stimulate = &Neuron::template stimulate<ForwardIt>;
	auto operation = std::bind(stimulate, _1, first);
	std::transform(neurons.begin(), neurons.end(), out, operation);
}

/**
	TODO
*/
template<typename T>
template<class InputIt, class ForwardIt1, class ForwardIt2>
void NeuronGroup<T>::modify(InputIt factors, ForwardIt1 args, ForwardIt2 out) {
	for (auto&& neuron : neurons) {
		neuron.nudge(args, *factors, out);
		++factors;
	}
}

/**
	@param[in] rate     Learning rate
	@param[in] momentum Momentum
*/
template<typename T>
void NeuronGroup<T>::apply(T rate, T momentum) {
	using namespace std::placeholders;
	std::for_each(neurons.begin(), neurons.end(), std::bind(Neuron::apply, _1, rate, momentum));
}

/**
	Fills bias values of all neurons in the layer with
	outputs of function `gen`.

	@tparam    Generator An invokable type with signature equivalent to
	                     `Ret f()`, such that a value of type `Ret` may
	                     be assigned to a variable of type `T`
	@param[in] gen       The generator function
*/
template<typename T>
template<class Generator>
void NeuronGroup<T>::generateBiases(Generator gen) {
	for (auto&& neuron : neurons) {
		neuron.setBias(gen());
	}
}

/**
	Fills bias and weight values of all neurons in the layer with
	outputs of function `gen`.

	@tparam    Generator An invokable type with signature equivalent to
	                     `Ret f()`, such that a value of type `Ret` may
	                     be assigned to a variable of type `T`
	@param[in] gen       The generator function
*/
template<typename T>
template<class Generator>
void NeuronGroup<T>::generateWeights(Generator gen) {
	using namespace std::placeholders;
	auto generate = &Neuron::template generateWeights<Generator>;
	auto operation = std::bind(generate, _1, gen);
	std::for_each(neurons.begin(), neurons.end(), operation);
}

}

#endif
