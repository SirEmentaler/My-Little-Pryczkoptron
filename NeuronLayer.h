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

#ifndef NEURON_LAYER_H_
#define NEURON_LAYER__H_

#include <algorithm>
#include <cstddef>
#include <functional>
#include <vector>
#include "Neuron.h"

namespace mlp {

/// Template class representing one layer of neurons
/**
	A neuron layer stores some number of neurons, as well as an activation
	function used collectively for all of them.

	@tparam T Must meet the requirements of `NumericType` and for objects
	          `a, b` of type `T`, the expressions `a + b` and `a * b` must
	          be well-formed and be of type assignable to T.
*/
template<typename T>
class NeuronLayer {
public:
	/// Data type the class operates on
	using ValueType = T;
	/// Neuron type
	using Neuron = Neuron<T>;
	/// Constructs the neuron layer
	template<class Function>
	NeuronLayer(std::size_t size, std::size_t inputSize, Function activation);
	/// Obtains number of neurons in the layer
	std::size_t size() const;
	/// Produces output based on provided input data
	template<class ForwardIt, class OutputIt>
	void process(ForwardIt first, ForwardIt last, OutputIt out) const;
	/// Generates biases and weights of neurons
	template<class Generator>
	void generateParameters(Generator gen);
private:
	std::vector<Neuron> neurons;
	std::function<T(T)> activation;
};

/**
	@tparam    Function   An invokable type with signature equivalent to
	                      `Ret f(Arg)`, such that a value of type `Ret` may
	                      be assigned to a variable of type `T` and `T` is
	                      implicitly convertible to `Arg`
	@param[in] size       Number of neurons in the layer
	@param[in] inputSize  Number of inputs to the layer
	@param[in] activation Activation function used by the layer
*/
template<typename T>
template<class Function>
NeuronLayer<T>::NeuronLayer(std::size_t size, std::size_t inputSize, Function activation)
	: neurons(size, Neuron(inputSize)), activation(activation) {}

/**
	@returns Size of the layer, i.e. number of neurons it contains
*/
template<typename T>
std::size_t NeuronLayer<T>::size() const {
	return neurons.size();
}

/**
	Interprets the range `[first, last)` as neuron layer input and forwards
	it to the neurons. The output of is then placed in the range beginning
	at `out`.

	The size of `[first, last)` must match exactly the input size expected
	by the layer, otherwise the behavior is undefined.

	@tparam     ForwardIt Must meet the requirements of `ForwardIterator`
	@tparam     OutputIt  Must meet the requirements of `OutputIterator`
	@param[in]  first     The beginning of the input range
	@param[in]  last      The end of the input range
	@param[out] out       The beginning of the destination range
*/
template<typename T>
template<class ForwardIt, class OutputIt>
void NeuronLayer<T>::process(ForwardIt first, ForwardIt last, OutputIt out) const {
	using namespace std::placeholders;
	auto stimulate = &Neuron::template stimulate<ForwardIt>;
	auto boundStimulate = std::bind(stimulate, _1, first, last);
	auto operation = std::bind(activation, boundStimulate);
	std::transform(neurons.begin(), neurons.end(), out, operation);
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
void NeuronLayer<T>::generateParameters(Generator gen) {
	using namespace std::placeholders;
	auto generate = &Neuron::template generateParameters<Generator>;
	auto operation = std::bind(generate, _1, gen);
	std::for_each(neurons.begin(), neurons.end(), operation);
}

}

#endif
