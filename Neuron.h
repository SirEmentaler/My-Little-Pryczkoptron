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

#ifndef NEURON_H_
#define NEURON_H_

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

namespace mlp {

/// Template class representing a single neuron of a perceptron
/**
	Each neuron stores an array of weights assigned to its inputs as well
	as a single bias. The key functionality of a neuron is returning its
	activation for given inputs via `Neuron::stimulate`.

	@tparam T Must meet the requirements of `NumericType` and for objects
	          `a, b` of type `T`, the expressions `a + b` and `a * b` must
	          be well-formed and be of type assignable to T.
*/
template<typename T>
class Neuron {
public:
	/// Data type the class operates on
	using ValueType = T;
	/// Constructs the neuron from number of inputs
	explicit Neuron(std::size_t inputSize);
	/// Feeds input to the neuron and obtains result
	template<class InputIt>
	T stimulate(InputIt first, InputIt last) const;
	/// Generates bias and weights
	template<class Generator>
	void generateParameters(Generator gen);
private:
	T bias = T();
	std::vector<T> weights;
};

/**
	Constructs the neuron. Bias and all weight values are initialized
	to the default value of T.

	@param[in] inputSize Number of inputs of the neuron
*/
template<typename T>
Neuron<T>::Neuron(std::size_t inputSize)
	: weights(inputSize) {}

/**
	Interprets the range `[first, last)` as neuron input and return the
	resulting activation level. Internally, multiplies input by corresponding
	weights and sums with bias.

	The size of `[first, last)` must match exactly the input size expected
	by the neuron, otherwise the behavior is undefined.

	@tparam     InputIt   Must meet the requirements of `InputIterator`
	@tparam     OutputIt  Must meet the requirements of `OutputIterator`
	@param[in]  first     The beginning of the input range
	@param[in]  last      The end of the input range

	@returns Activation level of the neuron
*/
template<typename T>
template<class InputIt>
T Neuron<T>::stimulate(InputIt first, InputIt last) const {
	return std::inner_product(first, last, weights.begin(), bias);
}

/**
	Fills bias and weight values of the neuron with outputs of function `gen`.

	@tparam    Generator An invokable type with signature equivalent to
	                     `Ret f()`, such that a value of type `Ret` may
	                     be assigned to a variable of type `T`
	@param[in] gen       The generator function
*/
template<typename T>
template<class Generator>
void Neuron<T>::generateParameters(Generator gen) {
	bias = gen();
	std::generate(weights.begin(), weights.end(), gen);
}

}

#endif
