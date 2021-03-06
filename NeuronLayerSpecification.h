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

#ifndef NEURON_LAYER_SPECIFICATION_H_
#define NEURON_LAYER_SPECIFICATION_H_

#include <cstddef>
#include "ActivationFunction.h"

namespace mlp {

/// Template structure representing a neuron layer specification
/**
	A neuron layer specification is sufficient to construct a neuron layer.
	It contains the size of the layer and its activation function.

	@tparam T Must meet the requirements of `NumericType` and for objects
	          `a, b` of type `T`, the expressions `a + b` and `a * b` must
	          be well-formed and be of type assignable to T.
*/
template<typename T>
struct NeuronLayerSpecification {
	/// Data type the class operates on
	using ValueType = T;
	/// Size of the layer, i.e. its number of neurons
	std::size_t size;
	/// Activation function of the layer
	ActivationFunction<T> activation;
};

}

#endif

