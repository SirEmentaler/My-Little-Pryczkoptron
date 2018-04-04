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
#define NEURON_LAYER_H_

#include "NeuronGroup.h"
#include "ActivationFunction.h"

namespace mlp {

/// Template structure representing a neuron layer with activation function
/**
	A neuron layer stores a group of neurons, as well as an activation
	function used collectively for all of them.

	@tparam T Must meet the requirements of `NumericType` and for objects
	          `a, b` of type `T`, the expressions `a + b` and `a * b` must
	          be well-formed and be of type assignable to T.
*/
template<typename T>
struct NeuronLayer {
	/// Data type the class operates on
	using ValueType = T;
	/// The group of neurons
	NeuronGroup<T> group;
	/// The activation function
	ActivationFunction<T> activation;
};

}

#endif


