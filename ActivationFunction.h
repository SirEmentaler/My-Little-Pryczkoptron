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

#ifndef ACTIVATION_FUNCTION_H_
#define ACTIVATION_FUNCTION_H_

namespace mlp {

/// Template abstract class representing an activation function with derivative
/**
	An activation function is used by neuron layers.

	@tparam T Any copyable type
*/
template<typename T>
class ActivationFunction {
public:
	/// Data type the class operates on
	using ValueType = T;
	/// Default virtual destructor
	virtual ~ActivationFunction() = default;
	/// Calls the function and returns a value
	virtual T operator()(T) const = 0;
	/// Calls the derivative of the function and returns a value
	virtual T derivative(T) const = 0;
};

}

#endif
