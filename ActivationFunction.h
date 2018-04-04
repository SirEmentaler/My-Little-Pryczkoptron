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

/// Lightweight template class representing an activation function
/**
	An activation function is used by neuron layers.

	@tparam T Any copyable type
*/
template<typename T>
class ActivationFunction {
public:
	/// Data type the class operates on
	using ValueType = T;
	/// Function type
	using FunctionType = T(*)(T);
	/// Default constructor
	ActivationFunction() = default;
	/// Constructor from wrapped function
template<class WrappedFunction>
	ActivationFunction(const WrappedFunction&);
	/// Assigns underlying function
	template<class WrappedFunction>
	T operator=(const WrappedFunction&);
	/// Calls the function and returns a value
	T operator()(T x) const;
	/// Calls the derivative of the function and returns a value
	T derivative(T x) const;
private:
	FunctionType f;
	FunctionType df;
};

/**
	TODO
*/
template<typename T>
template<class WrappedFunction>
ActivationFunction<T>::ActivationFunction(const WrappedFunction&)
	: f(WrappedFunction::f), df(WrappedFunction::df) {}

/**
	TODO
*/
template<typename T>
template<class WrappedFunction>
T ActivationFunction<T>::operator=(const WrappedFunction&) {
	f = WrappedFunction::f;
	df = WrappedFunction::df;
}

/**
	TODO
*/
template<typename T>
T ActivationFunction<T>::operator()(T x) const {
	return f(x);
}

/**
	TODO
*/
template<typename T>
T ActivationFunction<T>::derivative(T x) const {
	return df(x);
}

}

#endif
