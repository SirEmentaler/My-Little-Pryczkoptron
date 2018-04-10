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

#ifndef LOGISTIC_FUNCTION_H_
#define LOGISTIC_FUNCTION_H_

#include <cmath>
#include "ActivationFunction.h"

namespace mlp {

/// Template class representing a sigmoid activation function
/**
	Activation function @f$ f(x) = \frac{1}{1+e^{-x}} @f$ .

	@tparam T Must meet the requirements of `NumericType` and for a variable
	          `x` of type `T`, the expression `std::exp(x)` must be well formed
*/
template<typename T>
class LogisticFunction {
public:
	/// Returns logistic function value
	constexpr static T f(T x);
	/// Returns logistic function derivative value
	constexpr static T df(T x);
};

/**
	@param[in] x Function argument

	@returns Function value @f$ f(x) = \frac{1}{1+e^{-x}} @f$
*/
template<typename T>
constexpr T LogisticFunction<T>::f(T x) {
	return T(1) / (T(1) + std::exp(-x));
}

/**
	@param[in] x Derivative argument

	@returns Derivative value @f$ f^\prime(x) = \frac{e^x}{(1+e^x)^2} @f$
*/
template<typename T>
constexpr T LogisticFunction<T>::df(T x) {
	double e = std::exp(x);
	double e1 = e + T(1);
	return e / (e1 * e1);
}

}

#endif

