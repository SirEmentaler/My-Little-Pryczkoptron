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

#ifndef HYPERBOLIC_TANGENT_H_
#define HYPERBOLIC_TANGENT_H_

#include <cmath>
#include "ActivationFunction.h"

namespace mlp {

/// Template class representing a sigmoid activation function
/**
	TODO

	@tparam T Must meet the requirements of `NumericType` and for a variable
	          `x` of type `T`, the expression `std::tanh(x)` must be well
	          formed
*/
template<typename T>
class HyperbolicTangent {
public:
	/// Returns hyperbolic tangent value
	static T f(T x);
	/// Returns hyperbolic tangent derivative value
	static T df(T x);
};

/**
	TODO
*/
template<typename T>
T HyperbolicTangent<T>::f(T x) {
	return std::tanh(x);
}

/**
	TODO
*/
template<typename T>
T HyperbolicTangent<T>::df(T x) {
	T t = std::tanh(x);
	return T(1) - t * t;
}

}

#endif

