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

#ifndef RECTIFIER_H_
#define RECTIFIER_H_

#include <algorithm>
#include <cmath>
#include "ActivationFunction.h"

namespace mlp {

/// Template class representing a rectifier activation function
/**
	Activation function @f$ f(x) = x^+ = \max(0,x) @f$ .

	@tparam T Must meet the requirements of `NumericType` and for a variable
	          `x` of type `T`, the expression `std::signbit(x)` must be well
	          formed
*/
template<typename T>
class Rectifier {
public:
	/// Returns rectifier function value
	constexpr static T f(T x);
	/// Returns rectifier derivative value
	constexpr static T df(T x);
};

/**
	@param[in] x Function argument

	@returns Function value @f$ f(x) = x^+ = \max(0,x) @f$
*/
template<typename T>
constexpr T Rectifier<T>::f(T x) {
	return std::max(T(), x);
}

/**
	The result is unspecified if `x == 0` but guaranteed to be within the
	range `[0, 1]`.

	@param[in] x Derivative argument

	@returns Derivative value @f$ f^\prime(x) = \begin{cases} 0, & x<0 \\ 1,
	         & x>0 \\ \text{unspecified}, & x=0 \end{cases} @f$
*/
template<typename T>
constexpr T Rectifier<T>::df(T x) {
	return std::signbit(x) ? T() : T(1);
}

}

#endif

