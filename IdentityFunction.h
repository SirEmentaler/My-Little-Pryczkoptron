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

#ifndef IDENTITY_FUNCTION_H_
#define IDENTITY_FUNCTION_H_

#include "ActivationFunction.h"

namespace mlp {

/// Template class representing a linear activation function
/**
	Activation function @f$ f(x) = x @f$ .

	@tparam T Must meet the requirements of `NumericType`
*/
template<typename T>
class IdentityFunction {
public:
	/// Returns its argument
	constexpr static T f(T x);
	/// Returns 1
	constexpr static T df(T);
};

/**
	@param[in] x Function argument

	@returns Function value @f$ f(x) = x @f$
*/
template<typename T>
constexpr T IdentityFunction<T>::f(T x) {
	return x;
}

/**
	@returns Derivative value @f$ f^\prime(x) = 1 @f$
*/
template<typename T>
constexpr T IdentityFunction<T>::df(T) {
	return T(1);
}

}

#endif
