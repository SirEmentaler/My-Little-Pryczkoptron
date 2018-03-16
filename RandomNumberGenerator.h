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

#ifndef RANDOM_NUMBER_GENERATOR_H_
#define RANDOM_NUMBER_GENERATOR_H_

#include <chrono>
#include <random>

namespace mlp {

/// Template class representing a pseudo-random real number generator
/**
	TODO: Detailed description
*/
template<typename T, class UnderlyingType>
class RandomNumberGenerator {
public:
	/// Type of generated numbers
	using ResultType = T;
	/// Constructs the generator
	RandomNumberGenerator(T min, T max);
	/// Advances state and returns the generated value
	T operator()();
private:
	std::uniform_real_distribution<T> distribution;
	UnderlyingType generator;
};

/**
	@param[in] min Minimum generated value
	@param[in] max Maximum generated value
*/
template<typename T, class U>
RandomNumberGenerator<T, U>::RandomNumberGenerator(T min, T max)
	: distribution(min, max) {
	auto time = std::chrono::system_clock::now().time_since_epoch().count();
	std::random_device randomDevice;
	generator.seed(time ^ randomDevice());
}

/**
	@returns Pseudo-random real number from the range [min, max]
*/
template<typename T, class U>
T RandomNumberGenerator<T, U>::operator()() {
	return distribution(generator);
}

}

#endif
