// Defining this would make REGISTER_DISPATCH macro
// replace CPU kernel ptr with HB kernel ptr in 
// the dispatch table
#define __HB__

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/hb/HammerBladeOffload.h>

namespace at {
namespace native {

namespace { // anonymous

#include <iostream>
void add_kernel(TensorIterator& iter, Scalar alpha_scalar) {
  std::cout << "\nCalled HB add kernel! Add offloading call!\n" << std::endl;
  exit(1);
}

} // namespace anonymous

REGISTER_DISPATCH(add_stub, &add_kernel);

}} // namepsace at::native
