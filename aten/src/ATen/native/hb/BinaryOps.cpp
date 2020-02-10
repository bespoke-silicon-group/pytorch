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

#define define_kernel_op_binary_alpha(kernel_stub, kernel_fn) \
  void kernel_fn(TensorIterator& iter, Scalar alpha_scalar) { \
    hb::offload_op_binary(iter, alpha_scalar, #kernel_fn); \
  } \
    \
  REGISTER_DISPATCH(kernel_stub, &kernel_fn);

#define define_kernel_op_binary(kernel_stub, kernel_fn) \
  void kernel_fn(TensorIterator& iter) { \
    hb::offload_op_binary(iter, 1.0, #kernel_fn); \
  } \
    \
  REGISTER_DISPATCH(kernel_stub, &kernel_fn);


// HammberBlade offloading:
//
// PyTorch kernel stubs (eg., add_stub) are device agnostic lower level
// kernels that are invoked by PyTorch's c++ backend. These stubs furthur
// dispatch device specific kernels depending on the tensor location. "add_stub"
// for eg., would dispatch "add_kernel_cuda" in ../cuda/BinaryArithemeticKernel.cu
// if add_stub's arguments are cuda tensors. Similarly, add_stub would dispatch
// "add_kernel" defined in ../cpu/BinaryOpsKernel.cpp, if add_stub's arguments
// are cpu tensors. Following single line definitions substitute cpu dispatch
// destinations (add_kernel for eg.,) with HammerBlade version (hb_mc_add).
// Meaning, these definitions offload CPU tensor computations to HammerBlade.
//
// These definitions require corresponding device kernel binaries in ${HB_DEVICE_DIR}:
// hb_mc_<kernel> executes ${HB_DEVICE_DIR}/<kernel>.riscv on the device.
define_kernel_op_binary_alpha(add_stub, hb_mc_add);
define_kernel_op_binary_alpha(sub_stub, hb_mc_sub);
define_kernel_op_binary(mul_stub, hb_mc_mul);
define_kernel_op_binary(div_stub, hb_mc_div);

}} // namepsace at::native
