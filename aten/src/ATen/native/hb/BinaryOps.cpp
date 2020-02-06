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

#define define_kernel_op_binary(kernel_stub, kernel_fn) \
  void kernel_fn(TensorIterator& iter, Scalar alpha_scalar) { \
    hb::offload_op_binary(iter, alpha_scalar, #kernel_fn); \
  } \
    \
  REGISTER_DISPATCH(kernel_stub, &kernel_fn);

// hb_mc_<kernel> executes ${HB_DEVICE_DIR}/<kernel>.riscv to the device
define_kernel_op_binary(add_stub, hb_mc_add);

}} // namepsace at::native
