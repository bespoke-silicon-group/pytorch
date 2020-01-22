#include <type_traits>
#include <ATen/native/BinaryOps.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>

#include <iostream>
#include <ATen/native/cpu/Loops.h>
#include <bsg_manycore_cuda.h>
#include <bsg_manycore_tile.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_printing.h>

namespace at {
namespace native {

DEFINE_DISPATCH(add_stub);
DEFINE_DISPATCH(sub_stub);
DEFINE_DISPATCH(mul_stub);
DEFINE_DISPATCH(div_stub);
DEFINE_DISPATCH(atan2_stub);
DEFINE_DISPATCH(bitwise_and_stub);
DEFINE_DISPATCH(bitwise_or_stub);
DEFINE_DISPATCH(bitwise_xor_stub);
DEFINE_DISPATCH(lshift_stub);
DEFINE_DISPATCH(rshift_stub);
DEFINE_DISPATCH(logical_and_stub);
DEFINE_DISPATCH(logical_or_stub);
DEFINE_DISPATCH(logical_xor_stub);
DEFINE_DISPATCH(lt_stub);
DEFINE_DISPATCH(le_stub);
DEFINE_DISPATCH(gt_stub);
DEFINE_DISPATCH(ge_stub);
DEFINE_DISPATCH(eq_stub);
DEFINE_DISPATCH(ne_stub);
DEFINE_DISPATCH(sigmoid_backward_stub);
DEFINE_DISPATCH(tanh_backward_stub);

Tensor& add_out(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha) {
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  alpha_check(iter.dtype(), alpha);
  add_stub(iter.device_type(), iter, alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == iter.output().dtype());
  return result;
}

Tensor add(const Tensor& self, const Tensor& other, Scalar alpha) {
  /************************************/
  /*      Original implementation     */
  /************************************/
  /*
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  alpha_check(iter.dtype(), alpha);
  add_stub(iter.device_type(), iter, alpha);
  return iter.output();
  */


  /************************************/
  /*      HammberBlade version        */
  /************************************/

  //===================================
  // Device initiation
  //

  #define ALLOC_NAME "default_allocator"
  int rc;
  hb_mc_dimension_t tg_dim = { .x = 2, .y = 2}; 
  hb_mc_dimension_t grid_dim = { .x = 1, .y = 1}; 
  hb_mc_device_t device;
  char bin_path[] = "/mnt/users/ssd1/no_backup/bandhav/"
    "bsg_bladerunner/bsg_manycore/software/torch/add/add.riscv"; // add(c, a, b, alpha): c <- a + alpha * b

  //+---------------------------------
  // Init device and load the program

  rc = hb_mc_device_init(&device, "add", 0);
  if (rc != HB_MC_SUCCESS) { 
          bsg_pr_err("failed to initialize device.\n");
  }

  rc = hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0);
  if (rc != HB_MC_SUCCESS) { 
          bsg_pr_err("failed to initialize program.\n");
  }


  //==================================
  // Data loading
  //

  typedef struct {
    uint32_t N;
    eva_t strides;
    eva_t data;
  } hb_mc_tensor_t;

  alpha_check(self.scalar_type(), alpha);

  Tensor self_c = self.toType(ScalarType::Float);
  Tensor other_c = other.toType(ScalarType::Float);
  float alpha_c = alpha.to<float>();

  // Device tensor pointers
  eva_t self_dev, other_dev, result_dev, alpha_dev;

  rc = hb_mc_device_malloc(&device, sizeof(hb_mc_tensor_t), &self_dev);
  if (rc != HB_MC_SUCCESS) { 
          bsg_pr_err("failed to allocate memory on device.\n");
  }

  rc = hb_mc_device_malloc(&device, sizeof(hb_mc_tensor_t), &other_dev);
  if (rc != HB_MC_SUCCESS) { 
          bsg_pr_err("failed to allocate memory on device.\n");
  }

  rc = hb_mc_device_malloc(&device, sizeof(hb_mc_tensor_t), &result_dev);
  if (rc != HB_MC_SUCCESS) { 
          bsg_pr_err("failed to allocate memory on device.\n");
  }

  rc = hb_mc_device_malloc(&device, sizeof(float), &alpha_dev);
  if (rc != HB_MC_SUCCESS) { 
          bsg_pr_err("failed to allocate memory on device.\n");
  }

  // Device data pointers
  eva_t self_dev_data, other_dev_data, result_dev_data;

  rc = hb_mc_device_malloc(&device, self.numel() * sizeof(float), &self_dev_data);
  if (rc != HB_MC_SUCCESS) { 
          bsg_pr_err("failed to allocate memory on device.\n");
  }

  rc = hb_mc_device_malloc(&device, other.numel() * sizeof(float), &other_dev_data);
  if (rc != HB_MC_SUCCESS) { 
          bsg_pr_err("failed to allocate memory on device.\n");
  }

  rc = hb_mc_device_malloc(&device, self.numel() * sizeof(float), &result_dev_data);
  if (rc != HB_MC_SUCCESS) { 
          bsg_pr_err("failed to allocate memory on device.\n");
  }

  // Copy raw tensor data
  //

  void *dst = (void *) ((intptr_t) self_dev_data);
  void *src = (void *) ((intptr_t) self_c.data_ptr());
  rc = hb_mc_device_memcpy (&device, dst, src, self_c.numel() * sizeof(float), HB_MC_MEMCPY_TO_DEVICE); 
  if (rc != HB_MC_SUCCESS) { 
          bsg_pr_err("failed to copy a to device.\n");
  }

  dst = (void *) ((intptr_t) other_dev_data);
  src = (void *) ((intptr_t) other_c.data_ptr());
  rc = hb_mc_device_memcpy (&device, dst, src, other_c.numel() * sizeof(float), HB_MC_MEMCPY_TO_DEVICE); 
  if (rc != HB_MC_SUCCESS) { 
          bsg_pr_err("failed to copy a to device.\n");
  }

  // Device strides pointers
  eva_t self_dev_stride, other_dev_stride, result_dev_stride;

  // Copy tensor args to device; this kernel doesn't need strides, so they are skipped.
  //

  // Kernel arguments
  hb_mc_tensor_t self_host = {.N = self.numel(), .strides = self_dev_stride, .data = self_dev_data};
  hb_mc_tensor_t other_host = {.N = other.numel(), .strides = other_dev_stride, .data = other_dev_data};
  hb_mc_tensor_t result_host = {.N = self.numel(), .strides = result_dev_stride, .data = result_dev_data};
  float alpha_host = alpha_c;

  dst = (void *) ((intptr_t) self_dev);
  src = (void *) ((intptr_t) &self_host);
  rc = hb_mc_device_memcpy (&device, dst, src, sizeof(hb_mc_tensor_t), HB_MC_MEMCPY_TO_DEVICE); 
  if (rc != HB_MC_SUCCESS) {
          bsg_pr_err("failed to copy a to device.\n");
  }

  dst = (void *) ((intptr_t) other_dev);
  src = (void *) ((intptr_t) &other_host);
  rc = hb_mc_device_memcpy (&device, dst, src, sizeof(hb_mc_tensor_t), HB_MC_MEMCPY_TO_DEVICE); 
  if (rc != HB_MC_SUCCESS) {
          bsg_pr_err("failed to copy a to device.\n");
  }

  dst = (void *) ((intptr_t) result_dev);
  src = (void *) ((intptr_t) &result_host);
  rc = hb_mc_device_memcpy (&device, dst, src, sizeof(hb_mc_tensor_t), HB_MC_MEMCPY_TO_DEVICE); 
  if (rc != HB_MC_SUCCESS) {
          bsg_pr_err("failed to copy a to device.\n");
  }

  dst = (void *) ((intptr_t) alpha_dev);
  src = (void *) ((intptr_t) &alpha_host);
  rc = hb_mc_device_memcpy (&device, dst, src, sizeof(float), HB_MC_MEMCPY_TO_DEVICE); 
  if (rc != HB_MC_SUCCESS) {
          bsg_pr_err("failed to copy a to device.\n");
  }


  //===================================================
  // Kernel Offload
  //

  const uint32_t cuda_argv[4] = {result_dev, self_dev, other_dev, alpha_dev};

  rc = hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "add", 4, cuda_argv);
  if (rc != HB_MC_SUCCESS) { 
          bsg_pr_err("failed to initialize grid.\n");
  }

  rc = hb_mc_device_tile_groups_execute(&device);
  if (rc != HB_MC_SUCCESS) { 
          bsg_pr_err("failed to execute tile groups.\n");
  }

  //===================================================
  // Copy the result
  //

  // Result tensor
  Tensor result = at::empty({0}, self.options());
  result.resize_(self.sizes());

  // Copy the result from device memory
  src = (void*) ((intptr_t) result_dev_data);
  dst = (void*) ((intptr_t) result.data_ptr());
  rc = hb_mc_device_memcpy (&device, dst, src, result.numel() * sizeof(float), HB_MC_MEMCPY_TO_HOST); 
  if (rc != HB_MC_SUCCESS) { 
    bsg_pr_err("failed to copy c from device.\n");
  }    

  // Freeze tiles and cleanup memory manager
  rc = hb_mc_device_finish(&device); 
  if (rc != HB_MC_SUCCESS) { 
          bsg_pr_err("failed to de-initialize device.\n");
  }

  return result;
}

Tensor& add_(Tensor& self, const Tensor& other, Scalar alpha) {
  return native::add_out(self, self, other, alpha);
}

Tensor& div_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  div_stub(iter.device_type(), iter);
  return result;
}

Tensor div(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  div_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& div_(Tensor& self, const Tensor& other) {
  return native::div_out(self, self, other);
}

Tensor truncate(const Tensor& tensor) {
  if (tensor.is_floating_point()) {
    return tensor.trunc();
  }
  return tensor;
}

Tensor floor_divide(const Tensor& input, const Tensor& other) {
  Tensor out = input / other;
  return truncate(out);
}

Tensor floor_divide(const Tensor& input, Scalar other) {
  Tensor out = input / other;
  return truncate(out);
}

Tensor& mul_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  mul_stub(iter.device_type(), iter);
  return result;
}

Tensor mul(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  mul_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& mul_(Tensor& self, const Tensor& other) {
  return native::mul_out(self, self, other);
}

Tensor& sub_out(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha) {
  sub_check(self, other);
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  alpha_check(iter.dtype(), alpha);
  sub_stub(iter.device_type(), iter, alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == iter.output().dtype());
  return result;
}

Tensor sub(const Tensor& self, const Tensor& other, Scalar alpha) {
  sub_check(self, other);
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  alpha_check(iter.dtype(), alpha);
  sub_stub(iter.device_type(), iter, alpha);
  return iter.output();
}

Tensor& sub_(Tensor& self, const Tensor& other, Scalar alpha) {
  return native::sub_out(self, self, other, alpha);
}

Tensor& sigmoid_backward_out(Tensor& result, const Tensor& grad_output, const Tensor& output) {
  auto iter = TensorIterator::binary_op(result, grad_output, output);
  sigmoid_backward_stub(iter.device_type(), iter);
  return result;
}

Tensor sigmoid_backward(const Tensor& grad_output, const Tensor& output) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, grad_output, output);
  sigmoid_backward_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& tanh_backward_out(Tensor& result, const Tensor& grad_output, const Tensor& output) {
  auto iter = TensorIterator::binary_op(result, grad_output, output);
  tanh_backward_stub(iter.device_type(), iter);
  return result;
}

Tensor tanh_backward(const Tensor& grad_output, const Tensor& output) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, grad_output, output);
  tanh_backward_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor rsub(const Tensor& self, const Tensor& other, Scalar alpha) {
  return native::sub(other, self, alpha);
}

Tensor& atan2_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other);
  atan2_stub(iter.device_type(), iter);
  return result;
}

Tensor atan2(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  return native::atan2_out(result, self, other);
}

Tensor& atan2_(Tensor& self, const Tensor& other) {
  return native::atan2_out(self, self, other);
}

// These are still needed because we don't have C++ conversions from number
// types (int, float, etc.) to Tensor (only to Scalar). They're not exposed
// to Python.

static Tensor wrapped_scalar_tensor(Scalar scalar) {
  auto tensor = scalar_to_tensor(scalar);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

static void check_convert(Scalar scalar, ScalarType scalarType) {
  // Validate that is possible to convert scalar to tensor dtype without overflow
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half, scalarType, "check_convert", [&]{
    scalar.to<scalar_t>();
  });
}

static Tensor wrapped_scalar_tensor_and_check_convert(Scalar scalar, Tensor tensor) {
  check_convert(scalar, tensor.scalar_type());
  return wrapped_scalar_tensor(scalar);
}

Tensor add(const Tensor& self, Scalar other, Scalar alpha) {
  return native::add(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& add_(Tensor& self, Scalar other, Scalar alpha) {
  return native::add_(self, wrapped_scalar_tensor(other), alpha);
}

// WARNING: There doesn't appear to be any testing for this function
// with sparse self input.
Tensor div(const Tensor& self, Scalar other) {
  return self.div(wrapped_scalar_tensor(other)); // redispatch!
}

// WARNING: This function, with a sparse self, is currently only
// exercised by DistributedDataParallelTest.test_sparse_gradients
// (you need to exercise it from C++, because this overload is never
// used for Python)
Tensor& div_(Tensor& self, Scalar other) {
  return self.div_(wrapped_scalar_tensor(other)); // redispatch!
}

Tensor mul(const Tensor& self, Scalar other) {
  return native::mul(self, wrapped_scalar_tensor(other));
}

Tensor& mul_(Tensor& self, Scalar other) {
  return native::mul_(self, wrapped_scalar_tensor(other));
}

Tensor sub(const Tensor& self, Scalar other, Scalar alpha) {
  return native::sub(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& sub_(Tensor& self, Scalar other, Scalar alpha) {
  return native::sub_(self, wrapped_scalar_tensor(other), alpha);
}

Tensor rsub(const Tensor& self, Scalar other, Scalar alpha) {
  return native::rsub(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& bitwise_and_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  bitwise_and_stub(iter.device_type(), iter);
  return result;
}

Tensor bitwise_and(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  at::bitwise_and_out(result, self, other);
  return result;
}

Tensor& bitwise_and_(Tensor& self, const Tensor& other) {
  return at::bitwise_and_out(self, self, other);
}

Tensor& bitwise_and_out(Tensor& result, const Tensor& self, Scalar other) {
  return at::bitwise_and_out(result, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_and(const Tensor& self, Scalar other) {
  Tensor result = at::empty({0}, self.options());
  return at::bitwise_and_out(result, self, other);
}

Tensor& bitwise_and_(Tensor& self, Scalar other) {
  return at::bitwise_and_out(self, self, other);
}

// Legacy and interfaces. They are aliased to bitwise_and* functions
Tensor __and__(const Tensor& self, const Tensor& other) {
  return at::bitwise_and(self, other);
}

Tensor __and__(const Tensor& self, Scalar other) {
  return at::bitwise_and(self, other);
}

Tensor& __iand__(Tensor& self, const Tensor& other) {
  return self.bitwise_and_(other);
}

Tensor& __iand__(Tensor& self, Scalar other) {
  return self.bitwise_and_(other);
}

Tensor& bitwise_or_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  bitwise_or_stub(iter.device_type(), iter);
  return result;
}

Tensor bitwise_or(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  at::bitwise_or_out(result, self, other);
  return result;
}

Tensor& bitwise_or_(Tensor& self, const Tensor& other) {
  return at::bitwise_or_out(self, self, other);
}

Tensor& bitwise_or_out(Tensor& result, const Tensor& self, Scalar other) {
  return at::bitwise_or_out(result, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_or(const Tensor& self, Scalar other) {
  Tensor result = at::empty({0}, self.options());
  return at::bitwise_or_out(result, self, other);
}

Tensor& bitwise_or_(Tensor& self, Scalar other) {
  return at::bitwise_or_out(self, self, other);
}

// Legacy or interfaces. They are aliased to bitwise_or* functions
Tensor __or__(const Tensor& self, const Tensor& other) {
  return at::bitwise_or(self, other);
}

Tensor __or__(const Tensor& self, Scalar other) {
  return at::bitwise_or(self, other);
}

Tensor& __ior__(Tensor& self, const Tensor& other) {
  return self.bitwise_or_(other);
}

Tensor& __ior__(Tensor& self, Scalar other) {
  return self.bitwise_or_(other);
}

Tensor& bitwise_xor_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  bitwise_xor_stub(iter.device_type(), iter);
  return result;
}

Tensor bitwise_xor(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  at::bitwise_xor_out(result, self, other);
  return result;
}

Tensor& bitwise_xor_(Tensor& self, const Tensor& other) {
  return at::bitwise_xor_out(self, self, other);
}

Tensor& bitwise_xor_out(Tensor& result, const Tensor& self, Scalar other) {
  return at::bitwise_xor_out(result, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_xor(const Tensor& self, Scalar other) {
  Tensor result = at::empty({0}, self.options());
  return at::bitwise_xor_out(result, self, other);
}

Tensor& bitwise_xor_(Tensor& self, Scalar other) {
  return at::bitwise_xor_out(self, self, other);
}

// Legacy xor interfaces. They are aliased to bitwise_xor* functions
Tensor __xor__(const Tensor& self, const Tensor& other) {
  return at::bitwise_xor(self, other);
}

Tensor __xor__(const Tensor& self, Scalar other) {
  return at::bitwise_xor(self, other);
}

Tensor& __ixor__(Tensor& self, const Tensor& other) {
  return self.bitwise_xor_(other);
}

Tensor& __ixor__(Tensor& self, Scalar other) {
  return self.bitwise_xor_(other);
}

Tensor __lshift__(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  lshift_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor __lshift__(const Tensor& self, Scalar other) { 
  Tensor result;
  auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
  auto iter = TensorIterator::binary_op(result, self, wrapper);
  lshift_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& __ilshift__(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(self, self, other);
  lshift_stub(iter.device_type(), iter);
  return self;
}

Tensor& __ilshift__(Tensor& self, Scalar other) {
  auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
  auto iter = TensorIterator::binary_op(self, self, wrapper);
  lshift_stub(iter.device_type(), iter);
  return self;
}

Tensor __rshift__(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  rshift_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor __rshift__(const Tensor& self, Scalar other) { 
  Tensor result;
  auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
  auto iter = TensorIterator::binary_op(result, self, wrapper);
  rshift_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& __irshift__(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(self, self, other);
  rshift_stub(iter.device_type(), iter);
  return self;
}

Tensor& __irshift__(Tensor& self, Scalar other) {
  auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
  auto iter = TensorIterator::binary_op(self, self, wrapper);
  rshift_stub(iter.device_type(), iter);
  return self;
}

template <typename Stub>
Tensor& comparison_op_out(Tensor& result, const Tensor& self, const Tensor& other, Stub& stub) {
  // Validate that is possible to convert zero-dim tensor's dtype to other dtype without overflow
  if (self.scalar_type() != other.scalar_type()) {
    if (self.dim() != 0 && other.dim() == 0) {
      check_convert(other.item(), self.scalar_type());
    } else if (self.dim() == 0 && other.dim() != 0) {
      check_convert(self.item(), other.scalar_type());
    }
  }
  auto iter = TensorIterator::comparison_op(result, self, other, /*check_mem_overlap=*/true);
  stub(iter.device_type(), iter);
  return result;
}

template <typename OutImpl>
Tensor comparison_op(const Tensor& self, const Tensor& other, OutImpl& out_impl) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return out_impl(result, self, other);
}

// To avoid overflow during type promotion we will check that both dtypes of self and other are same
template <typename OutImpl>
Tensor& comparison_op_(Tensor& self, const Tensor& other, OutImpl& out_impl) {
  TORCH_CHECK(self.dtype() == other.dtype(),
              "Expected object of scalar type ", self.dtype(), " but got scalar type ",
              other.dtype(), " for argument 'other'");
  return out_impl(self, self, other);
}

// validates that is possible to convert Scalar other to self's dtype without overflow.
// This behavior is unique to comparison ops; arithmetic operations don't do this.
// In the future, we should reconsider this inconsistency and decide if we want to add the same check to arithmetic ops.
template <typename OutImpl>
Tensor& comparison_op_out(Tensor& result, const Tensor& self, Scalar other, OutImpl& out_impl) {
  return out_impl(result, self, wrapped_scalar_tensor_and_check_convert(other, self));
}

template <typename OutImpl>
Tensor comparison_op(const Tensor& self, Scalar other, OutImpl& out_impl) {
  return comparison_op(self, wrapped_scalar_tensor_and_check_convert(other, self), out_impl);
}

template <typename OutImpl>
Tensor& comparison_op_(Tensor& self, Scalar other, OutImpl& out_impl) {
  return out_impl(self, self, wrapped_scalar_tensor_and_check_convert(other, self));
}

// We need explicit cast to OutFunc because each *_out func is overloaded twice. Without An explicit cast, merely
// referring to *_out function is ambiguious.
using OutFunc = std::add_const<Tensor&(&)(Tensor&, const Tensor&, const Tensor&)>::type;

Tensor& lt_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, lt_stub); }
Tensor lt(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::lt_out)); }
Tensor& lt_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::lt_out)); }
Tensor& lt_out(Tensor& result, const Tensor& self, Scalar other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::lt_out)); }
Tensor lt(const Tensor& self, Scalar other) { return comparison_op(self, other, static_cast<OutFunc>(at::lt_out)); }
Tensor& lt_(Tensor& self, Scalar other) { return comparison_op_(self, other, static_cast<OutFunc>(at::lt_out)); }

Tensor& le_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, le_stub); }
Tensor le(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::le_out)); }
Tensor& le_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::le_out)); }
Tensor& le_out(Tensor& result, const Tensor& self, Scalar other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::le_out)); }
Tensor le(const Tensor& self, Scalar other) { return comparison_op(self, other, static_cast<OutFunc>(at::le_out)); }
Tensor& le_(Tensor& self, Scalar other) { return comparison_op_(self, other, static_cast<OutFunc>(at::le_out)); }

Tensor& gt_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, gt_stub); }
Tensor gt(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::gt_out)); }
Tensor& gt_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::gt_out)); }
Tensor& gt_out(Tensor& result, const Tensor& self, Scalar other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::gt_out)); }
Tensor gt(const Tensor& self, Scalar other) { return comparison_op(self, other, static_cast<OutFunc>(at::gt_out)); }
Tensor& gt_(Tensor& self, Scalar other) { return comparison_op_(self, other, static_cast<OutFunc>(at::gt_out)); }

Tensor& ge_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, ge_stub); }
Tensor ge(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::ge_out)); }
Tensor& ge_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::ge_out)); }
Tensor& ge_out(Tensor& result, const Tensor& self, Scalar other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::ge_out)); }
Tensor ge(const Tensor& self, Scalar other) { return comparison_op(self, other, static_cast<OutFunc>(at::ge_out)); }
Tensor& ge_(Tensor& self, Scalar other) { return comparison_op_(self, other, static_cast<OutFunc>(at::ge_out)); }

Tensor& eq_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, eq_stub); }
Tensor eq(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::eq_out)); }
Tensor& eq_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::eq_out)); }
Tensor& eq_out(Tensor& result, const Tensor& self, Scalar other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::eq_out)); }
Tensor eq(const Tensor& self, Scalar other) { return comparison_op(self, other, static_cast<OutFunc>(at::eq_out)); }
Tensor& eq_(Tensor& self, Scalar other) { return comparison_op_(self, other, static_cast<OutFunc>(at::eq_out)); }

Tensor& ne_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, ne_stub); }
Tensor ne(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::ne_out)); }
Tensor& ne_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::ne_out)); }
Tensor& ne_out(Tensor& result, const Tensor& self, Scalar other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::ne_out)); }
Tensor ne(const Tensor& self, Scalar other) { return comparison_op(self, other, static_cast<OutFunc>(at::ne_out)); }
Tensor& ne_(Tensor& self, Scalar other) { return comparison_op_(self, other, static_cast<OutFunc>(at::ne_out)); }

Tensor& logical_and_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, logical_and_stub); }
Tensor logical_and(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_and_out)); }
Tensor& logical_and_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_and_out)); }
Tensor& logical_and_out(Tensor& result, const Tensor& self, Scalar other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::logical_and_out)); }
Tensor logical_and(const Tensor& self, Scalar other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_and_out)); }
Tensor& logical_and_(Tensor& self, Scalar other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_and_out)); }

Tensor& logical_or_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, logical_or_stub); }
Tensor logical_or(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_or_out)); }
Tensor& logical_or_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_or_out)); }
Tensor& logical_or_out(Tensor& result, const Tensor& self, Scalar other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::logical_or_out)); }
Tensor logical_or(const Tensor& self, Scalar other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_or_out)); }
Tensor& logical_or_(Tensor& self, Scalar other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_or_out)); }

Tensor& logical_xor_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, logical_xor_stub); }
Tensor logical_xor(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_xor_out)); }
Tensor& logical_xor_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_xor_out)); }
Tensor& logical_xor_out(Tensor& result, const Tensor& self, Scalar other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::logical_xor_out)); }
Tensor logical_xor(const Tensor& self, Scalar other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_xor_out)); }
Tensor& logical_xor_(Tensor& self, Scalar other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_xor_out)); }
}
}  // namespace at
