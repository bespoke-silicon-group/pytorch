#include <ATen/ATen.h>
#include <bsg_manycore_cuda.h>

namespace at {
namespace native {

/**
 * HammerBlade Tensor Struct
 * 
 * This struct defines the tensor layout on HB device. 
 * HB kernel offloading routines cast PyTorch's tensors 
 * to this format before loading and launching a kernel. 
 * The layout of this struct matches that of the C struct 
 * defined in HB device runtime.
 */
typedef struct {
  uint32_t N;    // Number of elements in the tensor 
  uint32_t dims; // Number of dimensions 
  eva_t strides;  // Pointer to stride vector; number of strides = dims
  eva_t data;    // Pointer to raw data
} hb_mc_tensor_t;


/**
 * HammberBlade binary operator offloading routine:
 *
 *   - Initializes the device.
 *   - Loads the binary.
 *   - Allocates device buffer for arguments and result.
 *   - Loads arguments and executes the kernel.
 *   - Copies the result tensor to host buffer and cleans memory manager. 
 *   - Returns the result tensor. 
 * 
 * @param[in] result Reference to result tensor
 * @param[in] self   Input tensor 1
 * @param[in] other  Input tensor 2
 * @param[in] alpha  Scalar multiplication factor for tensor 2
 * @param[in] kernel Name of the compute kernel to be launched
 * @return result tensor
 */
void hb_mc_offload_op_binary(Tensor& result, const Tensor& self, 
    const Tensor& other, Scalar alpha, const char* kernel);

} // namespace native
} // namespace at
