#include <ATen/native/HammerBladeOffload.h>

#include <bsg_manycore.h>
#include <bsg_manycore_tile.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_printing.h>

namespace at {
namespace native {

#define ALLOC_NAME "default_allocator"

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
    const Tensor& other, Scalar alpha, const char* kernel) {

  //===================================
  // Device initiation
  //

  int rc;
  hb_mc_dimension_t tg_dim = { .x = 2, .y = 2}; 
  hb_mc_dimension_t grid_dim = { .x = 1, .y = 1}; 
  hb_mc_device_t device;
  char base_path[] = "/mnt/users/ssd1/no_backup/bandhav/"
    "bsg_bladerunner/bsg_manycore/software/tensorlib/build/"; // HB_MC_TODO: dectect this in build flow
  char kernel_ext[] = ".riscv";

  //+---------------------------------
  // Init device and load the program

  char* bin_path = (char*) malloc(strlen(base_path) +
        strlen(kernel) + strlen(kernel_ext) + 1);
  strcpy(bin_path, base_path);
  strcpy(bin_path + strlen(base_path), kernel);
  strcpy(bin_path + strlen(base_path) + strlen(kernel), kernel_ext); 

  rc = hb_mc_device_init(&device, kernel, 0);
  if (rc != HB_MC_SUCCESS) { // HB_MC_TODO: replace these check with TORCH_CHECK
          bsg_pr_err("failed to initialize device.\n");
  }

  rc = hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0);
  if (rc != HB_MC_SUCCESS) { 
          bsg_pr_err("failed to initialize program.\n");
  }

  free(bin_path);


  //==================================
  // Data loading
  //

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

  rc = hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, kernel, 4, cuda_argv);
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
}

#undef ALLOC_NAME

} // namespace native
} // namespace at
