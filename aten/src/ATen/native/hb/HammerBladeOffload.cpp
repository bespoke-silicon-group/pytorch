#include <ATen/native/hb/HammerBladeOffload.h>

#include <bsg_manycore.h>
#include <bsg_manycore_tile.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_printing.h>

namespace at {
namespace native {
namespace hb {

#define ALLOC_NAME "default_allocator"

#define PATH(a) str(a)
#define str(a) #a "/"
static char hb_mc_kernel_base_path[] = PATH(HB_DEVICE_DIR);
#undef str
#undef PATH

static void init_device_kernel(hb_mc_device_t& device, const char* kernel) {
  int rc;

  char kernel_ext[] = ".riscv";

  //+---------------------------------
  // Init device and load the program

  char* bin_path = (char*) malloc(strlen(hb_mc_kernel_base_path) +
        strlen(kernel) + strlen(kernel_ext) + 1);
  strcpy(bin_path, hb_mc_kernel_base_path);
  strcpy(bin_path + strlen(hb_mc_kernel_base_path), kernel);
  strcpy(bin_path + strlen(hb_mc_kernel_base_path) + strlen(kernel), kernel_ext); 

  rc = hb_mc_device_init(&device, kernel, 0);
  if (rc != HB_MC_SUCCESS) { // HB_MC_TODO: replace these check with TORCH_CHECK
    bsg_pr_err("failed to initialize device.\n");
  }

  rc = hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0);
  if (rc != HB_MC_SUCCESS) { 
    bsg_pr_err("failed to initialize program.\n");
  }

  free(bin_path);
}

static eva_t create_device_tensor(hb_mc_device_t& device, uint32_t N, 
    uint32_t dims, const int64_t* strides, const void* data, bool copy) {
  int rc;
  eva_t tensor, tensor_strides, tensor_data;

  // allocate memory for tensor struct
  rc = hb_mc_device_malloc(&device, sizeof(hb_mc_tensor_t), &tensor);
  if (rc != HB_MC_SUCCESS) { 
          bsg_pr_err("failed to allocate memory on device.\n");
  }

  // allocate memory for strides
  rc = hb_mc_device_malloc(&device, dims * sizeof(uint32_t), &tensor_strides);
  if (rc != HB_MC_SUCCESS) { 
          bsg_pr_err("failed to allocate memory on device.\n");
  }

  // allocate memory for data
  rc = hb_mc_device_malloc(&device, N * sizeof(float), &tensor_data);
  if (rc != HB_MC_SUCCESS) { 
          bsg_pr_err("failed to allocate memory on device.\n");
  }

  // tensor struct on host
  hb_mc_tensor_t tensor_host = {
    .N = N, 
    .dims = dims,
    .strides = tensor_strides, 
    .data = tensor_data,
  };

  // copy tensor struct
  void* dst = (void *) ((intptr_t) tensor);
  void* src = (void *) ((intptr_t) &tensor_host);
  rc = hb_mc_device_memcpy (&device, dst, src, sizeof(hb_mc_tensor_t), HB_MC_MEMCPY_TO_DEVICE); 
  if (rc != HB_MC_SUCCESS) {
          bsg_pr_err("failed to copy a to device.\n");
  }

  if(copy) {
    // copy strides
    dst = (void *) ((intptr_t) tensor_strides);
    src = (void *) ((intptr_t) strides);
    rc = hb_mc_device_memcpy (&device, dst, src, N * sizeof(float), HB_MC_MEMCPY_TO_DEVICE); 
    if (rc != HB_MC_SUCCESS) {
            bsg_pr_err("failed to copy a to device.\n");
    }

    // copy data
    dst = (void *) ((intptr_t) tensor_data);
    src = (void *) ((intptr_t) data);
    rc = hb_mc_device_memcpy (&device, dst, src, N * sizeof(float), HB_MC_MEMCPY_TO_DEVICE); 
    if (rc != HB_MC_SUCCESS) {
            bsg_pr_err("failed to copy a to device.\n");
    }
  }

  return tensor;
}

static hb_mc_tensor_t get_device_tensor(hb_mc_device_t& device, eva_t dtensor_d) {
  int rc;
  hb_mc_tensor_t dtensor_h; // device tensor on host

  // Copy tensor struct from device to host
  void* src = (void*) ((intptr_t) dtensor_d);
  void* dst = (void*) ((intptr_t) &dtensor_h);
  rc = hb_mc_device_memcpy (&device, dst, src, sizeof(hb_mc_tensor_t), HB_MC_MEMCPY_TO_HOST); 
  if (rc != HB_MC_SUCCESS) { 
    bsg_pr_err("failed to copy from device.\n");
  }

  // host tensor
  hb_mc_tensor_t htensor = {
    .N = dtensor_h.N,
    .dims = dtensor_h.dims,
    .strides = (eva_t) ((intptr_t) malloc(dtensor_h.dims * sizeof(uint32_t))),
    .data = (eva_t) ((intptr_t) malloc(dtensor_h.N * sizeof(float))),
  };

  // copy strides array
  src = (void*) ((intptr_t) dtensor_h.strides);
  dst = (void*) ((intptr_t) htensor.strides);
  rc = hb_mc_device_memcpy (&device, dst, src, dtensor_h.dims * sizeof(uint32_t), HB_MC_MEMCPY_TO_HOST); 
  if (rc != HB_MC_SUCCESS) { 
    bsg_pr_err("failed to copy from device.\n");
  }

  // copy data
  src = (void*) ((intptr_t) dtensor_h.data);
  dst = (void*) ((intptr_t) htensor.data);
  rc = hb_mc_device_memcpy (&device, dst, src, dtensor_h.N * sizeof(float), HB_MC_MEMCPY_TO_HOST); 
  if (rc != HB_MC_SUCCESS) { 
    bsg_pr_err("failed to copy from device.\n");
  }

  return htensor;
}

static eva_t create_device_scalar(hb_mc_device_t& device, float alpha) {
  int rc;
  eva_t alpha_d;

  rc = hb_mc_device_malloc(&device, sizeof(float), &alpha_d);
  if (rc != HB_MC_SUCCESS) { 
    bsg_pr_err("failed to copy from device.\n");
  }

  void* src = (void*) ((intptr_t) &alpha);
  void* dst = (void*) ((intptr_t) alpha_d);
  rc = hb_mc_device_memcpy (&device, dst, src, sizeof(float), HB_MC_MEMCPY_TO_DEVICE); 
  if (rc != HB_MC_SUCCESS) { 
    bsg_pr_err("failed to copy from device.\n");
  }

  return alpha_d;
}

static float get_device_scalar(hb_mc_device_t& device, eva_t alpha_d) {
  int rc;
  float alpha;

  void* src = (void*) ((intptr_t) alpha_d);
  void* dst = (void*) ((intptr_t) &alpha);
  rc = hb_mc_device_memcpy (&device, dst, src, sizeof(float), HB_MC_MEMCPY_TO_HOST); 
  if (rc != HB_MC_SUCCESS) { 
    bsg_pr_err("failed to copy from device.\n");
  }

  return alpha;
}

static void execute_kernel(hb_mc_device_t& device, 
    const char* kernel, std::vector<eva_t> args) {
  int rc;
  hb_mc_dimension_t tg_dim = { .x = 2, .y = 2}; 
  hb_mc_dimension_t grid_dim = { .x = 1, .y = 1}; 

  uint32_t* cuda_argv = (uint32_t*) malloc(args.size() * sizeof(eva_t));
  if(!cuda_argv) {
    bsg_pr_err("Falied to allocate cuda_argv!\n");
  }

  for(int i=0; i<args.size(); ++i) {
    cuda_argv[i] = args[i];
  }

  rc = hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, kernel, 
        args.size(), (const uint32_t*) cuda_argv);
  if (rc != HB_MC_SUCCESS) {
    bsg_pr_err("failed to initialize grid.\n");
  }

  rc = hb_mc_device_tile_groups_execute(&device);
  if (rc != HB_MC_SUCCESS) {
    bsg_pr_err("failed to execute tile groups.\n");
  }

  free(cuda_argv);
}

static void freeze_device(hb_mc_device_t& device) {
  // Freeze tiles and cleanup memory manager
  int rc;
  rc = hb_mc_device_finish(&device); 
  if (rc != HB_MC_SUCCESS) { 
    bsg_pr_err("failed to de-initialize device.\n");
  }
}


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
 * @param[in] iter   Tensor iterator for binary op
 * @param[in] alpha  Scalar multiplication factor for tensor 2
 * @param[in] kernel Name of the compute kernel to be launched
 */
void offload_op_binary(TensorIterator& iter, Scalar alpha, const char* kernel) {
  kernel += strlen("hb_mc_"); // remove prepending tag in the kernel name
  bsg_pr_info("Launching %s kernel...\n", kernel);

  // Initialize the device and load the kernel binary
  hb_mc_device_t device;
  init_device_kernel(device, kernel);

  iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
    // Device pointers to tensors on the device
    std::vector<eva_t> device_args;

    // Allocate device tensors and copy the data
    for(int i=0; i<iter.ntensors(); ++i) {
      // Iterate over all tensors: a, b and result, to create
      // corresponding tensors on the device. Last argument
      // below tells the function whether to copy data to tensors.
      // We don't copy data to result tensor, which is in index 0.
      eva_t device_arg = create_device_tensor(device, n, iter.ndim(), 
          &strides[i], data[i], i!=0);
      device_args.push_back(device_arg);
    }
    device_args.push_back(create_device_scalar(device, alpha.to<float>()));

    execute_kernel(device, kernel, device_args);

    // Populate the result tensor
    hb_mc_tensor_t result_tensor = get_device_tensor(device, device_args[0]);
    memcpy(data[0], ((char*) ((intptr_t) result_tensor.data)), n * strides[0]);
  });

  freeze_device(device);
  iter.cast_outputs();
}

#undef ALLOC_NAME

} // namespace hb
} // namespace native
} // namespace at
