import ctypes
import time
import numpy as np
import multiprocessing as mp
import dummy
from PIL import Image
from image_filter import apply_filter_to_row
import tqdm

# region Filters

filter1 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])

filter2 = np.array([[0.5, 0, -0.5]])

filter3 = np.array([[0.5], [0], [-0.5]])

filter4 = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

filter5 = np.array([
    [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
    [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
    [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
    [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
    [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]
])

# endregion


# region Helpers

def allocate_shared_memory(shape, lock=False):
    return mp.Array(ctypes.c_uint8, shape[0] * shape[1] * shape[2], lock=lock)


def to_np_array(mp_array, shape=None):
    array = np.frombuffer(mp_array.get_obj() if 'get_obj' in mp_array.__dict__ else mp_array, dtype=np.uint8)
    return array.reshape(shape) if shape else array


def create_parallel_filter_pool(src_img_shape, image_memory, image_filters):
    init_args = (src_img_shape, image_memory, image_filters)
    return mp.Pool(mp.cpu_count(), initializer=parallel_filter_initializer, initargs=init_args)


# endregion


# region Parallel Filter Implementation

def parallel_filter_initializer(src_img_shape, image_memory, image_filters):
    dummy.shared_arrays = [
        v for v in image_memory
    ]
    dummy.np_arrays = [
        to_np_array(v, shape=src_img_shape) for v in image_memory
    ]
    dummy.filters = [
        v for v in image_filters
    ]


def parallel_filter_step(args):
    # Get input data
    row, input_index, output_index, filter_index = args

    # Get corresponding arrays/filter from global memory
    src_img = dummy.np_arrays[input_index]
    dst_img = dummy.np_arrays[output_index]
    flt = dummy.filters[filter_index]

    # Apply filter and update shared memory
    dst_img[row, :, :] = apply_filter_to_row(row, src_img, flt)


# endregion


# region Filter functions

def serial_filter(output_img, src_img, flt):
    # All processes use the same input, output and filter
    input_index, output_index = 0, 1
    filter_index = 0

    image_memory = [
        allocate_shared_memory(src_img.shape, lock=False),  # Input
        allocate_shared_memory(src_img.shape, lock=False)   # Output
    ]

    # Copy image into shared memory
    to_np_array(image_memory[input_index], shape=src_img.shape)[:] = src_img

    parallel_filter_initializer(src_img.shape, image_memory, [flt])
    for i in tqdm.tqdm(range(src_img.shape[0])):
        parallel_filter_step((i, input_index, output_index, filter_index))

    output_img[:] = to_np_array(image_memory[output_index], shape=src_img.shape)


def parallel_filter(output_img, src_img, flt):
    # All processes use the same input, output and filter
    input_index, output_index = 0, 1
    filter_index = 0

    image_memory = [
        allocate_shared_memory(src_img.shape, lock=False),  # Input
        allocate_shared_memory(src_img.shape, lock=False)   # Output
    ]

    # Copy image into shared memory
    to_np_array(image_memory[input_index], shape=src_img.shape)[:] = src_img

    with create_parallel_filter_pool(src_img.shape, image_memory, [flt]) as pool:
        pool.map_async(parallel_filter_step, [
            (i, input_index, output_index, filter_index) for i in range(src_img.shape[0])
        ]).wait()

    output_img[:] = to_np_array(image_memory[output_index], shape=src_img.shape)


def double_parallel_filter(output_img_1, output_img_2, src_img, flt_1, flt_2):
    # All processes use the same input, output and filter
    input_index, output_1_index, output_2_index = 0, 1, 2

    image_memory = [
        allocate_shared_memory(src_img.shape, lock=False),  # Input
        allocate_shared_memory(src_img.shape, lock=False),  # Output1
        allocate_shared_memory(src_img.shape, lock=False)   # Output2
    ]

    # Copy image into shared memory
    to_np_array(image_memory[input_index], shape=src_img.shape)[:] = src_img[:]

    with create_parallel_filter_pool(src_img.shape, image_memory, [flt_1, flt_2]) as pool:
        maps = []
        for filter_index in range(2):
            maps.append(pool.map_async(parallel_filter_step, [
                (i, input_index, filter_index + 1, filter_index) for i in range(src_img.shape[0])
            ]))
        for m in maps:
            m.wait()

    output_img_1[:] = to_np_array(image_memory[output_1_index], shape=src_img.shape)
    output_img_2[:] = to_np_array(image_memory[output_2_index], shape=src_img.shape)

# endregion


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img_name = 'hand-x-ray.jpg'  # 'hand-x-ray.jpg'  # 'cloudsonjupi.jpg'  # 'chess.jpg'  # 'cloudsonjupi.jpg'
    src = np.array(Image.open(img_name))
    results = []

    # Single
    # for i, fl in enumerate([filter1, filter2, filter3, filter4, filter5]):
    #     dst = np.zeros(shape=src.shape, dtype=np.uint8)
    #     print('Filter %d' % (i + 1))
    #     start_time = time.time()
    #     parallel_filter(dst, src, fl)
    #     end_time = time.time()
    #     print("Time: %.4fs" % (end_time - start_time))
    #     results.append(dst)

    # Double
    dst1 = np.zeros(shape=src.shape, dtype=np.uint8)
    dst2 = np.zeros(shape=src.shape, dtype=np.uint8)
    start_time = time.time()
    double_parallel_filter(dst1, dst2, src, filter2, filter3)
    end_time = time.time()
    print("Time: %.4fs" % (end_time - start_time))
    results.extend([dst1, dst2])

    # Plot
    plt.figure()
    plt.imshow(src)
    for result in results:
        plt.figure()
        plt.imshow(result)
    plt.show()
