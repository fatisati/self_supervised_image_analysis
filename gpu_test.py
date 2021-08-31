import tensorflow as tf

gpu_available = tf.test.is_gpu_available()
print('gpu available', gpu_available)
is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
print('cude gpu available', is_cuda_gpu_available)
is_cuda_gpu_min_3 = tf.test.is_gpu_available(True, (3, 0))
print('min 3 cuda gpu available', is_cuda_gpu_min_3)

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
