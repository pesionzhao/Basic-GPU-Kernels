from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

source = glob.glob('./build.cpp') + glob.glob('./transpose.cu')
# source = glob.glob('./*.cu')
setup(
    name='my_cuda_ops',  # 自定义包名
    ext_modules=[
        CUDAExtension(
            name='my_cuda_ops',
            sources=source,
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)