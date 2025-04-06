# launch command: `python setup.py build_ext --inplace`
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

source = glob.glob('./build.cpp') + glob.glob('demo.cu')
# source = glob.glob('./*.cu')
setup(
    name='customOP',  # 自定义包名
    ext_modules=[
        CUDAExtension(
            name='my_cuda_ops',  #扩展模块名称，不一定和name相同, import my_cuda_ops
            sources=source,  #源文件
            # include_dirs=[
            #     'ops',  # 让编译器能找到 .h / .cuh 文件
            # ],
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)