from distutils.core import setup, Extension


module = Extension('blockmatch',
                    sources = ['blockmatch.cpp'],
                    include_dirs = [],
                    libraries = [],
                    library_dirs = ['/usr/local/lib'],                    
                    extra_compile_args=['-std=c++11'])
 
setup(name = 'blockmatch',
      version = '1.0',
      description = 'blockmatching tool for denoising',
      ext_modules = [module])
