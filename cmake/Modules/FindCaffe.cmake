# Caffe package
unset(Caffe_FOUND)

###Set the variable Caffe_DIR as the root of your caffe directory
set(Caffe_DIR /home/ros02/caffe/)     #应该是我的caffe 安装的目录
 

find_path(Caffe_INCLUDE_DIRS NAMES  caffe/caffe.hpp caffe/common.hpp  caffe/net.hpp  caffe/proto/caffe.pb.h  caffe/util/io.hpp   /caffe/vision_layers.hpp
  HINTS
  ${Caffe_DIR}/include)  #包含的一些头文件



find_library(Caffe_LIBRARIES NAMES caffe
  HINTS
  ${Caffe_DIR}/build/lib)       #

message("lib_dirs:${Caffe_LIBRARIES}")

if(Caffe_LIBRARIES AND Caffe_INCLUDE_DIRS)
    set(Caffe_FOUND 1)
endif()
