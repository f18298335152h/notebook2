程序编译和运行：

/home/fenghui/tpncnn/examples  下有CMakeLists.txt 

编译运行程序的时候，将要编译的程序添加到该CMakeLists.txt  文件中：
例如我要编译运行tp_onnx.cpp ，则添加如下语句到CMakeLists.txt 
add_executable(tp_onnx tp_onnx.cpp)
target_link_libraries(tp_onnx ncnn ${OpenCV_LIBS})

然后执行:
cmake ..
make
在 example文件夹下，就能找到编译好的tp_onnx  ，可直接运行
以后要添加新的程序只需执行上述操作即可。



模型转化：

同时 在convert onnx程序是，需要在build 文件夹下，执行编译操作，
才能产生所有依赖的函数，在tools中的onnx下执行模型的转化。失败后有提示，一半就是不支持模型的某些operation，
转化完成后，会在当前目录下生成.param 和.bin 两个文件，这个即是ncnn所识别的模型文件，此时完成了onnx模型到ncnn的转化，
然后在，在自己的main程序中继承ncnn的net、layer等等类，加载该模型，并执行inference，执行的过程参照上述编译运行的步骤



在tools 目录的onnx下，修改其中的onnx2ncnn.cpp ，然后到根目录下，创建build目录，进入build目录，cmake ..  make 
即可在build/tools/onnx 目录下得到编译好的onnx2ncnn.cpp，可直接运行，，所以修改/tools/onnx/onnx2ncnn.cpp---> 根目录创建build->cmake .. -> make-->即可debugonnx2ncnn.cpp