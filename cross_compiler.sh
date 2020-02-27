
针对不同平台构建交叉编译的环境，对应常用的andriod平台来说，只需要配置ndk即可。
下载官方的ndk，解压，调用makestone..  针对不同的平台生成对应的编译器，
环境变量配置 ANDROID_NDK_HOME 和生成编译器的bin:$PATH 即可执行下面的脚本

#!/bin/sh

# set your target arch, options are:
#   x86-64
#   android-armv7a-with-neon
# ABI="x86-64"
# ABI="armeabi-v7a"
ABI="hisi3516v300"
# ABI="android-armv7a-with-neon"
# ABI="rk3399pro-linux"
# ABI="rk3399pro-android"

# linux x86-64
function build_x86_64() {
    mkdir -p build-x86-64
    pushd build-x86-64
    cmake -DABI=$ABI ..
    # make -j32
    popd
}

# linux arm hisi3516v300
function build_hisi3516v300() {
    rm -rf build-hisi3516v300
    mkdir -p build-hisi3516v300
    pushd build-hisi3516v300
    cmake \
        -DCMAKE_TOOLCHAIN_FILE=../hisi.toolchain.cmake \
        -DABI=$ABI \
        ..
    make
    popd
}


# android armv7a with neon
function build_armv7a_with_neon() {
    rm -rf build-android-armv7a-with-neon
    mkdir -p build-android-armv7a-with-neon
    pushd build-android-armv7a-with-neon
    cmake \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI="armeabi-v7a with NEON" \
        -DANDROID_STL=c++_static \
        -DANDROID_PLATFORM=android-16 \
        -DABI=$ABI \
        ..
    make
    popd
}

# android armv7a with neon
function build_rk3399pro_linux() {
    rm -rf build-rk3399pro-linux
    mkdir -p build-rk3399pro-linux
    pushd build-rk3399pro-linux
    cmake \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI="arm64-v8a" \
        -DANDROID_STL=c++_static \
        -DANDROID_PLATFORM=android-16 \
        -DABI=$ABI \
        ..
    make
    popd
}
        # -DCMAKE_TOOLCHAIN_FILE=../cmake/android.toolchain.cmake \

case "$ABI" in
    "x86-64")
        build_x86_64
        ;;
    "armeabi-v7a")
        build_armv7a_with_neon
        ;;
    "hisi3516v300")
        build_hisi3516v300
        ;;
    "rk3399pro-linux")
        build_rk3399pro_linux 
        ;;
esac
