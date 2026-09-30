# SimpleCV

一个用 **stb** 封装的极简 OpenCV 风格库：提供 `Mat`、`imread/imwrite/imencode/imdecode`、`resize`、`cvtColor`。

## 依赖

项目已自带 stb 头文件：

- `src/stb_image.h`
- `src/stb_image_write.h`
- `src/stb_image_resize2.h`

## 构建

```bash
cmake -S . -B build -DSIMPLECV_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## 头文件

- 全量入口：`include/SimpleCV.hpp`
- 类型定义：`include/SimpleCV_Types.hpp`
- API 声明：`include/SimpleCV_API.hpp`

## API

- `Mat`：浅拷贝 + 引用计数（`shared_ptr`）
- `Mat_<T>`：类型化矩阵封装（类似 OpenCV），提供 `Mat8u/Mat32i/Mat32f/Mat64f` 与简写 `Matf`
- 说明：图像相关 API（`imread/imwrite/resize/cvtColor/draw/text`）目前仅支持 `Depth::U8`（即 `Mat8u`）
- `imread/imdecode`：支持 `ColorSpace` flag（RGB/BGR/RGBA/BGRA/GRAY/UNCHANGED）
- `cvtColor`：RGB/BGR/RGBA/BGRA/GRAY 任意互转
- `KalmanFilter`：线性卡尔曼滤波（配套 `Matf` 矩阵类型）
