# Assignment3 CUDA SAXPY

完成两个输入的设备分配与H2D、独立输出缓冲、kernel调用和D2H，输出无需预复制。非整块尺寸保持原kernel越界保护，零尺寸直接返回，负尺寸/非空空指针拒绝。CUDA调用失败抛异常并通过RAII释放已成功分配的内存；正常路径检查释放错误。

kernel计时包含启动及cudaDeviceSynchronize，整体计时包含两个输入传输、kernel和输出传输，不含分配释放；流量3*N*sizeof(float)，打印GiB/s避免把1024^3误称GB/s。尚无真实GPU计时，无法回答题目CPU对比、T4G设备带宽及总线开销的实验问题。

原Makefile、原main和kernel未修改。独立48组尺寸/alpha测试含0、512块边界、奇数大尺寸与输出哨兵、精确二进制可表示输入，并拒绝负数；GPU job运行compute-sanitizer。当前仓库没有自托管runner，仅自动进行CUDA 12.8远端编译。device job需注册有CUDA工具链的self-hosted/linux/gpu runner后手动启用；未执行时不视为通过。没有GPU正确性、性能或整作业完成结论，scan/renderer尚未实现。本地未编译运行，无新增Python源文件。
