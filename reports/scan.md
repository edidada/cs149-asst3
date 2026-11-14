# Assignment3 Scan / Find Repeats

采用题目指定的Blelloch upsweep/downsweep：初始化逻辑输入并将补齐区清零；每层仅为有效树节点启动线程，根置0后逆向分配前缀。支持原位/非原位；空输入无kernel，最大长度限制2^30避免int索引溢出。结果累加仍要求int范围内，与原接口一致。

重复检测先标记相邻相等（最后标记0），原位exclusive scan将标记变为稳定输出位置，读取最后前缀取得数量，再散射原索引。每对输出位置唯一，保持升序；scratch通过RAII释放，CUDA启动和关键同步/拷贝错误检查。

原计时wrapper、Makefile、main和checker保持不变。独立192组检查覆盖16尺寸×4输入模式×3操作（原位scan、非原位scan、find_repeats），含空、非整块、非2次幂、负数、全相等/无相等、哨兵及CPU精确oracle。device job需要真实GPU和compute-sanitizer，并运行原checker。无GPU runner时只编译，不能声称测试通过或性能达到参考20%范围。原wrapper本身的内存管理不在本次修改范围。尚无GPU运行结论，renderer待实现；本地不编译运行，无新增Python源文件。
