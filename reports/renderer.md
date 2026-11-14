# Assignment3 Renderer

每个16×16 block拥有一个像素tile，每线程独占一个像素并在寄存器累计颜色。圆按原ID递增分批256个，线程并行判断圆是否可能覆盖tile，用原附带sharedMemExclusiveScan稳定压缩候选圆，再每像素按原顺序调用未改动shadePixel。批内及批间都保序，消除写冲突同时保持非交换的alpha混色语义。边缘无效像素不读写图像，但参与所有barrier；候选表在读完后才复用。每帧重建候选，动画位置更新可见。

原shadePixel、动画、辅助scan、box测试、接口/Makefile/main/checker均未修改。复杂度每tile扫描全部圆，且只对候选进行逐像素着色；固定shared memory容量，无全局候选列表上限，拥挤场景仍可能昂贵，未取得参考性能。CUDA启动/同步错误抛出。

独立设备job准备7场景×6尺寸共42次CPU参考比较与compute-sanitizer，覆盖1/15/16/17/31/65的tile边界和重叠顺序；再运行原8场景checker并断言每份正确性日志含成功。无GPU runner时仅远端编译，GPU执行0次，不能称正确性或性能验收完成。本地不编译运行，无新增Python源文件。
