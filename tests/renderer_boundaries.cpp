#include "cudaRenderer.h"
#include "refRenderer.h"
#include "image.h"
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>
int main() {
    int checked=0;
    for (SceneName scene : {CIRCLE_RGB,CIRCLE_RGBY,BIG_LITTLE,LITTLE_BIG,HYPNOSIS,BOUNCING_BALLS}) {
        for (auto size : {std::make_pair(1,1),std::make_pair(15,17),std::make_pair(17,31),std::make_pair(65,33)}) {
            RefRenderer ref; CudaRenderer gpu;
            ref.loadScene(scene,17); gpu.loadScene(scene,17);
            ref.allocOutputImage(size.first,size.second); gpu.allocOutputImage(size.first,size.second);
            ref.setup(); gpu.setup();
            for(int frame=0;frame<3;++frame) {
                ref.clearImage(); gpu.clearImage();
                ref.advanceAnimation(); gpu.advanceAnimation();
                ref.render(); gpu.render();
                const Image* expected=ref.getImage(); const Image* actual=gpu.getImage();
                for(int i=0;i<4*size.first*size.second;++i) {
                    if(!std::isfinite(actual->data[i])) return 1;
                    if(i%4!=3 && std::fabs(actual->data[i]-expected->data[i])>0.1f) return 1;
                }
                std::vector<float> first(actual->data,actual->data+4*size.first*size.second);
                gpu.clearImage(); gpu.render(); actual=gpu.getImage();
                for(size_t i=0;i<first.size();++i) if(first[i]!=actual->data[i]) return 1;
                ++checked;
            }
        }
    }
    std::cout<<"non-square animated reference/deterministic frame checks="<<checked<<std::endl;
}
