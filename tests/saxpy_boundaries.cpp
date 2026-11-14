#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
void saxpyCuda(int, float, float*, float*, float*);
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cerr << "A real CUDA device is required\n"; return 2;
    }
    int cases = 0;
    for (int n : {0,1,2,7,31,511,512,513,1023,1024,1025,100003}) {
        for (float alpha : {-2.0f,0.0f,0.5f,2.0f}) {
            std::vector<float> x(n+2, -999), y(n+2, -999), out(n+2, -999);
            for (int i=0;i<n;++i) { x[i+1]=(i%31-15)*0.25f; y[i+1]=(i%19-9)*0.5f; }
            saxpyCuda(n, alpha, x.data()+1, y.data()+1, out.data()+1);
            for (int i=0;i<n;++i) {
                float expected=alpha*x[i+1]+y[i+1];
                if (!std::isfinite(out[i+1]) || out[i+1]!=expected) return 1;
            }
            if (out.front()!=-999 || out.back()!=-999) return 1;
            ++cases;
        }
    }
    bool rejected = false;
    try { saxpyCuda(-1, 1, nullptr, nullptr, nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) return 1;
    std::cout << "SAXPY correctness cases=" << cases << " negative-size rejection=1\n";
}
