#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>
void exclusive_scan(int*,int,int*);
int find_repeats(int*,int,int*);
void check(cudaError_t e) { if (e!=cudaSuccess) { std::cerr<<cudaGetErrorString(e); std::abort(); } }
int main() {
    int devices=0; check(cudaGetDeviceCount(&devices)); if (!devices) return 2;
    int cases=0;
    for (int n : {0,1,2,3,7,31,255,256,257,511,512,513,1023,1024,1025,100003}) {
        int size=1; while(size<n) size*=2;
        for (int pattern=0;pattern<4;++pattern) {
            std::vector<int> in(n),prefix(n),repeats;
            int sum=0;
            for(int i=0;i<n;++i) {
                in[i]=pattern==0 ? 1 : pattern==1 ? i : pattern==2 ? (i/3)%7 : (i*17%11)-5;
                prefix[i]=sum; sum+=in[i];
                if(i && in[i-1]==in[i]) repeats.push_back(i-1);
            }
            int *x,*y; check(cudaMalloc(&x,(size+1)*sizeof(int))); check(cudaMalloc(&y,(size+1)*sizeof(int)));
            int guard=-987654;
            check(cudaMemcpy(x,in.data(),n*sizeof(int),cudaMemcpyHostToDevice));
            check(cudaMemcpy(y+size,&guard,sizeof(int),cudaMemcpyHostToDevice));
            for(int alias=0;alias<2;++alias) {
                check(cudaMemcpy(x,in.data(),n*sizeof(int),cudaMemcpyHostToDevice));
                int* target=alias ? x : y;
                check(cudaMemcpy(target+size,&guard,sizeof(int),cudaMemcpyHostToDevice));
                exclusive_scan(x,n,target); check(cudaDeviceSynchronize());
                std::vector<int> actual(n); check(cudaMemcpy(actual.data(),target,n*sizeof(int),cudaMemcpyDeviceToHost));
                if(actual!=prefix) return 1;
                int after; check(cudaMemcpy(&after,target+size,sizeof(int),cudaMemcpyDeviceToHost)); if(after!=guard) return 1;
                ++cases;
            }
            check(cudaMemcpy(x,in.data(),n*sizeof(int),cudaMemcpyHostToDevice));
            int count=find_repeats(x,n,y); if(count!=int(repeats.size())) return 1;
            std::vector<int> actual(count); check(cudaMemcpy(actual.data(),y,count*sizeof(int),cudaMemcpyDeviceToHost));
            if(actual!=repeats) return 1;
            int after; check(cudaMemcpy(&after,y+size,sizeof(int),cudaMemcpyDeviceToHost)); if(after!=guard) return 1;
            ++cases;
            check(cudaFree(x)); check(cudaFree(y));
        }
    }
    std::cout<<"scan alias/nonalias and repeat cases="<<cases<<std::endl;
}
