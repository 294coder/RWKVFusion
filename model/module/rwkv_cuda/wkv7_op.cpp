#include <torch/extension.h>
#include "ATen/ATen.h"

// typedef at::Float32 float;

void cuda_forward(int B, int T, int C, int H, float *r, float *w, float *k, float *v, float *a, float *b, float *y, float *saa, float* sss);
void cuda_backward(int B, int T, int C, int H, float *r, float *w, float *k, float *v, float *a, float *b, float *saa, float* sss, float* zzz, float *gy, float *gr, float *gw, float *gk, float *gv, float *ga, float *gb);

void forward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &y, torch::Tensor &saa, torch::Tensor &sss) {
    cuda_forward(B, T, C, H, r.data_ptr<float>(), w.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), saa.data_ptr<float>(), sss.data_ptr<float>());
}
void backward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &saa, torch::Tensor &sss, torch::Tensor &zzz, torch::Tensor &gy, torch::Tensor &gr, torch::Tensor &gw, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &ga, torch::Tensor &gb) {
    cuda_backward(B, T, C, H, r.data_ptr<float>(), w.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), saa.data_ptr<float>(), sss.data_ptr<float>(), zzz.data_ptr<float>(), gy.data_ptr<float>(), gr.data_ptr<float>(), gw.data_ptr<float>(), gk.data_ptr<float>(), gv.data_ptr<float>(), ga.data_ptr<float>(), gb.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "wkv7 forward");
    m.def("backward", &backward, "wkv7 backward");
}

TORCH_LIBRARY(wkv7, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}
