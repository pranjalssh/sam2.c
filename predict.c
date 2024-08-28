#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <cassert>
#include <cmath>
#include <omp.h>
// A single header file to include image processing deps
// Source: https://raw.githubusercontent.com/GreycLab/CImg/master/CImg.h
// #include "CImg.h"

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -i <image.ppm> -x x -y y\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -i <string>  Input image in ppm format\n");
    fprintf(stderr, "  -x <int>     x coordinate\n");
    fprintf(stderr, "  -y <int>     y coordinate\n");
    exit(EXIT_FAILURE);
}

// Config variables
const int use_high_res_features_in_sam=1;
const float transformMeans[3] = {0.485, 0.456, 0.406};
const float transformStds[3] = {0.229, 0.224, 0.225};
const int MAX_IMAGE_ENCODER_LAYERS = 48;
const int MAX_TRUNK_LAYER_NORM_1_SIZE = 1152;
const int MAX_TRUNK_LAYER_QKV_SIZE = 3456;
const int MAX_TRUNK_LAYER_QPOOL_SIZE = 2;
const int MAX_TRUNK_LAYER_PROJ_SIZE = 1152;
const int MAX_TRUNK_LAYER_MLP_SIZE = 1152;

struct Config {
    int numLayers; // 48
    int trunkLayerNorm1Sizes[MAX_IMAGE_ENCODER_LAYERS]; // max 1152
    int trunkLayerQPoolSizes[MAX_IMAGE_ENCODER_LAYERS]; // max 2
    int trunkLayerQKVSizes[MAX_IMAGE_ENCODER_LAYERS]; // max 3456
    int trunkLayerNumHeads[MAX_IMAGE_ENCODER_LAYERS]; // max 1152
    int windowSizes[MAX_IMAGE_ENCODER_LAYERS]; // max 8
};

Config modelConfig;

// All magic numbers here, derived from sam2_hiera_l.yaml
void init_config() {
    memset(modelConfig.trunkLayerQPoolSizes, 0, sizeof(modelConfig.trunkLayerQPoolSizes));
    int nh = 2, ws = 8;
    for (int i = 0; i < 48; ++i) {
        if (i == 2 || i == 8 || i == 44) { // Special layers
            modelConfig.trunkLayerQPoolSizes[i] = 2;
            nh *= 2;
        }
        modelConfig.trunkLayerNumHeads[i] = nh;
        if (i == 2) ws = 4;
        if (i == 8) ws = 16;
        if (i == 44) ws = 8;
        if (i == 23 || i == 33 || i == 43) ws = 0;
        modelConfig.windowSizes[i] = ws;
    }
    
}

// Inference variables
const int width = 1024;
const int height = 1024;
uint8_t *originalImg;  // [H, W, 3]
float *tensorImg;     // [3, H, W]
float *tensorConv1; // [144, 256, 256]
float *tensorConv2; // [256, 256, 144]
float *tensorConv3; // [1024, 8, 8, 144]
float *tensorConvTemp; // [256,256,576]
float *tensorQKV; // [1024, 8*8, 144*3]
float *tensorQKV2; // [1024, 8*8, 144*3]
float *tensorQKV3; // [1024, 8*8, 144*3]
float *tensorATT; // [2048, 8*8, 8*8]
float *posEmbed2; // [144, 256, 256]


//---------Image Encoder---------
float *trunkPatchEmbedWeights; // [144, 3, 7, 7]
float *trunkPatchEmbedBias;    // [144]
float *posEmbed; // [144, 7, 7]
float *posEmbedWindow; // [144, 8, 8]


float *trunkLayerNorm1Weight[MAX_IMAGE_ENCODER_LAYERS]; // [144]
float *trunkLayerNorm1Bias[MAX_IMAGE_ENCODER_LAYERS]; // [144]
float *blockAttnQKVWeight[MAX_IMAGE_ENCODER_LAYERS]; // [144*3, 144] (transposed)
float *blockAttnQKVBias[MAX_IMAGE_ENCODER_LAYERS]; // [144*3]
float *blockAttnProjWeight[MAX_IMAGE_ENCODER_LAYERS]; // [144, 144]
float *blockAttnProjBias[MAX_IMAGE_ENCODER_LAYERS]; // [144]
float *trunkLayerNorm2Weight[MAX_IMAGE_ENCODER_LAYERS]; // [144]
float *trunkLayerNorm2Bias[MAX_IMAGE_ENCODER_LAYERS]; // [144]
float *trunkMlp0Weight[MAX_IMAGE_ENCODER_LAYERS]; // [576, 144] (transposed)
float *trunkMlp0Bias[MAX_IMAGE_ENCODER_LAYERS]; // [576]
float *trunkMlp1Weight[MAX_IMAGE_ENCODER_LAYERS]; // [144, 576] (transposed)
float *trunkMlp1Bias[MAX_IMAGE_ENCODER_LAYERS]; // [144]


void load_ppm(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file %s", filename);
        exit(EXIT_FAILURE);
    }

    char format[3];
    if (!fgets(format, sizeof(format), fp)) {
        fprintf(stderr, "Invalid format");
        exit(EXIT_FAILURE);
    }

    if (strcmp(format, "P6") != 0 && strcmp(format, "P6\n") != 0) {
        fprintf(stderr, "Unsupported file format (must be P6).\n");
        exit(EXIT_FAILURE);
    }

    int max_val;
    char c;
    
    // Skip comments
    c = fgetc(fp);
    while (c == '#') {
        while (fgetc(fp) != '\n');
        c = fgetc(fp);
    }
    ungetc(c, fp);

    // Read image size
    int w, h;
    fscanf(fp, "%d %d", &w, &h);
    if (w != width || h != height) {
        fprintf(stderr, "Image size mismatch (must be %d x %d).\n", width, height);
        exit(EXIT_FAILURE);
    }
    fscanf(fp, "%d", &max_val);

    if (max_val != 255) {
        fprintf(stderr, "Unsupported max color value (must be 255).\n");
        exit(EXIT_FAILURE);
    }

    // Consume the newline character after max_val
    fgetc(fp);

    // Allocate memory for pixel data
    originalImg = (uint8_t*)malloc(3 * width * height * sizeof(char));

    // Read pixel data
    size_t read = fread(originalImg, sizeof(char), 3 * width * height, fp);
    if (read != 3 * width * height) {
        fprintf(stderr, "Error reading pixel data.\n");
        exit(EXIT_FAILURE);
    }
    fclose(fp);
}

void load_model(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file %s", filename);
        exit(EXIT_FAILURE);
    }
    #define load_tensor(name, size) \
        name = (float*)malloc(size * sizeof(float)); \
        assert(fread(name, sizeof(float), size, fp) == size);
    #define load_list(name, size) \
        assert(fread(name, sizeof(int32_t), size, fp) == size);

    load_list(&modelConfig.numLayers, 1);
    assert(modelConfig.numLayers == 48);
    load_list(&modelConfig.trunkLayerNorm1Sizes[0], modelConfig.numLayers);
    load_list(&modelConfig.trunkLayerQKVSizes[0], modelConfig.numLayers);

    load_tensor(trunkPatchEmbedWeights, 144 * 3 * 7 * 7);
    load_tensor(trunkPatchEmbedBias, 144);
    load_tensor(posEmbed, 144 * 7 * 7);
    load_tensor(posEmbedWindow, 144 * 8 * 8);

    for (int i = 0; i < modelConfig.numLayers; ++i) {
        int attSize = modelConfig.trunkLayerQKVSizes[i]/3;
        int norm1Size = modelConfig.trunkLayerNorm1Sizes[i];
        load_tensor(trunkLayerNorm1Weight[i], norm1Size); //144
        load_tensor(trunkLayerNorm1Bias[i], norm1Size); //144
        load_tensor(blockAttnQKVWeight[i], norm1Size*attSize*3); //144*144*3
        load_tensor(blockAttnQKVBias[i], attSize*3);
        load_tensor(blockAttnProjWeight[i], attSize*attSize); //144*144
        load_tensor(blockAttnProjBias[i], attSize); //144
        load_tensor(trunkLayerNorm2Weight[i], attSize);
        load_tensor(trunkLayerNorm2Bias[i], attSize);
        load_tensor(trunkMlp0Weight[i], attSize*attSize*4);
        load_tensor(trunkMlp0Bias[i], attSize*4);
        load_tensor(trunkMlp1Weight[i], attSize*attSize*4);
        load_tensor(trunkMlp1Bias[i], attSize);
    }
    fclose(fp);
    #undef load_tensor
}

void allocate_memory() {
    int maxBlockLayerNormSize = modelConfig.trunkLayerNorm1Sizes[modelConfig.numLayers-1];
    int maxBlockAttentionSize = modelConfig.trunkLayerQKVSizes[modelConfig.numLayers-1]/3;
    tensorImg = (float*)malloc(3*width*height*sizeof(float));
    tensorConv1 = (float*)malloc(maxBlockLayerNormSize*256*256*sizeof(float));
    tensorConv2 = (float*)malloc(maxBlockLayerNormSize*256*256*sizeof(float));
    tensorConv3 = (float*)malloc(maxBlockLayerNormSize*256*256*sizeof(float));
    tensorConvTemp = (float*)malloc(256*256*maxBlockAttentionSize*4*sizeof(float)); // MLP intermediate layer
    posEmbed2 = (float*)malloc(144*256*256*sizeof(float));
    tensorQKV = (float*)malloc(256*256*maxBlockAttentionSize*3*sizeof(float));
    tensorQKV2 = (float*)malloc(256*256*maxBlockAttentionSize*3*sizeof(float));
    tensorQKV3 = (float*)malloc(256*256*maxBlockAttentionSize*3*sizeof(float));
    tensorATT = (float*)malloc(256*256*16*16*8*sizeof(float)); // TODO
}

void debug_tensor(float* t, int d0, int d1, int d2, int d3) {
    int i0 = 0, i1 = 0, i2 = 0, i3 = 0;
    printf("[%d %d %d %d] = %f\n", i0, i1, i2, i3, t[d1*d2*d3*i0 + d2*d3*i1 + d3*i2 + i3]);
    // i0 = 0, i1 = 0, i2 = 0, i3 = 1;
    // printf("[%d %d %d %d] = %f\n", i0, i1, i2, i3, t[d1*d2*d3*i0 + d2*d3*i1 + d3*i2 + i3]);
    // i0 = 0, i1 = 0, i2 = 1, i3 = 0;
    // printf("[%d %d %d %d] = %f\n", i0, i1, i2, i3, t[d1*d2*d3*i0 + d2*d3*i1 + d3*i2 + i3]);
    // i0 = 0, i1 = 1, i2 = 0, i3 = 0;
    // printf("[%d %d %d %d] = %f\n", i0, i1, i2, i3, t[d1*d2*d3*i0 + d2*d3*i1 + d3*i2 + i3]);
    // i0 = 1, i1 = 0, i2 = 0, i3 = 0;
    // printf("[%d %d %d %d] = %f\n", i0, i1, i2, i3, t[d1*d2*d3*i0 + d2*d3*i1 + d3*i2 + i3]);
    // i0 = 0, i1 = 0, i2 = 15, i3 = 50;
    // printf("[%d %d %d %d] = %f\n", i0, i1, i2, i3, t[d1*d2*d3*i0 + d2*d3*i1 + d3*i2 + i3]);
    i0 = 0, i1 = 1, i2 = 2, i3 = 61;
    printf("[%d %d %d %d] = %f\n", i0, i1, i2, i3, t[d1*d2*d3*i0 + d2*d3*i1 + d3*i2 + i3]);
    printf("\n\n");
}

void preprocess() {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < 3; c++) {
                int imgIdx = 3*(x * width + y) + c;
                int tensorIdx = c*width*height + x*width + y;
                tensorImg[tensorIdx] = (originalImg[imgIdx]/255.0 - transformMeans[c]) / transformStds[c];
            }
        }
    }
}

void conv2d(float *output, float *input, float *weight, float *bias, int input_channels, int input_height,
            int input_width, int output_channels, int kernel_size, int stride, int padding) {
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    printf("Output height: %d, Output width: %d\n", output_height, output_width);

    
    // measure time spent here using C time:
    for (int oc = 0; oc < output_channels; oc++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                float sum = 0.0;
                for (int ic = 0; ic < input_channels; ic++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;
                            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                int input_idx = ic * input_height * input_width + ih * input_width + iw;
                                int weight_idx = oc * input_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw;
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                int output_idx = oc * output_height * output_width + oh * output_width + ow;
                output[output_idx] = sum + bias[oc];
            }
        }
    }
}

void chw_to_hwc(float* output, float* input, int channels, int height, int width) {
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int c = 0; c < channels; c++) {
                int idx = h*channels*width + w*channels + c;
                output[idx] = input[c*height*width + h*width + w];
            }
        }
    }
}

inline float cubic_convolution1(float x, float A) {
    return ((A + 2.0f) * x - (A + 3.0f)) * x * x + 1.0f;
}

inline float cubic_convolution2(float x, float A) {
    return ((A * x - 5.0f * A) * x + 8.0f * A) * x - 4.0f * A;
}

void get_cubic_upsampling_coefficients(float coeffs[4], float t) {
    const float A = -0.75f;
    coeffs[0] = cubic_convolution2(t + 1.0f, A);
    coeffs[1] = cubic_convolution1(t, A);
    coeffs[2] = cubic_convolution1(1.0f - t, A);
    coeffs[3] = cubic_convolution2(2.0f - t, A);
}

float get_value_bounded(const float* data, int width, int height, int x, int y, int c, int channels) {
    x = fmaxf(fminf(x, width - 1), 0);
    y = fmaxf(fminf(y, height - 1), 0);
    return data[(c * height * width) + (y * width) + x];
}

void bicubic_interpolate(float* output, const float* input, int channels, int input_height, int input_width, int output_height, int output_width, int align_corners = 0) {
    float height_scale = (float)input_height / output_height;
    float width_scale = (float)input_width / output_width;

    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < output_height; y++) {
            float real_y = height_scale * (y + 0.5f) - 0.5f;
            int input_y = floorf(real_y);
            float ty = real_y - input_y;
            float y_coeffs[4];
            get_cubic_upsampling_coefficients(y_coeffs, ty);

            for (int x = 0; x < output_width; x++) {
                float real_x = width_scale * (x + 0.5f) - 0.5f;
                int input_x = floorf(real_x);
                float tx = real_x - input_x;
                float x_coeffs[4];
                get_cubic_upsampling_coefficients(x_coeffs, tx);

                float result = 0.0f;
                for (int m = 0; m < 4; m++) {
                    for (int n = 0; n < 4; n++) {
                        result += get_value_bounded(input, input_width, input_height, input_x - 1 + n, input_y - 1 + m, c, channels) * y_coeffs[m] * x_coeffs[n];
                    }
                }
                output[(c * output_height * output_width) + (y * output_width) + x] = result;
            }
        }
    }
}

void patch_embed() {
    conv2d(tensorConv1, tensorImg, trunkPatchEmbedWeights, trunkPatchEmbedBias, 3, height, width, 144, 7, 4, 3);
    chw_to_hwc(tensorConv2, tensorConv1, 144, 256, 256);
}

void pos_embed() {
    bicubic_interpolate(posEmbed2, posEmbed, 144, 7, 7, 256, 256);
    for (int c = 0; c < 144; c++) {
        for (int h = 0; h < 256; h++) {
            for (int w = 0; w < 256; w++) {
                int outIdx = h*256*144 + w*144 + c;
                int posIdx = c*256*256 + h*256 + w;
                int windowIdx = c*8*8 + (h%8)*8 + w%8;
                tensorConv2[outIdx] += posEmbed2[posIdx] + posEmbedWindow[windowIdx];
            }
        }
    }
}

void layer_norm(float* output, float* input, float* weight, float* bias, int height, int width, int dim) {
    for (int i = 0; i < height*width; i++) {
        float mean = 0.0;
        float variance = 0.0;
        int start_index = i * dim;

        // Calculate the mean
        for (int j = 0; j < dim; j++) {
            mean += input[start_index + j];
        }
        mean /= dim;

        // Calculate the variance
        for (int j = 0; j < dim; j++) {
            float diff = input[start_index + j] - mean;
            variance += diff * diff;
        }
        variance /= dim;

        // Calculate the standard deviation
        float stddev = sqrt(variance + 1e-6); // Adding epsilon for numerical stability

        // Normalize and apply scale and shift
        for (int j = 0; j < dim; j++) {
            output[start_index + j] = weight[j] * ((input[start_index + j] - mean) / stddev) + bias[j];
        }
    }
}

// H,W,C -> B,WS,WS,C
void window_partition(float *output, float* input, int H, int W, int C, int WS) {
    assert(H % WS == 0);
    assert(W % WS == 0);
    int num_windows_h = H / WS;
    int num_windows_w = W / WS;
    int B = num_windows_h * num_windows_w;


    for (int i = 0; i < num_windows_h; i++) {
        for (int j = 0; j < num_windows_w; j++) {
            for (int h = 0; h < WS; h++) {
                for (int w = 0; w < WS; w++) {
                    for (int c = 0; c < C; c++) {
                        int output_index = (i*num_windows_w+j)*WS*WS*C + h*WS*C + w*C + c;
                        int ih = i*WS + h;
                        int iw = j*WS + w;
                        int input_index = ih*W*C + iw*C + c;
                        output[output_index] = input[input_index];
                    }
                }
            }
        }
    }
}

// B,WS,WS,C -> H,W,C
void window_unpartition(float *output, float* input, int H, int W, int C, int WS) {
    assert(H % WS == 0);
    assert(W % WS == 0);
    int num_windows_h = H / WS;
    int num_windows_w = W / WS;
    int B = num_windows_h * num_windows_w;


    for (int i = 0; i < num_windows_h; i++) {
        for (int j = 0; j < num_windows_w; j++) {
            for (int h = 0; h < WS; h++) {
                for (int w = 0; w < WS; w++) {
                    for (int c = 0; c < C; c++) {
                        // Swap input and output index from window_partition function
                        int input_index = (i*num_windows_w+j)*WS*WS*C + h*WS*C + w*C + c;
                        int ih = i*WS + h;
                        int iw = j*WS + w;
                        int output_index = ih*W*C + iw*C + c;
                        output[output_index] = input[input_index];
                    }
                }
            }
        }
    }
}

// (B, T, C) X (OC, C) -> (B, T, OC)
void matmul_forward_naive(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int B, int T, int C, int OC) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int bt = b * T + t;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++) {
                    val += inp[bt * C + i] * weight[o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

void matmul_forward(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    const int LOOP_UNROLL = 8;
    if (B*T % LOOP_UNROLL != 0) {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        return;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #pragma omp parallel for
    for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        for (int o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            float result[LOOP_UNROLL];
            // initialize the bias, if it exists
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            for (int i = 0; i < C; i++) {
                float w = weight[i + o * C];
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    int bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                int bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

// (B, T, C) X (B,OC,C) -> (B, T, OC)
void matmul_forward_naive_batched(float* out,
                         const float* inp, const float* weight,
                         int B, int T, int C, int OC) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int bt = b * T + t;
            for (int o = 0; o < OC; o++) {
                float val = 0.0f;
                for (int i = 0; i < C; i++) {
                    val += inp[bt * C + i] * weight[b*C*OC +o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

void matmul_forward_batched(float* out,
                    const float* inp, const float* weight,
                    int B, int T, int C, int OC) {
    const int LOOP_UNROLL = 8;
    if (T % LOOP_UNROLL != 0) {
        matmul_forward_naive_batched(out, inp, weight, B, T, C, OC);
        return;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #pragma omp parallel for
    for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        for (int o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            float result[LOOP_UNROLL];
            memset(result, 0, sizeof(result));
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            for (int i = 0; i < C; i++) {
                float w = weight[(obt/T)*C*OC + o*C + i];
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    int bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                int bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

// [B, T, 3, NH, C/NH] -> [B,T,NH,C/NH], [B,T,NH,C/NH], [B,T,NH,C/NH]
// same as [B,T,3,C] -> [B,T,C], [B,T,C], [B,T,C]
void reshapeQKV(float* Q, float* K, float* V, float *input, int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                int idxq = b*T*3*C + t*3*C + c;
                int idxk = b*T*3*C + t*3*C + C + c;
                int idxv = b*T*3*C + t*3*C + 2*C + c;
                int outidx = b*T*C + t*C + c;
                Q[outidx] = input[idxq];
                K[outidx] = input[idxk];
                V[outidx] = input[idxv];
            }
        }
    }
}

// [B, T, NH, C] -> [B, NH, T, C]
void transpose_0213(float* output, float* input, int B, int T, int NH, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                for (int nh = 0; nh < NH; nh++) {
                    int iidx = b*T*NH*C + t*NH*C + nh*C + c;
                    int oidx = b*NH*T*C + nh*T*C + t*C + c;
                    output[oidx] = input[iidx];
                }
            }
        }
    }
}

// B, T, NH, C -> B, NH, C, T
void transpose_0231(float* output, float* input, int B, int T, int NH, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                for (int nh = 0; nh < NH; nh++) {
                    int iidx = b*T*NH*C + t*NH*C + nh*C + c;
                    int oidx = b*NH*C*T + nh*C*T + c*T + t;
                    output[oidx] = input[iidx];
                }
            }
        }
    }
}

void softmax(float* input, int B, int T, float scale) {
    for (int b = 0; b < B; b++) {
        float maxVal = input[b*T] * scale;
        for (int t = 0; t < T; t++) {
            int idx = b*T + t;
            float value = input[idx] * scale;
            if (value > maxVal) {
                maxVal = value;
            }
            input[idx] = value;
        }
        float sum = 0;
        for (int t = 0; t < T; t++) {
            int idx = b*T + t;
            float value = exp(input[idx] - maxVal);
            input[idx] = value;
            sum += value;
        }
        float sum_inv = sum == 0 ? 0 : 1.0f/sum;
        for (int t = 0; t < T; t++) {
            int idx = b*T + t;
            input[idx] *= sum_inv;
        }
    }
}

void attention(float* input, int layer, int B, int NH, int T, int C, int ATT) {
    printf("layer: %d B: %d NH: %d T: %d C: %d ATT: %d\n", layer, B, NH, T, C, ATT);
    // input is [B, T, C]
    matmul_forward(tensorQKV, input, blockAttnQKVWeight[layer], blockAttnQKVBias[layer], B, T, C, ATT*3);
    // tensorQKV is now [B, T, ATT*3]
    // Lets view it as [B, T, 3, NH, ATT/NH] and split Q, K, V [B, T, NH, ATT/NH]
    float *q = tensorQKV2;
    float *k = tensorQKV2 + B*T*ATT;
    float *v = tensorQKV2 + 2*B*T*ATT;
    reshapeQKV(q, k, v, tensorQKV, B, T, ATT);
    // TODO: qstride
    
    // To [B, T, NH, C/NH] -> [B,NH,T,C/NH]
    transpose_0213(tensorQKV3, q, B, T, NH, ATT/NH);
    transpose_0213(tensorQKV3 + B*T*ATT, k, B, T, NH, ATT/NH);
    transpose_0231(tensorQKV3 + 2*B*T*ATT, v, B, T, NH, ATT/NH);
    q = tensorQKV3, k = tensorQKV3 + B*T*ATT, v = tensorQKV3 + 2*B*T*ATT;

    assert(B*NH*T*T <= 256*256*16*16*8);
    matmul_forward_batched(tensorATT, q, k, B*NH, T, ATT/NH, T);
    softmax(tensorATT, B*NH*T, T, 1.0f/sqrtf(ATT/NH));
    matmul_forward_batched(tensorQKV, tensorATT, v, B*NH, T, T, ATT/NH);

    // debug_tensor(tensorQKV, B, NH, T, ATT/NH);

    transpose_0213(tensorQKV3, tensorQKV, B, NH, T, ATT/NH);
    // Now we have [B, T, NH, ATT/NH] == [B, T, ATT]
    matmul_forward(input, tensorQKV3, blockAttnProjWeight[layer], blockAttnProjBias[layer], B, T, ATT, ATT);
}

void add(float* output, float* input1, float* input2, int B, int H, int W, int C) {
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int c = 0; c < C; c++) {
                    int idx = b*H*W*C + h*W*C + w*C + c;
                    output[idx] = input1[idx] + input2[idx];
                }
            }
        }
    }
}

void gelu(float* output, float* input, int B, int H, int W, int C) {
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int c = 0; c < C; c++) {
                    int idx = b*H*W*C + h*W*C + w*C + c;
                    float x = input[idx];
                    // implement gelu with approximate='none'
                    output[idx] = x * 0.5 * (1.0 + std::erf(x / sqrt(2.0)));
                }
            }
        }
    }
}

void apply_block(int layer, int DIM, int ATT, int NH, int WS) {
    // Block 0
    // shortcut = tensorConv2
    layer_norm(tensorConv1, tensorConv2, trunkLayerNorm1Weight[layer], trunkLayerNorm1Bias[layer], 256, 256, DIM);
    // Window partition tensorConv2 from [B, H, W, C] to [B * H/8 * W/8, 8, 8, C]
    // TODO: choose window size
    window_partition(tensorConv3, tensorConv1, 256, 256, DIM, WS);
    // debug_tensor(tensorConv3, 1024, 8, 8, 144);
    int B = (256*256)/(WS*WS);
    attention(tensorConv3, layer, B, NH, WS*WS, DIM, ATT);
    // debug_tensor(tensorConv3, 1024, 8, 8, 144);
    // TODO: qstride
    window_unpartition(tensorConv1, tensorConv3, 256, 256, ATT, WS);
    // Now [1, 256, 256, 144]
    // save tensorConv1
    add(tensorConv1, tensorConv1, tensorConv2, 1, 256, 256, ATT);
    layer_norm(tensorConv2, tensorConv1, trunkLayerNorm2Weight[layer], trunkLayerNorm2Bias[layer], 256, 256, ATT);

    matmul_forward(tensorConvTemp, tensorConv2, trunkMlp0Weight[layer], trunkMlp0Bias[layer], 256, 256, ATT, ATT*4);
    gelu(tensorConvTemp, tensorConvTemp, 1, 256, 256, ATT*4);
    matmul_forward(tensorConv2, tensorConvTemp, trunkMlp1Weight[layer], trunkMlp1Bias[layer], 256, 256, ATT*4, ATT);
    add(tensorConv2, tensorConv1, tensorConv2, 1, 256, 256, ATT);
    debug_tensor(tensorConv2, 1, 256, 256, ATT);
}

void encode_image() {
    patch_embed();
    pos_embed();
    for (int i = 0; i < modelConfig.numLayers; i++) {
        apply_block(i, modelConfig.trunkLayerNorm1Sizes[i], modelConfig.trunkLayerQKVSizes[i]/3, modelConfig.trunkLayerNumHeads[i], modelConfig.windowSizes[i]);
        if (i == 1) break;
    }
}

int main(int argc, char *argv[]) {
    char *checkpoint_path = NULL;  // model checkpoint
    char *img_path = NULL;
    int x, y;
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 'i') { img_path = argv[i + 1]; }
        else if (argv[i][1] == 'x') { x = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'y') { y = atoi(argv[i + 1]); }
        else { error_usage(); }
    }
    init_config();
    load_model(checkpoint_path);
    load_ppm(img_path);
    printf("Loaded model and image\n");
    allocate_memory();
    preprocess();
    encode_image();
}