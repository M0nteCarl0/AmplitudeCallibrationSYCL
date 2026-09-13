#include "sycl_ip/SYCL_Suppress.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace sycl_ip {

bool SYCL_Suppress::suppressSpikes(SYCLContext& ctx,
                                   const uint16_t* d_in,
                                   uint16_t* d_out,
                                   int width,
                                   int height,
                                   double p0Noise,
                                   double p1Noise,
                                   double coeff1,
                                   double coeff2,
                                   int coeff3) {
    if (!d_in || !d_out || width < 8 || height < 8) return false;

    auto& q = ctx.getQueue();
    float p0N = static_cast<float>(p0Noise);
    float p1N = static_cast<float>(p1Noise);
    float coeff1F = static_cast<float>(coeff1);
    float coeff2F = static_cast<float>(coeff2);

    q.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
        int y = static_cast<int>(id[0]);
        int x = static_cast<int>(id[1]);

        if (x < 3 || x >= width - 3 || y < 3 || y >= height - 3) {
            d_out[y * width + x] = d_in[y * width + x];
            return;
        }

        uint16_t A0 = d_in[y * width + x];

        // 3x3 ring (8 pixels)
        int sum33 = 0;
        int a3x3[8];
        int j3 = 0;
        for (int i = -1; i <= 1; ++i) {
            a3x3[j3++] = d_in[(y - 1) * width + (x + i)];
            sum33 += a3x3[j3 - 1];
            a3x3[j3++] = d_in[(y + 1) * width + (x + i)];
            sum33 += a3x3[j3 - 1];
        }
        a3x3[j3++] = d_in[y * width + (x - 1)];
        sum33 += a3x3[j3 - 1];
        a3x3[j3++] = d_in[y * width + (x + 1)];
        sum33 += a3x3[j3 - 1];

        int A33 = sum33 / 8;

        // 5x5 ring (16 pixels)
        int sum55 = 0;
        for (int i = -2; i <= 2; ++i) {
            sum55 += d_in[(y - 2) * width + (x + i)];
            sum55 += d_in[(y + 2) * width + (x + i)];
        }
        for (int i = -1; i <= 1; ++i) {
            sum55 += d_in[(y + i) * width + (x - 2)];
            sum55 += d_in[(y + i) * width + (x + 2)];
        }
        int A55 = sum55 / 16;

        float Rms55 = p1N * static_cast<float>(A55) + p0N;
        int coeffRms55 = static_cast<int>(coeff1F * Rms55);

        uint16_t outVal = A0;

        if (A33 > A55 + coeffRms55) {
            // 7x7 ring (24 pixels)
            int sum77 = 0;
            int a7x7[24];
            int j7 = 0;
            for (int i = -3; i <= 3; ++i) {
                a7x7[j7++] = d_in[(y - 3) * width + (x + i)];
                sum77 += a7x7[j7 - 1];
                a7x7[j7++] = d_in[(y + 3) * width + (x + i)];
                sum77 += a7x7[j7 - 1];
            }
            for (int i = -2; i <= 2; ++i) {
                a7x7[j7++] = d_in[(y + i) * width + (x - 3)];
                sum77 += a7x7[j7 - 1];
                a7x7[j7++] = d_in[(y + i) * width + (x + 3)];
                sum77 += a7x7[j7 - 1];
            }

            int A77 = sum77 / 24;
            int sumDiff77 = 0;
            for (int i = 0; i < 24; ++i) {
                sumDiff77 += std::abs(A77 - a7x7[i]);
            }
            int Rms77 = sumDiff77 / 24;

            float Rms77Theory = p1N * static_cast<float>(A77) + p0N;
            int coeffRms77 = static_cast<int>(coeff2F * Rms77Theory);

            if (Rms77 < coeffRms77) {
                int amplLimit = A77 + (coeff3 * static_cast<int>(Rms77Theory));
                if (static_cast<int>(A0) > amplLimit) {
                    outVal = static_cast<uint16_t>(sycl::clamp(A77, 0, 65535));
                }
            }
        } else {
            // Check single pixel spike
            float Rms33Theory = p1N * static_cast<float>(A33) + p0N;
            int coeffRmsSingle = static_cast<int>(coeff1F * Rms33Theory * 4.0f);

            if (static_cast<int>(A0) > A33 + coeffRmsSingle) {
                int sumDiff33 = 0;
                for (int i = 0; i < 8; ++i) {
                    sumDiff33 += std::abs(a3x3[i] - A33);
                }
                int Rms33 = sumDiff33 / 8;
                int coeffRmsThresh = static_cast<int>(coeff2F * Rms33Theory);

                if (Rms33 < coeffRmsThresh) {
                    outVal = static_cast<uint16_t>(sycl::clamp(A33, 0, 65535));
                }
            }
        }

        d_out[y * width + x] = outVal;
    }).wait();

    return true;
}

void SYCL_Suppress::hostSuppressSpikes(const uint16_t* in,
                                      uint16_t* out,
                                      int width,
                                      int height,
                                      double p0Noise,
                                      double p1Noise,
                                      double coeff1,
                                      double coeff2,
                                      int coeff3) {
    std::memcpy(out, in, static_cast<size_t>(width) * height * sizeof(uint16_t));

    for (int y = 3; y < height - 3; ++y) {
        for (int x = 3; x < width - 3; ++x) {
            uint16_t A0 = in[y * width + x];

            // 3x3 ring
            int sum33 = 0;
            int a3x3[8];
            int j3 = 0;
            for (int i = -1; i <= 1; ++i) {
                a3x3[j3++] = in[(y - 1) * width + (x + i)];
                sum33 += a3x3[j3 - 1];
                a3x3[j3++] = in[(y + 1) * width + (x + i)];
                sum33 += a3x3[j3 - 1];
            }
            a3x3[j3++] = in[y * width + (x - 1)];
            sum33 += a3x3[j3 - 1];
            a3x3[j3++] = in[y * width + (x + 1)];
            sum33 += a3x3[j3 - 1];

            int A33 = sum33 / 8;

            // 5x5 ring
            int sum55 = 0;
            for (int i = -2; i <= 2; ++i) {
                sum55 += in[(y - 2) * width + (x + i)];
                sum55 += in[(y + 2) * width + (x + i)];
            }
            for (int i = -1; i <= 1; ++i) {
                sum55 += in[(y + i) * width + (x - 2)];
                sum55 += in[(y + i) * width + (x + 2)];
            }
            int A55 = sum55 / 16;

            double Rms55 = p1Noise * static_cast<double>(A55) + p0Noise;
            int coeffRms55 = static_cast<int>(coeff1 * Rms55);

            if (A33 > A55 + coeffRms55) {
                int sum77 = 0;
                int a7x7[24];
                int j7 = 0;
                for (int i = -3; i <= 3; ++i) {
                    a7x7[j7++] = in[(y - 3) * width + (x + i)];
                    sum77 += a7x7[j7 - 1];
                    a7x7[j7++] = in[(y + 3) * width + (x + i)];
                    sum77 += a7x7[j7 - 1];
                }
                for (int i = -2; i <= 2; ++i) {
                    a7x7[j7++] = in[(y + i) * width + (x - 3)];
                    sum77 += a7x7[j7 - 1];
                    a7x7[j7++] = in[(y + i) * width + (x + 3)];
                    sum77 += a7x7[j7 - 1];
                }
                int A77 = sum77 / 24;

                int sumDiff77 = 0;
                for (int i = 0; i < 24; ++i) sumDiff77 += std::abs(A77 - a7x7[i]);
                int Rms77 = sumDiff77 / 24;

                double Rms77Theory = p1Noise * static_cast<double>(A77) + p0Noise;
                int coeffRms77 = static_cast<int>(coeff2 * Rms77Theory);

                if (Rms77 < coeffRms77) {
                    int amplLimit = A77 + (coeff3 * static_cast<int>(Rms77Theory));
                    if (static_cast<int>(A0) > amplLimit) {
                        out[y * width + x] = static_cast<uint16_t>(std::clamp(A77, 0, 65535));
                    }
                }
            } else {
                double Rms33Theory = p1Noise * static_cast<double>(A33) + p0Noise;
                int coeffRmsSingle = static_cast<int>(coeff1 * Rms33Theory * 4.0);

                if (static_cast<int>(A0) > A33 + coeffRmsSingle) {
                    int sumDiff33 = 0;
                    for (int i = 0; i < 8; ++i) sumDiff33 += std::abs(a3x3[i] - A33);
                    int Rms33 = sumDiff33 / 8;
                    int coeffRmsThresh = static_cast<int>(coeff2 * Rms33Theory);

                    if (Rms33 < coeffRmsThresh) {
                        out[y * width + x] = static_cast<uint16_t>(std::clamp(A33, 0, 65535));
                    }
                }
            }
        }
    }
}

} // namespace sycl_ip
