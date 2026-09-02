#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "gemm.h"
#include "host_matrix.h"

namespace {

struct GemmVariant {
    const char *name;
    void (*run)(const DeviceMatrix &, const DeviceMatrix &, DeviceMatrix &);
};

const GemmVariant kGemmVariants[] = {
    {"v1", GEMM_v1},
    {"v2", GEMM_v2},
    {"v3", GEMM_v3},
    {"v4", GEMM_v4},
};

std::string VariantTag(const GemmVariant &variant) {
    return std::string("gemm ") + variant.name;
}

void FillRandom(HostMatrix &matrix, std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    for (size_t row = 0; row < matrix.rows(); row++) {
        for (size_t col = 0; col < matrix.cols(); col++) {
            matrix.at(row, col) = __float2half(dist(rng));
        }
    }
}

HostMatrix Identity(size_t n) {
    HostMatrix identity(n, n, n);
    for (size_t i = 0; i < n; i++) {
        identity.at(i, i) = __float2half(1.0f);
    }
    return identity;
}

void ExpectGemmMatches(const GemmVariant &variant, const HostMatrix &a,
                       const HostMatrix &b, float relative_tolerance = 0.02f,
                       float absolute_tolerance = 1.0f) {
    SCOPED_TRACE(VariantTag(variant));

    ASSERT_EQ(a.cols(), b.rows());

    size_t out_bytes = a.rows() * b.cols() * sizeof(__half);

    DeviceMatrix a_dev = a.TransformToGpu();
    DeviceMatrix b_dev = b.TransformToGpu();
    DeviceMatrix d_dev{nullptr, a.rows(), b.cols(), b.cols()};
    ASSERT_EQ(cudaMalloc(&d_dev.data, out_bytes), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_dev.data, 0, out_bytes), cudaSuccess);

    variant.run(a_dev, b_dev, d_dev);

    std::vector<__half> d_half(a.rows() * b.cols());
    ASSERT_EQ(cudaMemcpy(d_half.data(), d_dev.data,
                         d_half.size() * sizeof(__half),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    HostMatrix expected = a * b;

    for (size_t row = 0; row < a.rows(); row++) {
        for (size_t col = 0; col < b.cols(); col++) {
            float result = __half2float(d_half[row * b.cols() + col]);
            float expected_value = __half2float(expected.at(row, col));
            float tolerance =
                std::max(absolute_tolerance,
                         std::abs(expected_value) * relative_tolerance);
            EXPECT_NEAR(result, expected_value, tolerance)
                << "at [" << row << "][" << col << "]";
        }
    }

    cudaFree(a_dev.data);
    cudaFree(b_dev.data);
    cudaFree(d_dev.data);
}

std::string VariantTestName(const testing::TestParamInfo<GemmVariant> &info) {
    return info.param.name;
}

} // namespace

class GemmTest : public ::testing::TestWithParam<GemmVariant> {};

TEST_P(GemmTest, MatchesReferenceImplementation) {
    constexpr size_t N = 64;

    HostMatrix a(N, N, N);
    HostMatrix b(N, N, N);
    for (size_t row = 0; row < N; row++) {
        for (size_t col = 0; col < N; col++) {
            a.at(row, col) = __float2half(static_cast<float>((row + col) % 5));
            b.at(row, col) = __float2half(static_cast<float>((row * col) % 7));
        }
    }

    ExpectGemmMatches(GetParam(), a, b);
}

TEST_P(GemmTest, MultiplyByIdentityReturnsOriginal) {
    constexpr size_t N = 65;

    HostMatrix a(N, N, N);
    std::mt19937 rng(2024);
    FillRandom(a, rng);

    ExpectGemmMatches(GetParam(), a, Identity(N), 0.0f, 1e-3f);
}

TEST_P(GemmTest, MultiplyByZeroReturnsZero) {
    constexpr size_t N = 65;

    HostMatrix a(N, N, N);
    std::mt19937 rng(4096);
    FillRandom(a, rng);

    HostMatrix zero(N, N, N);

    ExpectGemmMatches(GetParam(), a, zero, 0.0f, 1e-3f);
}

INSTANTIATE_TEST_SUITE_P(Variants, GemmTest,
                         ::testing::ValuesIn(kGemmVariants), VariantTestName);

class GemmVectorShapeTest
    : public ::testing::TestWithParam<std::tuple<GemmVariant, size_t>> {};

TEST_P(GemmVectorShapeTest, ColumnTimesRow) {
    auto [variant, n] = GetParam();
    std::mt19937 rng(12345 + n);

    HostMatrix a(n, 1, 1);
    HostMatrix b(1, n, n);
    FillRandom(a, rng);
    FillRandom(b, rng);

    ExpectGemmMatches(variant, a, b);
}

TEST_P(GemmVectorShapeTest, RowTimesColumn) {
    auto [variant, n] = GetParam();
    std::mt19937 rng(54321 + n);

    HostMatrix a(1, n, n);
    HostMatrix b(n, 1, 1);
    FillRandom(a, rng);
    FillRandom(b, rng);

    ExpectGemmMatches(variant, a, b);
}

namespace {

std::string VectorShapeTestName(
    const testing::TestParamInfo<std::tuple<GemmVariant, size_t>> &info) {
    return std::string(std::get<0>(info.param).name) + "_n" +
           std::to_string(std::get<1>(info.param));
}

} // namespace

INSTANTIATE_TEST_SUITE_P(
    Dimensions, GemmVectorShapeTest,
    ::testing::Combine(::testing::ValuesIn(kGemmVariants),
                       ::testing::Values(size_t{33}, size_t{65}, size_t{129})),
    VectorShapeTestName);

struct RectangularShape {
    size_t m;
    size_t k;
    size_t n;
};

class GemmRectangularShapeTest
    : public ::testing::TestWithParam<
          std::tuple<GemmVariant, RectangularShape>> {};

TEST_P(GemmRectangularShapeTest, MatchesReferenceImplementation) {
    auto [variant, shape] = GetParam();
    std::mt19937 rng(shape.m * 1000003 + shape.k * 1009 + shape.n);

    HostMatrix a(shape.m, shape.k, shape.k);
    HostMatrix b(shape.k, shape.n, shape.n);
    FillRandom(a, rng);
    FillRandom(b, rng);

    ExpectGemmMatches(variant, a, b);
}

namespace {

std::string RectangularShapeTestName(
    const testing::TestParamInfo<std::tuple<GemmVariant, RectangularShape>>
        &info) {
    const RectangularShape &shape = std::get<1>(info.param);
    return std::string(std::get<0>(info.param).name) + "_" +
           std::to_string(shape.m) + "x" + std::to_string(shape.k) + "x" +
           std::to_string(shape.n);
}

} // namespace

INSTANTIATE_TEST_SUITE_P(
    Shapes, GemmRectangularShapeTest,
    ::testing::Combine(::testing::ValuesIn(kGemmVariants),
                       ::testing::Values(RectangularShape{33, 65, 129},
                                         RectangularShape{65, 129, 33},
                                         RectangularShape{129, 33, 65})),
    RectangularShapeTestName);
