include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(gemm_tests gemm_test.cu)
target_link_libraries(gemm_tests PRIVATE gemm GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(gemm_tests)
