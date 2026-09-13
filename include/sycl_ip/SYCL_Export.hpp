#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
    #if defined(SYCL_IP_EXPORTS)
        #define SYCL_IP_API __declspec(dllexport)
    #elif defined(SYCL_IP_STATIC)
        #define SYCL_IP_API
    #else
        #define SYCL_IP_API __declspec(dllimport)
    #endif
#else
    #if defined(SYCL_IP_EXPORTS)
        #define SYCL_IP_API __attribute__((visibility("default")))
    #else
        #define SYCL_IP_API
    #endif
#endif
