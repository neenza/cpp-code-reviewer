cmake_minimum_required(VERSION 3.16)
project(SampleOrderSystem CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

add_library(orders_lib
    src/order_repository.cpp
    src/session_manager.cpp
    src/payment_processor.cpp
)

add_executable(order_server
    src/main.cpp
)

target_link_libraries(order_server PRIVATE orders_lib pthread)
