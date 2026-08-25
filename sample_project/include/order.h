#pragma once

#include <string>
#include <chrono>
#include <cstdint>
#include <optional>

namespace order_system {

enum class OrderStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    Cancelled
};

struct OrderItem {
    std::string item_id;
    std::string item_name;
    double price{0.0};
    uint32_t quantity{1};
};

struct Order {
    std::string order_id;
    std::string customer_id;
    std::vector<OrderItem> items;
    double total_amount{0.0};
    OrderStatus status{OrderStatus::Pending};
    std::chrono::system_clock::time_point created_at;

    [[nodiscard]] double calculate_total() const noexcept {
        double sum = 0.0;
        for (const auto& item : items) {
            sum += item.price * item.quantity;
        }
        return sum;
    }
};

} // namespace order_system
