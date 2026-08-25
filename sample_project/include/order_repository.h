#pragma once

#include "order.h"
#include <unordered_map>
#include <shared_mutex>
#include <memory>
#include <vector>
#include <optional>

namespace order_system {

/**
 * Exceptional modern C++ design:
 * Uses RAII, std::shared_mutex for thread safety (read/write lock),
 * std::optional, noexcept where appropriate, and clean value semantics.
 */
class OrderRepository {
public:
    OrderRepository() = default;
    ~OrderRepository() = default;

    // Non-copyable, movable
    OrderRepository(const OrderRepository&) = delete;
    OrderRepository& operator=(const OrderRepository&) = delete;
    OrderRepository(OrderRepository&&) noexcept = default;
    OrderRepository& operator=(OrderRepository&&) noexcept = default;

    bool add_order(const Order& order);
    std::optional<Order> get_order(const std::string& order_id) const;
    bool update_status(const std::string& order_id, OrderStatus new_status);
    std::vector<Order> get_orders_by_customer(const std::string& customer_id) const;
    size_t count() const noexcept;

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, Order> orders_;
};

} // namespace order_system
