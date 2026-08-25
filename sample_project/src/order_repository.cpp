#include "order_repository.h"
#include <mutex>

namespace order_system {

bool OrderRepository::add_order(const Order& order) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (orders_.find(order.order_id) != orders_.end()) {
        return false;
    }
    orders_.emplace(order.order_id, order);
    return true;
}

std::optional<Order> OrderRepository::get_order(const std::string& order_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return std::nullopt;
    }
    return it->second;
}

bool OrderRepository::update_status(const std::string& order_id, OrderStatus new_status) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return false;
    }
    it->second.status = new_status;
    return true;
}

std::vector<Order> OrderRepository::get_orders_by_customer(const std::string& customer_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<Order> result;
    for (const auto& [id, order] : orders_) {
        if (order.customer_id == customer_id) {
            result.push_back(order);
        }
    }
    return result;
}

size_t OrderRepository::count() const noexcept {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return orders_.size();
}

} // namespace order_system
