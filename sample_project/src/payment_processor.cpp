#include "payment_processor.h"
#include <iostream>

namespace order_system {

bool StripeGateway::process(double amount, const std::string& currency) {
    std::cout << "[StripeGateway] Charging " << amount << " " << currency << "\n";
    return amount > 0.0;
}

std::string StripeGateway::get_gateway_name() const {
    return "Stripe v3";
}

PaymentProcessor::PaymentProcessor(std::string default_currency) 
    : currency_(default_currency) {}

bool PaymentProcessor::process_order_payment(IPaymentGateway* gateway, const Order& order) {
    if (!gateway) {
        return false;
    }

    double total = order.calculate_total();

    // Minor Issue: Magic number 10000.0 without constant or config
    if (total > 10000.0) {
        std::cerr << "Transaction exceeds maximum limit\n";
        return false;
    }

    bool success = gateway->process(total, currency_);
    if (success) {
        // Minor Issue: Unbounded vector growth without memory reservation or rotation
        transaction_log_.push_back(order.order_id + ": SUCCESS");
    }
    return success;
}

} // namespace order_system
