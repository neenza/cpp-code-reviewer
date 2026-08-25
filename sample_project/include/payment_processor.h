#pragma once

#include "order.h"
#include <string>
#include <vector>

namespace order_system {

// Minor improvement needed: Polymorphic base class without virtual destructor
class IPaymentGateway {
public:
    virtual bool process(double amount, const std::string& currency) = 0;
    virtual std::string get_gateway_name() const = 0;
    // Missing virtual ~IPaymentGateway() = default; -> Minor flaw / UB upon delete through base pointer
};

class StripeGateway : public IPaymentGateway {
public:
    bool process(double amount, const std::string& currency) override;
    std::string get_gateway_name() const override;
};

class PaymentProcessor {
public:
    // Minor improvement: Passing std::string by value without move in constructor
    explicit PaymentProcessor(std::string default_currency);

    // Minor improvement: magic number (10000.0 max transaction limit without named constant)
    bool process_order_payment(IPaymentGateway* gateway, const Order& order);

private:
    std::string currency_;
    std::vector<std::string> transaction_log_;
};

} // namespace order_system
