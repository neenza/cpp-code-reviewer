#include "order_repository.h"
#include "session_manager.h"
#include "payment_processor.h"
#include <iostream>
#include <thread>
#include <vector>

using namespace order_system;

int main() {
    std::cout << "Starting Order System Server...\n";

    OrderRepository repo;
    SessionManager session_mgr;
    PaymentProcessor processor("USD");

    Order ord1;
    ord1.order_id = "ORD-001";
    ord1.customer_id = "CUST-99";
    ord1.items.push_back({"ITEM-1", "Mechanical Keyboard", 150.0, 1});
    ord1.items.push_back({"ITEM-2", "USB-C Cable", 15.0, 2});

    repo.add_order(ord1);

    session_mgr.create_session("sess-1", "auth_token_xyz_1234567890", "VIP Client Session");

    StripeGateway stripe;
    if (processor.process_order_payment(&stripe, ord1)) {
        repo.update_status(ord1.order_id, OrderStatus::Completed);
        std::cout << "Order processed successfully. Total: $" << ord1.calculate_total() << "\n";
    }

    return 0;
}
