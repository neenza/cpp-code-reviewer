#include "session_manager.h"
#include <iostream>
#include <cstdlib>

namespace order_system {

SessionManager::SessionManager() {}

SessionManager::~SessionManager() {
    cleanup_all();
}

bool SessionManager::create_session(const std::string& session_id, const char* token_str, const char* notes) {
    // Critical Flaw 1: Raw manual allocation with no smart pointers
    SessionData* data = new SessionData();

    // Critical Flaw 2: Buffer overflow risk (strcpy without bound checks into fixed 32-byte char array)
    strcpy(data->token, token_str);

    if (notes) {
        data->notes_len = strlen(notes);
        data->user_notes = (char*)malloc(data->notes_len + 1);
        strcpy(data->user_notes, notes);
    } else {
        data->user_notes = nullptr;
        data->notes_len = 0;
    }

    data->active_orders_count = 0;

    // Critical Flaw 3: Data race - active_sessions_ accessed without mutex
    active_sessions_[session_id] = data;
    last_accessed_cache_ = data;
    return true;
}

const SessionData* SessionManager::get_session(const std::string& session_id) {
    // Minor Issue: Inefficient lookup and non-thread-safe cache update
    auto it = active_sessions_.find(session_id);
    if (it != active_sessions_.end()) {
        last_accessed_cache_ = it->second;
        return it->second;
    }
    return nullptr;
}

void SessionManager::invalidate_session(const std::string& session_id) {
    auto it = active_sessions_.find(session_id);
    if (it != active_sessions_.end()) {
        SessionData* data = it->second;
        
        // Free user notes
        if (data->user_notes) {
            free(data->user_notes);
            // data->user_notes not set to nullptr
        }

        // Critical Flaw 4: Double-free / Use-After-Free risk
        // If last_accessed_cache_ points to data, it becomes dangling!
        delete data;
        active_sessions_.erase(it);
        // last_accessed_cache_ is left dangling
    }
}

void SessionManager::cleanup_all() {
    for (auto& pair : active_sessions_) {
        if (pair.second) {
            if (pair.second->user_notes) {
                free(pair.second->user_notes);
            }
            delete pair.second;
        }
    }
    active_sessions_.clear();
    last_accessed_cache_ = nullptr;
}

} // namespace order_system
