#pragma once

#include <string>
#include <unordered_map>
#include <cstring>

namespace order_system {

struct SessionData {
    char token[32]; // Fixed-size buffer
    char* user_notes; // Raw pointer with manual memory management
    size_t notes_len;
    uint32_t active_orders_count;
};

/**
 * Contains Critical Flaws & Minor Issues:
 * 1. Memory Safety: Double free and Use-After-Free in cleanup_session/invalidate.
 * 2. Buffer Overflow: strcpy without bounds check in create_session.
 * 3. Concurrency: Data race on active_sessions_ map (no synchronization across threads).
 * 4. Resource Leak: Exception in token generation leaks allocated user_notes.
 */
class SessionManager {
public:
    SessionManager();
    ~SessionManager();

    // Critical flaw: Rule of 3/5 violation (no custom copy constructor/assignment for raw ptrs)
    
    bool create_session(const std::string& session_id, const char* token_str, const char* notes);
    const SessionData* get_session(const std::string& session_id);
    void invalidate_session(const std::string& session_id);
    void cleanup_all();

private:
    std::unordered_map<std::string, SessionData*> active_sessions_;
    SessionData* last_accessed_cache_{nullptr}; // Dangling pointer risk
};

} // namespace order_system
