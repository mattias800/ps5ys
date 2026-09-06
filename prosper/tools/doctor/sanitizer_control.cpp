// Deliberate defects verify the instrument, never linked into prosper.
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstring>
#include <thread>

// UBSan's object-size check can diagnose this before ASan. Isolate the ASan control
// so its success proves ASan itself is active, not just the other half of the build.
__attribute__((no_sanitize("undefined"), noinline))
static void address_control(int index) {
    int* data = new int[4]{};
    std::printf("%d\n", data[index]);
    delete[] data;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    if (std::strcmp(argv[1], "clean") == 0) {
        std::puts("SANITIZER_CONTROL_CLEAN");
        return 0;
    }
    if (std::strcmp(argv[1], "address") == 0) {
        volatile int index = 4;
        address_control(index);
    } else if (std::strcmp(argv[1], "undefined") == 0) {
        volatile int value = INT_MAX;
        std::printf("%d\n", value + 1);
    } else if (std::strcmp(argv[1], "thread") == 0) {
        // Keep individual accesses observable: an optimized non-volatile loop collapses
        // to one access, making this control needlessly sensitive to scheduling.
        volatile int value = 0;
        std::atomic<bool> start{false};
        std::atomic<unsigned> finished{0};
        auto increment = [&] {
            while (!start.load()) std::this_thread::yield();
            for (int i = 0; i < 10000; ++i) value = value + 1;
            // Keep both workers alive without establishing a happens-before edge
            // between their writes to value. This is intentionally a data race.
            finished.fetch_add(1, std::memory_order_relaxed);
            while (finished.load(std::memory_order_relaxed) != 2) std::this_thread::yield();
        };
        std::thread a(increment), b(increment);
        start.store(true);
        a.join();
        b.join();
        std::printf("%d\n", value);
    } else {
        return 2;
    }
    return 0;
}
