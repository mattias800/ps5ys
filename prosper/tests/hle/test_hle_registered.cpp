// test_hle_registered — guards that the HLE functions we rely on are actually registered.
// register_builtin_hle() binds handlers by NID; a typo'd name or a forgotten registration would
// silently leave an import as an unimplemented stub (returns 0) and regress the boot. This checks
// a representative set across every HLE module (libc, math, file, kernel, time, service, graphics).
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include <atomic>
#include <array>
#include <cstdio>
#include <limits>
#include <thread>

using namespace prosper;

static int fails = 0;
// "Is this NID registered?" is a THIRD question, distinct from both "give me a handler I may call
// through HleFn" and "give me a guest-ABI handler" -- and `lookup` used to answer all three. It no
// longer answers this one for the printf family, which are guest-ABI (#3272): `lookup` refuses
// those, so asking it about `snprintf` here reported a registered handler as missing. That is the
// whole reason the accessor was split; `Hle::registered` is the one that means existence.
static void must(const char* name) {
    if (!Hle::registered(nid_hash(name))) {
        printf("  [FAIL] not registered: %s\n", name); fails++;
    }
}

int main() {
    printf("== test_hle_registered ==\n");
    register_builtin_hle();

    const char* names[] = {
        // libc core
        "memcpy", "memset", "strlen", "malloc", "free", "snprintf", "bcmp", "bsearch",
        "__error", "setjmp", "longjmp",
        // math (real host thunks)
        "cosf", "sinf", "sqrtf", "powf", "atan2f", "sincosf", "ldexp", "fmin", "tanh", "log1pf",
        // file / directory
        "open", "read", "stat", "sceKernelOpen", "sceKernelMkdir", "sceKernelPread", "sceKernelStat",
        // kernel: threads / sync / exceptions
        "scePthreadCreate", "scePthreadMutexLock", "scePthreadCondWait", "pthread_equal",
        "sceKernelInstallExceptionHandler", "sceKernelRaiseException",
        "sceKernelIsStack", "scePthreadGetschedparam",
        // POSIX spellings must answer identically to the sce ones (#2914)
        "pthread_getschedparam", "pthread_setschedparam",
        // time / event queues
        "sceKernelClockGettime", "sceKernelUsleep", "sceKernelCreateEqueue", "sceKernelWaitEqueue",
        "getpid",
        // services / dialogs
        "sceUserServiceGetInitialUser", "scePadOpen", "sceMsgDialogUpdateStatus",
        "sceSystemServiceHideSplashScreen",
        // HTTP helpers
        "sceHttpUriParse",
        // graphics (headless bring-up)
        "sceVideoOutOpen", "sceVideoOutSubmitFlip", "sceVideoOutGetFlipStatus",
        // audio (headless / pluggable backend)
        "sceAudioOutInit", "sceAudioOutOpen", "sceAudioOutOutput", "sceAudioOutOutputs",
        "sceAudioOutSetVolume", "sceAudioOutClose", "sceAudioOutGetPortState",
        "sceAudioInInit", "sceAudioInOpen", "sceAudioInInput", "sceAudioInClose",
    };
    for (const char* n : names) must(n);
    // sync_on_address futex is registered by raw NID (no symbol name) — check it directly.
    if (!Hle::registered("Hc4CaR6JBL0")) { printf("  [FAIL] sceKernelWaitOnAddress raw NID\n"); fails++; }

    // __ctype_get_mb_cur_max returns the VALUE of MB_CUR_MAX (1 in the "C" locale we
    // present), not a pointer to it — guest code sizes buffers as MB_CUR_MAX*n (#141).
    if (HleFn fn = Hle::lookup(nid_hash("__ctype_get_mb_cur_max"))) {
        uint64_t v = fn(0, 0, 0, 0, 0, 0);
        if (v != 1) { printf("  [FAIL] __ctype_get_mb_cur_max returned %llu, want 1 (the value, not a pointer)\n", (unsigned long long)v); fails++; }
    } else { printf("  [FAIL] not registered: __ctype_get_mb_cur_max\n"); fails++; }

    // sceNetPoolCreate returns SCE_NET_ERROR_EINVAL (0x80410116 = 0x80410100 | BSD EINVAL 22) on
    // invalid arguments, and a positive pool ID on valid arguments (#3300).
    if (HleFn fn = Hle::lookup(nid_hash("sceNetPoolCreate"))) {
        uint64_t err = fn(0, 0, 0, 0, 0, 0);
        if (err != 0x80410116ull) {
            printf("  [FAIL] sceNetPoolCreate(0, 0) returned 0x%llx, want 0x80410116 (SCE_NET_ERROR_EINVAL)\n",
                   (unsigned long long)err);
            fails++;
        }
        char dummy[16] = "testpool";
        uint64_t id = fn((uint64_t)(uintptr_t)dummy, 4096, 0, 0, 0, 0);
        if (id == 0 || (int32_t)id <= 0) {
            printf("  [FAIL] sceNetPoolCreate valid call returned non-positive id %lld\n",
                   (long long)(int32_t)id);
            fails++;
        }
    } else { printf("  [FAIL] not registered: sceNetPoolCreate\n"); fails++; }

    // Sony's Dinkumware inline isspace uses (_Getpctype()[c] & 0x144), not the
    // incompatible MSVCRT bit layout. Pin representative C-locale masks and EOF/case slots.
    if (HleFn fn = Hle::lookup(nid_hash("_Getpctype"))) {
        const auto* table = reinterpret_cast<const short*>(fn(0, 0, 0, 0, 0, 0));
        struct Expected { unsigned char ch; short mask; };
        constexpr Expected expected[] = {
            {0x00, 0x080}, {'\t', 0x4c0}, {'\n', 0x0c0}, {' ', 0x004},
            {'!', 0x008}, {'0', 0x021}, {'A', 0x003}, {'G', 0x002},
            {'a', 0x011}, {'g', 0x010}, {0x7f, 0x080}, {0x80, 0x000}, {0xff, 0x000},
        };
        if (!table || table[-1] != 0) {
            printf("  [FAIL] _Getpctype missing table or EOF slot\n");
            fails++;
        } else {
            for (const auto& item : expected) {
                if (table[item.ch] != item.mask) {
                    printf("  [FAIL] _Getpctype[0x%02x] = 0x%x, want 0x%x\n",
                           item.ch, unsigned(short(table[item.ch])), unsigned(short(item.mask)));
                    fails++;
                }
            }
            constexpr short dinkumware_isspace = 0x144; // _CN|_SP|_XS
            if (!(table[' '] & dinkumware_isspace) || !(table['\t'] & dinkumware_isspace) ||
                !(table['\n'] & dinkumware_isspace) || (table['A'] & dinkumware_isspace)) {
                printf("  [FAIL] _Getpctype does not satisfy Dinkumware isspace mask 0x144\n");
                fails++;
            }
        }
    } else { printf("  [FAIL] not registered: _Getpctype\n"); fails++; }

    if (HleFn lower = Hle::lookup(nid_hash("_Getptolower")); lower) {
        const auto* table = reinterpret_cast<const short*>(lower(0, 0, 0, 0, 0, 0));
        if (!table || table[-1] != -1 || table['A'] != 'a' || table['a'] != 'a' || table[0xff] != 0xff) {
            printf("  [FAIL] _Getptolower C-locale/EOF table contract\n");
            fails++;
        }
    } else { printf("  [FAIL] not registered: _Getptolower\n"); fails++; }

    if (HleFn upper = Hle::lookup(nid_hash("_Getptoupper")); upper) {
        const auto* table = reinterpret_cast<const short*>(upper(0, 0, 0, 0, 0, 0));
        if (!table || table[-1] != -1 || table['a'] != 'A' || table['A'] != 'A' || table[0xff] != 0xff) {
            printf("  [FAIL] _Getptoupper C-locale/EOF table contract\n");
            fails++;
        }
    } else { printf("  [FAIL] not registered: _Getptoupper\n"); fails++; }

    if (HleFn fn = Hle::lookup(nid_hash("getpid"))) {
        uint64_t v = fn(0, 0, 0, 0, 0, 0);
        if (v == 0) { printf("  [FAIL] getpid returned kernel-special pid 0\n"); fails++; }
    } else { printf("  [FAIL] not registered: getpid\n"); fails++; }

    if (HleFn fn = Hle::lookup(nid_hash("scePthreadGetthreadid"))) {
        const uint64_t main_id = fn(0, 0, 0, 0, 0, 0);
        const uint64_t main_id_again = fn(0, 0, 0, 0, 0, 0);
        std::array<std::atomic<uint64_t>, 2> worker_id{};
        std::array<std::atomic<uint64_t>, 2> worker_id_again{};
        std::array<std::thread, 2> workers;
        for (size_t i = 0; i < workers.size(); ++i) {
            workers[i] = std::thread([&, i] {
                worker_id[i].store(fn(0, 0, 0, 0, 0, 0), std::memory_order_relaxed);
                worker_id_again[i].store(fn(0, 0, 0, 0, 0, 0), std::memory_order_relaxed);
            });
        }
        for (auto& worker : workers) worker.join();
        if (main_id == 0 || main_id > std::numeric_limits<int32_t>::max() ||
            main_id != main_id_again) {
            printf("  [FAIL] scePthreadGetthreadid is not a stable positive guest ID on the main thread\n");
            fails++;
        }
        const uint64_t worker0 = worker_id[0].load(std::memory_order_relaxed);
        const uint64_t worker1 = worker_id[1].load(std::memory_order_relaxed);
        if (worker0 == 0 || worker1 == 0 ||
            worker0 > std::numeric_limits<int32_t>::max() ||
            worker1 > std::numeric_limits<int32_t>::max() ||
            worker0 != worker_id_again[0].load(std::memory_order_relaxed) ||
            worker1 != worker_id_again[1].load(std::memory_order_relaxed) ||
            worker0 == main_id || worker1 == main_id || worker0 == worker1) {
            printf("  [FAIL] scePthreadGetthreadid does not assign stable, distinct guest IDs\n");
            fails++;
        }
    } else { printf("  [FAIL] not registered: scePthreadGetthreadid\n"); fails++; }

    if (fails) { printf("== FAIL: %d function(s) not registered ==\n", fails); return 1; }
    printf("== PASS: all %zu checked HLE functions registered ==\n", sizeof(names)/sizeof(names[0]) + 1);
    return 0;
}
