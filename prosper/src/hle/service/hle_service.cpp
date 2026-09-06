// hle_service.cpp — HLE of PS5 system services (user, NP/online, mouse, app content,
// dialogs). Bring-up policy: openers return a valid positive handle; queries zero their
// output struct and report a sane "not signed in / no device" state and success, so the
// game gets consistent values instead of uninitialized memory.
// (Game-controller input — libScePad — moved to hle_pad.cpp with a real host backend.)
#include "hle/dispatch/dispatch.hpp"
#include "hle/service/hle_addcontent.hpp"
#include "hle/fs/save_paths.hpp"   // per-title save roots (#2734)
#include "hle/util/hle_json2.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/sce_errno.hpp"   // libkernel error encoding (libSceRandom reject arms)
#include "diagnostics/env_numeric.hpp"   // #3267: a typo must not unregister a default-ON NID family
#include "hle/dispatch/callback_fs.hpp"
#include "hle/input/ime_input.hpp"
#include "hle/service/platform_ui.hpp"
#include "hle/video/video_backend.hpp"   // sceAvPlayer -> host hardware-decode backend (#705)
#include "hle/video/h264_sps.hpp"        // SPS/VUI extraction for GetPictureInfo (#2898)
#include "gpu/texture/guest_texture_layout.hpp" // exact HLE-produced sampled-linear layouts
#include "host/platform/posix_shim.hpp"   // Darwin process_vm_readv shim + asm portability
#include "host/image/boot_program.hpp"  // guest_module_name: is a callback target guest code?
#include "host/platform/lifecycle.hpp"  // cooperative stop when the guest reports its own crash (#3119)
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <atomic>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <thread>
#include <deque>
#include <filesystem>
#include <mutex>
#include <new>          // std::bad_alloc — the guest file-replacement buffer is guest-sized (#1955)
#include <set>
#include <unordered_map>
#include <vector>
#include <string>
#ifdef _WIN32
#include <direct.h>     // _mkdir (SaveDataMemory persistence dir)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>   // mkdir
#include <sys/uio.h>    // process_vm_readv: fault-contained diagnostic snapshots
#include <sys/random.h> // getentropy: the host CSPRNG behind sceRandomGetRandomNumber
#include <unistd.h>
#endif
#ifdef _WIN32
#include <bcrypt.h>     // BCryptGenRandom (prosper_core already links bcrypt on Windows)
#endif

namespace prosper {

#ifdef _WIN32
extern "C" uint64_t prosper_call_guest_sysv4(uint64_t fn, uint64_t a0, uint64_t a1,
                                               uint64_t a2, uint64_t a3);
#endif

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define PW(x) ((void*)(uintptr_t)(x))

namespace { std::atomic<uint64_t> g_handle{1}; }

// Diagnostic logging for the Sony service families in this file, gated on PROSPER_SVCLOG=1 (same
// pattern as PROSPER_FILELOG/[file]). Dumps call args and a bounded hexdump of pointer-shaped args
// so PS5-only ABIs with no Kyty/shadPS4 reference can be pinned from live captures instead of
// guessed.
namespace {
bool svclog() { static int v = getenv("PROSPER_SVCLOG") ? 1 : 0; return v; }
bool svc_ptrish(uint64_t v) { return v >= 0x10000 && v < 0x7fffffffffffull; }
bool svc_copy_bytes(uint64_t src, void* dst, size_t bytes) {
    if (!src || !dst || !bytes || src > UINT64_MAX - (bytes - 1)) return false;
#ifdef _WIN32
    SIZE_T copied = 0;
    return ReadProcessMemory(GetCurrentProcess(), (const void*)(uintptr_t)src, dst, bytes, &copied) &&
           copied == bytes;
#else
    iovec local{dst, bytes};
    iovec remote{(void*)(uintptr_t)src, bytes};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == (ssize_t)bytes;
#endif
}
bool svc_write_bytes(uint64_t dst, const void* src, size_t bytes) {
    if (!dst || !src || !bytes || dst > UINT64_MAX - (bytes - 1)) return false;
#ifdef _WIN32
    SIZE_T copied = 0;
    return WriteProcessMemory(GetCurrentProcess(), (void*)(uintptr_t)dst, src, bytes, &copied) &&
           copied == bytes;
#else
    iovec local{const_cast<void*>(src), bytes};
    iovec remote{(void*)(uintptr_t)dst, bytes};
    return process_vm_writev(getpid(), &local, 1, &remote, 1, 0) == (ssize_t)bytes;
#endif
}
size_t svc_copy_words(uint64_t src, uint64_t* dst, size_t words) {
    const size_t bytes = words * sizeof(uint64_t);
#ifdef _WIN32
    SIZE_T copied = 0;
    ReadProcessMemory(GetCurrentProcess(), (const void*)(uintptr_t)src, dst, bytes, &copied);
    return (size_t)copied / sizeof(uint64_t);
#else
    iovec local{dst, bytes};
    iovec remote{(void*)(uintptr_t)src, bytes};
    const ssize_t copied = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
    return copied > 0 ? (size_t)copied / sizeof(uint64_t) : 0;
#endif
}
void svc_log(const char* fn, uint64_t a0, uint64_t a1, uint64_t a2,
             uint64_t a3, uint64_t a4, uint64_t a5, int dump_words = 8) {
    if (!svclog()) return;
    fprintf(stderr, "[svc] %s(%#" PRIx64 ", %#" PRIx64 ", %#" PRIx64
                    ", %#" PRIx64 ", %#" PRIx64 ", %#" PRIx64 ")\n",
            fn, a0, a1, a2, a3, a4, a5);
    const uint64_t args[6] = { a0, a1, a2, a3, a4, a5 };
    for (int i = 0; i < 6; i++) {
        if (!svc_ptrish(args[i])) continue;
        // Never read across the arg's 4 KiB page end: an out-param can be a tiny heap block whose
        // page neighbor is unmapped, and a diagnostic must not be able to fault the boot.
        uint64_t page_left = 0x1000 - (args[i] & 0xfff);
        int words = (int)(page_left / 8); if (words > dump_words) words = dump_words;
        if (words > 8) words = 8;
        uint64_t q[8]{};
        words = (int)svc_copy_words(args[i], q, (size_t)words);
        if (!words) continue;
        fprintf(stderr, "[svc]   a%d ->", i);
        for (int w = 0; w < words; w++) fprintf(stderr, " %016" PRIx64, q[w]);
        fprintf(stderr, "\n");
    }
}
}

// PS5 Game Intent is how the shell starts a title at a selected activity instead of its default
// entry point. A host frontend can model that user action by setting an activity id before boot;
// without it the console truthfully has no pending intent. Keep the payload opaque to the guest:
// titles consume it through sceNpGameIntentGetPropertyValueString, as on the console.
// CONFIDENCE: HIGH — ABI/error contracts match Kyty; event id, type, key, and offsets are also
// independently exercised by Sonic's eboot and its launchActivity declaration in param.json.
namespace {
constexpr uint64_t NP_GAME_INTENT_ERROR_INVALID_ARGUMENT = 0x80553804ull;
constexpr uint64_t NP_GAME_INTENT_ERROR_INTENT_NOT_FOUND = 0x80553806ull;
constexpr uint64_t NP_GAME_INTENT_ERROR_VALUE_NOT_FOUND  = 0x80553807ull;
constexpr size_t NP_GAME_INTENT_TYPE_OFFSET     = 12;
constexpr size_t NP_GAME_INTENT_TYPE_SIZE       = 33;
constexpr size_t NP_GAME_INTENT_DATA_OFFSET     = 308;
constexpr size_t NP_GAME_INTENT_DATA_SIZE       = 16392;
constexpr uint32_t SYSTEM_SERVICE_EVENT_GAME_INTENT = 0x10000017u;

std::atomic<bool> g_gameintent_initialized{false};
std::atomic<bool> g_gameintent_event_delivered{false};
std::atomic<uint64_t> g_gameintent_data_ptr{0};

// Bound host-controlled text to the opaque intent payload capacity. The property accessor separately
// checks the caller's buffer, so a title with a smaller value limit gets an error, never truncation.
const char* gameintent_activity_id(size_t* length = nullptr) {
    const char* value = getenv("PROSPER_GAME_INTENT_ACTIVITY_ID");
    if (!value) return nullptr;
    size_t n = 0;
    while (n < NP_GAME_INTENT_DATA_SIZE && value[n]) ++n;
    if (n == 0 || n == NP_GAME_INTENT_DATA_SIZE) return nullptr;
    if (length) *length = n;
    return value;
}
}

// --- user service ---
HLE(s_user_initial)   { if (a0) *(int32_t*)PW(a0) = 1; return 0; }           // GetInitialUser -> userId 1
HLE(s_user_idlist)    { if (a0) { int32_t* p = (int32_t*)PW(a0); p[0] = 1; for (int i = 1; i < 4; i++) p[i] = -1; } return 0; }
// Bounded, non-padding write (strncpy would zero-pad the whole a2-byte buffer -> a
// stack-smash if a2 is large/garbage). snprintf writes only the string + NUL.
HLE(s_user_name)      { if (a1) snprintf((char*)PW(a1), a2 ? (size_t)a2 : 17, "%s", "Player"); return 0; }
HLE(s_user_int_out)   { if (a1) *(int32_t*)PW(a1) = 0; return 0; }           // accessibility getters -> 0
HLE(s_user_age)       { if (a1) *(int32_t*)PW(a1) = 18; return 0; }          // GetAgeLevel -> adult (no restriction)
// sceUserServiceGetUserNumber(userId, number): each local user has a stable controller/user number.
// Sonic imports the PS5 NID directly and consumes the out-param during its user bootstrap.  Returning
// success from the fallback stub without writing it leaves the caller's stack sentinel in place and
// prevents the frontend from reaching its pad/audio initialization.  This single-user model exposes
// default user 1 as number 1, consistent with the rest of the UserService handlers above.
HLE(s_user_number)    { if (a1) *(int32_t*)PW(a1) = 1; return 0; }
// --- libSceVideodec2 ---------------------------------------------------------------------------
// These ABI layouts/NIDs are the PS5 3.20 interfaces.  Decoding is intentionally a no-picture
// implementation for now: it consumes each access unit, reports a valid lifecycle, and lets the
// movie reach EOF instead of faking video pixels.  Critically, every successful query/open writes
// all output fields.  The former generic success stubs left a null compute queue and crashed libc
// at +0xfe66 on its first queue use.
//
// WHO ACTUALLY EXERCISES THIS, as recorded rather than as assumed.  A header line here used to read
// "Sonic's CRI Mana movie backend uses the native VideoDec2 compute queue and decoder lifecycle."
// It came from #1368 (8b37be95) — the same commit that added the unexplained compute_queue check
// below and an equally unsupported "Sonic Origins' CRI Mana backend passes it", which was already
// withdrawn.  No Videodec2 evidence for Sonic exists anywhere in docs/ or COMPATIBILITY.md, so it is
// withdrawn on the same basis rather than left to be inherited a third time.  What IS recorded:
//   - QueryComputeMemoryInfo only — Dragon Quest VII Reimagined and the UE4 bring-up.
//   - The full decoder lifecycle — Tales of Graces f Remastered (PPSA19991) via criMvPly, and only
//     after #1687; before that no title in this repo's history had ever reached CreateDecoder.
namespace {
// The current no-picture backend has no hardware decoder workspace: it only needs a stable,
// caller-owned address for the opaque compute-queue identity.  Keep one guest page so clients that
// require a non-null allocation can follow the real query/allocate lifecycle without reserving the
// 16 MiB placeholder used by software-decoding implementations.  That placeholder exhausted
// Sonic Origins' 80 MiB CRI CPU/GPU pool before its movie workers could allocate their own buffers.
constexpr uint64_t VDEC_MIN_MEMORY = 64ull << 10;
constexpr uint64_t VDEC_ERR_STRUCT = 0x811d0101ull;
constexpr uint64_t VDEC_ERR_ARG = 0x811d0102ull;
constexpr uint64_t VDEC_ERR_DECODER = 0x811d0103ull;
constexpr uint64_t VDEC_ERR_MEMORY_SIZE = 0x811d0104ull;
constexpr uint64_t VDEC_ERR_MEMORY_PTR = 0x811d0105ull;
constexpr uint64_t VDEC_ERR_FRAME_SIZE = 0x811d0106ull;
constexpr uint64_t VDEC_ERR_FRAME_PTR = 0x811d0107ull;
constexpr uint64_t VDEC_ERR_CONFIG = 0x811d0200ull;
constexpr uint64_t VDEC_ERR_PIPE = 0x811d0201ull;
constexpr uint64_t VDEC_ERR_QUEUE = 0x811d0202ull;
constexpr uint64_t VDEC_ERR_RESOURCE = 0x811d0203ull;
constexpr uint64_t VDEC_ERR_INPUT_DEPTH = 0x811d0206ull;
constexpr uint64_t VDEC_ERR_DPB = 0x811d0209ull;
constexpr uint64_t VDEC_ERR_DIMS = 0x811d020aull;

struct VdecComputeMemory { uint64_t size, shared_size, shared; };
struct VdecComputeConfig {
    uint64_t size; uint16_t pipe, queue; uint8_t check_memory, reserved0; uint16_t reserved1;
};
struct VdecConfig {
    uint64_t size; uint32_t resource, codec, profile, max_level;
    int32_t max_width, max_height, max_dpb; uint32_t input_depth;
    uint64_t compute_queue, affinity; int32_t priority;
    uint8_t optimize, check_memory, reserved0, reserved1; uint64_t extra;
};
struct VdecMemory {
    uint64_t size, cpu_size, cpu, gpu_size, gpu, shared_size, shared, max_frame_size;
    uint32_t frame_alignment, reserved;
};
struct VdecInput { uint64_t size, data, data_size, pts, dts, attached; };
struct VdecFrame { uint64_t size, data, data_size; uint8_t accepted, pad[7]; };
struct VdecOutput {
    uint64_t size; uint8_t valid, error, pictures, discarded;
    uint32_t codec, width, pitch, height; uint64_t frame, frame_size;
    uint32_t format, pitch_bytes;
};
static_assert(sizeof(VdecComputeMemory) == 24 && sizeof(VdecComputeConfig) == 16);
static_assert(sizeof(VdecConfig) == 72 && sizeof(VdecMemory) == 72);
static_assert(sizeof(VdecInput) == 48 && sizeof(VdecFrame) == 32 && sizeof(VdecOutput) == 56);
// LIVE CONFIRMATION of the two structs above, from the first title to reach the decoder lifecycle
// (Tales of Graces f Remastered PPSA19991, criMvPly, PROSPER_SVCLOG; see #1658). These were written
// in #1368 without title evidence; a real guest now writes values that are meaningful at nearly
// every offset, which is much stronger than a matching size:
//
// The two calls are labelled, because they are NOT the same dump: everything down to +0x28 appears
// in both, while affinity/priority differ between them and are therefore CreateDecoder's values.
//
//   VdecConfig  +0x00 size=0x48  +0x08 resource=1     +0x0c codec=1 (AVC)          [both calls]
//               +0x10 profile=100 (High)              +0x14 max_level=41 (4.1)     [both calls]
//               +0x18 max_width=1920                  +0x1c max_height=1088        [both calls]
//               +0x20 max_dpb=-1 (auto)               +0x24 input_depth=4          [both calls]
//               +0x28 compute_queue = 0                                            [sizing query]
//               +0x28 compute_queue = the allocated handle                         [CreateDecoder]
//               +0x30 affinity=0x1fff                 +0x38 priority=700           [CreateDecoder]
//               (the sizing query's affinity/priority are 0 / -1)
//   VdecMemory  +0x00 size=0x48, then alternating size/pointer slots at +0x08..+0x38
//
// NOTE the observation boundary: svc_log clamps to 8 qwords, so despite both call sites asking for
// 9, NOTHING at +0x40 was ever dumped — VdecConfig::extra and VdecMemory::frame_alignment/reserved
// are unobserved, not confirmed. The table stops at +0x38 for that reason. (The clamp is silent and
// is itself instrument-trap-13 shaped; filed rather than fixed here to keep this PR narrow.)
//
// Most of those pairings are FORCED, not merely consistent, which is what makes this evidence and
// not a coincidence: swap profile/max_level and you get profile=41 (not a valid H.264 profile_idc)
// with level=100 (not a valid level_idc); swap max_dpb/input_depth and input_depth is negative;
// swapping width/height would make the movie portrait 1088x1920.
//
// One pair is NOT discriminated and is called out rather than glossed: `resource` and `codec` are
// BOTH 1 in this title, so this evidence cannot tell their order apart. It rests on #1368 there, and
// a title using a non-AVC codec or a different resource selector would settle it in one line of log.
// So: CONFIDENCE: HIGH on VdecConfig's layout except the resource/codec pair, which stays MED.
//
// VdecMemory needs the SAME discipline applied to it, and it does not survive it as well. "The guest
// echoed back the sizes we wrote" is near-tautological: this file wrote one identical constant into
// cpu_size, gpu_size, shared_size AND max_frame_size, so the echo is byte-identical under any
// permutation of those four, and the three pointer slots are three distinct guest addresses carrying
// no role marker. What the echo genuinely proves is the SHAPE — size=0x48 at +0x00 and alternating
// size/pointer slots thereafter — which is real and non-trivial. So: CONFIDENCE: HIGH on VdecMemory's
// shape, MED on which of the four size roles and three pointer roles sits in which slot; those still
// rest on #1368. (Now that the sizing query reports a max_frame_size distinct from the other three,
// the next boot's Decode call discriminates that one field for free.)
//
// VdecInput's first five fields are confirmed (size=0x30, data, data_size) with the same caveat on
// the last two: pts and dts are BOTH -1 here and are mutually undiscriminated. VdecFrame's first
// three are confirmed (size=0x20, data, data_size). Only the SIZE of VdecOutput is confirmed (0x38)
// — its interior is written by us, not the guest, and the AVC picture-info layout remains open
// (#1658).
//
// One naming caveat, deliberately not "fixed": VdecInput's last field, named `attached` here,
// carried a clean 0,1,2,...  sequence across consecutive access units rather than a flag or a
// pointer. That is evidence it may be an access-unit index, not an attachment. Nothing depends on
// the name today and no title evidence settles it, so it is recorded rather than renamed.

std::mutex g_vdec_mx;
std::unordered_map<uint64_t, uint32_t> g_vdec_codecs;
// #2270 access-unit decoders, one per guest decoder handle. `id` is the backend's decoder id, or
// -1 when the backend refused the codec (or there is no backend at all) -- `opened` distinguishes
// "not tried yet" from "tried and refused", so the refusal is announced ONCE rather than on every
// access unit or, worse, never. `no_picture_run` counts consecutive access units that produced no
// picture from a decoder that DID open: a real decoder needs a handful of units before its first
// frame, so a long run of them is a decoder that is never going to produce one, and that is the
// exact shape #2270 was filed about -- indistinguishable, without this, from "no frame ready yet".
// `announced` survives Reset while `opened` does not, and that difference is the finding it fixes:
// Reset re-arms the open so a supported codec gets a fresh decoder, but a REFUSED codec would then
// re-print its two-line NO DECODER banner on every reset cycle. A title that resets in a retry loop
// turns the fail-visible path into a flood, which is how fail-visible paths get muted (#2571 N3).
struct VdecAu { int id = -1; bool opened = false; bool announced = false; unsigned no_picture_run = 0; };
std::unordered_map<uint64_t, VdecAu> g_vdec_au;

// Per-decoded-picture metadata for sceVideodec2GetPictureInfo (#2898).
//
// GetPictureInfo takes NO decoder handle -- its arguments are (outputInfo*, pictureInfo*,
// 0, 0) at every observed call site (PPSA06367 x2, PPSA29343 x3) -- so the only way to
// know which picture to describe is the one thing that identifies it: outputInfo->frame,
// the pointer Decode wrote into the caller's output struct. Keying on it means a title
// that recycles frame buffers collapses those pictures onto one metadata entry; SPS/VUI
// values rarely change within a stream, so the practical cost is negligible and it is
// recorded rather than hidden (CONFIDENCE: MED on cross-picture identity, HIGH on the
// offsets themselves).
//
// `record` is a small prosper-owned block handed to the guest through pictureInfo+0x20.
// Guest evidence about its SHAPE (#2898, refined by live boot once the pointer was no
// longer null -- the fault moved from "deref of the null pointer field" to "walk of the
// block behind it", which is what named the extra indirection):
//
//   P = *(void**)record;                    // +0x00 is a POINTER to the payload block
//   string_assign(s, (char*)P + 0x08);      // C-string at payload+0x08
//   if (payload[+0x30]) use qword payload[+0x28] else built-in default text;
//
//   (PPSA29343 never dereferences; it compares `record` values for identity across its
//   per-decoder arrays, which any stable per-picture address satisfies.)
//
// Our block therefore points its first qword at its own +0x08, where the payload lives:
// an EMPTY string and flag = 0, which takes the guest's default-text path -- nothing is
// fabricated. Records are allocated once per distinct frame buffer and deliberately
// NEVER freed while the process lives -- freeing them would turn Beast's stored keys
// (and any reference the guest kept) dangling, which is worse than a bounded few hundred
// bytes.
struct VdecPicMeta {
    bool has_meta = false;
    prosper::h264::SpsPictureMeta meta;
    void* record = nullptr;
};
constexpr size_t kVdecInfoRecordSize = 0x48;  // 8-byte self-pointer + 0x40 payload
std::unordered_map<uint64_t, VdecPicMeta> g_vdec_picmeta;

// A rejected decoder config is a hard stop for a title's movie playback, and the guest only prints the
// bare SCE code — 0x811d0200 covers TWO different conditions here, so "err=0x811d0200" alone cannot tell
// anyone which. Name the failing field and its observed value, unconditionally but rate-limited: this
// fires at most a handful of times per boot and is the difference between a one-run diagnosis and a
// guess. (Tales of Graces f / criMvPly stopped exactly here, on `compute_queue`, which is the
// evidence the split below rests on; see #1658.)
uint64_t vdec_reject(const char* field, uint64_t observed, uint64_t err) {
    // Rate-limit PER FIELD, not globally. One shared counter meant a caller looping on an early
    // condition could burn the whole budget and silence a later, different rejection — including the
    // one a run was started to capture. `field` is always a string literal, so the pointer is a stable
    // key and the map stays bounded by the number of call sites.
    static std::mutex mx;
    static std::unordered_map<const char*, int> shown;
    {
        std::lock_guard<std::mutex> lk(mx);
        if (shown[field]++ >= 4) return err;
    }
    fprintf(stderr, "[vdec] decoder config REJECTED: %s = 0x%llx -> err=0x%llx\n",
            field, (unsigned long long)observed, (unsigned long long)err);
    return err;
}
// `require_queue` splits the compute-queue requirement between the two callers of this validator.
// See the long note at the check itself for why the two differ.
uint64_t vdec_validate_config(const VdecConfig* c, bool require_queue) {
    if (!c) return VDEC_ERR_ARG;
    if (c->size != sizeof(*c)) return vdec_reject("size", c->size, VDEC_ERR_STRUCT);
    if (c->resource != 1) return vdec_reject("resource", c->resource, VDEC_ERR_RESOURCE);
    if (c->reserved0 || c->reserved1)
        return vdec_reject("reserved0/1", ((uint64_t)c->reserved0 << 8) | c->reserved1,
                           VDEC_ERR_CONFIG);
    if (!c->input_depth) return vdec_reject("input_depth", c->input_depth, VDEC_ERR_INPUT_DEPTH);
    if (c->max_dpb < -1 || c->max_dpb == 0)
        return vdec_reject("max_dpb", (uint64_t)(int64_t)c->max_dpb, VDEC_ERR_DPB);
    if (c->max_width < -1 || c->max_height < -1 || !c->max_width || !c->max_height)
        return vdec_reject("max_width/height",
                           ((uint64_t)(uint32_t)c->max_width << 32) | (uint32_t)c->max_height,
                           VDEC_ERR_DIMS);
    // THE SPLIT (#1658). A compute queue is required to CREATE a decoder and is NOT required to ask
    // how large one would be.
    //
    // The check arrived in #1368 (8b37be95) with no comment, no rationale and no test — defensive,
    // not derived — and applied to both callers. The predecessor comment here predicted exactly how
    // that would fail ("most likely to reject a caller that queries memory sizes BEFORE it allocates
    // a compute queue") and pre-registered this split as the evidenced move if a live log ever named
    // the field. A Tales of Graces f (PPSA19991) boot then named it: 7 calls to
    // sceVideodec2QueryDecoderMemoryInfo, all 7 rejected on `compute_queue = 0`, zero calls to
    // sceVideodec2CreateDecoder — so the guest is sizing a decoder it has not built yet, which is
    // precisely when it cannot hold a queue handle. Everything else in that config validates
    // (size=0x48, resource=1, AVC High@4.1, 1920x1088, max_dpb=-1 auto, input_depth=4).
    //
    // Rejecting the sizing query also protected no size contract: s_videodec2_query_decoder_memory
    // ignores compute_queue entirely, so the rejection only failed EARLIER than succeeding would.
    //
    // The requirement stays on CreateDecoder, which genuinely needs a queue to build a decoder on,
    // and which is where a caller that truly has none must still fail visibly rather than receive an
    // invented handle.
    //
    // CONFIDENCE: HIGH that the sizing query must not require it — live title evidence, and
    // *prosper's* query cannot use the field. (That second clause argues from our own implementation,
    // not from Sony's: a real library could legitimately consult a queue handle for queue-specific
    // alignment. The HIGH rests on the live evidence alone.)
    //
    // CONFIDENCE: MED that CreateDecoder must require it. The same title now DOES reach CreateDecoder
    // and supplies a real queue there — but that proves the guest HAS one, not that the library
    // demands one, and those are different claims. So this half remains the #1368 default,
    // deliberately kept fail-visible rather than relaxed on the strength of a caller being
    // well-behaved.
    if (require_queue && !c->compute_queue)
        return vdec_reject("compute_queue", c->compute_queue, VDEC_ERR_CONFIG);
    return 0;
}
// How many consecutive no-picture access units from an OPEN decoder are still plausible before the
// run is announced as a stall. A conforming decoder emits its first picture within a small multiple
// of its reordering depth; the guest's own config asks for input_depth=4 and max_dpb=-1 (auto), and
// the observed streams are IDR-first. 64 is two orders of magnitude above that and still fires
// within about a second of movie time, so it cannot mistake warm-up for a stall in either direction.
constexpr unsigned kVdecNoPictureAlarm = 64;

// The fail-visible announcement #2270 exists for. Everything below this returns SCE_OK with no
// picture, which is the CORRECT answer for "the decoder needs more input" and the catastrophic one
// for "there is no decoder" -- the guest cannot tell them apart, so it waits forever with nothing in
// the log. Print the codec AND the first bytes of the access unit: the codec field alone does not
// name a bitstream, while the head does (00 00 00 01 09 / 67 is Annex-B H.264 with an access-unit
// delimiter and an SPS; a VP9 key frame carries the sync code 49 83 42 at bytes 1..3).
void vdec_no_decoder_warning(uint32_t codec, const VdecInput* input, const char* why) {
    char head[64] = {0};
    if (input && input->data && input->data_size) {
        const auto* au = (const uint8_t*)(uintptr_t)input->data;
        const uint64_t n = input->data_size < 16 ? input->data_size : 16;
        for (uint64_t i = 0; i < n; ++i) snprintf(head + i * 3, 4, "%02x ", au[i]);
    }
    fprintf(stderr,
            "[vdec2] sceVideodec2Decode: NO DECODER (%s) for codec=%u -- au_bytes=%llu head=%s\n"
            "[vdec2]   Every call will report SCE_OK with NO PICTURE, which a title cannot "
            "distinguish from \"no frame ready yet\": if it waits for a decoded frame it will wait "
            "forever. This is #2270; the head bytes above name the bitstream a decoder must accept.\n",
            why, codec, input ? (unsigned long long)input->data_size : 0ull, head);
}

// PROSPER_VDEC2_DUMP_DIR -- the instrument that lets a decoded picture be CHECKED rather than
// believed (#2270). It writes two files into the named directory:
//
//   au.bin     every access unit the guest submitted, concatenated in submission order. The
//              observed streams are Annex-B (start-code delimited), so the concatenation is itself
//              a playable elementary stream: `ffmpeg -i au.bin -f rawvideo -pix_fmt nv12 ref.nv12`.
//   pic.nv12   the exact bytes prosper wrote into the guest's frame buffer, same order.
//
// `cmp ref.nv12 pic.nv12` is then an INDEPENDENT decode of the guest's own bitstream against ours.
// That distinction is the reason this exists: a decoded picture that merely "looks like a movie" is
// not evidence — this project has a recorded trap where a plausible palette convinced the eye and a
// gradient beat real content on a metric. H.264 reconstruction is normatively exact, so byte
// equality against another decoder's output is a claim that cannot be faked by a wrong chroma plane,
// a swapped U/V, or a stale reference frame, all of which produce something that still looks filmic.
//
// BOUNDED, because it is not: a 1920x1088 NV12 picture is 3.1 MB and these titles feed thousands.
// PROSPER_VDEC2_DUMP_FRAMES (default 16) caps both files together, so a full-rate movie costs 50 MB
// rather than 5 GB. Nothing is written unless the directory variable is set.
struct VdecDump {
    FILE* au = nullptr;
    FILE* pic = nullptr;
    unsigned limit = 16;
    unsigned aus = 0, pics = 0;
    std::mutex mx;
};
VdecDump& vdec_dump() {
    // A std::mutex member makes VdecDump immovable, so the usual `static X x = []{...}()` form does
    // not compile here. Initialise in place under a once-flag instead.
    static VdecDump d;
    static std::once_flag once;
    std::call_once(once, [] {
        const char* dir = getenv("PROSPER_VDEC2_DUMP_DIR");
        if (!dir || !*dir) return;
        if (const char* n = getenv("PROSPER_VDEC2_DUMP_FRAMES")) {
            // `if (parsed)` refuses only a value that parses to 0. `=-1` saturated to ULONG_MAX and
            // lifted the 16-frame cap this block's own comment describes as the difference between
            // 50 MB and 5 GB on disk. Base 0 was the pre-existing grammar, so keep `0x` (#3267 B1).
            const uint64_t parsed = prosper::diag::env_u64_or_default_auto_capped(
                "PROSPER_VDEC2_DUMP_FRAMES", n, 16ull, UINT32_MAX, "frames");
            if (parsed) d.limit = (unsigned)parsed;
        }
        const std::string base(dir);
        d.au = fopen((base + "/au.bin").c_str(), "wb");
        d.pic = fopen((base + "/pic.nv12").c_str(), "wb");
        fprintf(stderr, "[vdec2] dump: au=%s pic=%s limit=%u frames (#2270)\n",
                d.au ? "open" : "FAILED", d.pic ? "open" : "FAILED", d.limit);
    });
    return d;
}
void vdec_dump_au(const uint8_t* au, uint64_t bytes) {
    VdecDump& d = vdec_dump();
    if (!d.au || !au || !bytes) return;
    std::lock_guard<std::mutex> lk(d.mx);
    if (d.aus >= d.limit) return;
    fwrite(au, 1, (size_t)bytes, d.au); fflush(d.au);
    ++d.aus;
}
void vdec_dump_picture(const uint8_t* nv12, size_t bytes, uint32_t w, uint32_t h) {
    VdecDump& d = vdec_dump();
    if (!d.pic || !nv12 || !bytes) return;
    std::lock_guard<std::mutex> lk(d.mx);
    if (d.pics >= d.limit) return;
    if (d.pics == 0)
        fprintf(stderr, "[vdec2] dump: pictures are %ux%u NV12, %zu bytes each -- compare with\n"
                        "[vdec2]   ffmpeg -i <dir>/au.bin -f rawvideo -pix_fmt nv12 <dir>/ref.nv12\n"
                        "[vdec2]   cmp <dir>/ref.nv12 <dir>/pic.nv12\n", w, h, bytes);
    fwrite(nv12, 1, bytes, d.pic); fflush(d.pic);
    ++d.pics;
}

void vdec_no_picture(VdecFrame* frame, VdecOutput* out, uint32_t codec) {
    out->valid = out->error = out->pictures = out->discarded = 0;
    out->codec = codec; out->width = out->pitch = out->height = 0;
    out->frame = frame ? frame->data : 0;
    out->frame_size = frame ? frame->data_size : 0;
    out->format = out->pitch_bytes = 0;
}
}

HLE(s_videodec2_query_compute_memory) {
    svc_log("sceVideodec2QueryComputeMemoryInfo", a0, a1, a2, a3, a4, a5, 3);
    auto* info = (VdecComputeMemory*)PW(a0);
    if (!info) return VDEC_ERR_ARG;
    if (info->size != sizeof(*info)) return VDEC_ERR_STRUCT;
    info->shared_size = VDEC_MIN_MEMORY; info->shared = 0;
    return 0;
}
HLE(s_videodec2_allocate_compute_queue) {
    svc_log("sceVideodec2AllocateComputeQueue", a0, a1, a2, a3, a4, a5, 3);
    auto* config = (const VdecComputeConfig*)PW(a0);
    auto* memory = (const VdecComputeMemory*)PW(a1);
    auto* out = (uint64_t*)PW(a2);
    if (!config || !memory || !out) return VDEC_ERR_ARG;
    if (config->size != sizeof(*config) || memory->size != sizeof(*memory)) return VDEC_ERR_STRUCT;
    // These share 0x811d0200/0x811d02xx with the decoder-config validator, so a bare guest error code
    // cannot say which call produced it. Name them too, or a run that fails HERE prints nothing.
    if (config->reserved0 || config->reserved1)
        return vdec_reject("computeQueue.reserved0/1",
                           ((uint64_t)config->reserved0 << 8) | config->reserved1, VDEC_ERR_CONFIG);
    if (config->pipe > 4) return vdec_reject("computeQueue.pipe", config->pipe, VDEC_ERR_PIPE);
    if (config->queue > 7) return vdec_reject("computeQueue.queue", config->queue, VDEC_ERR_QUEUE);
    if (memory->shared_size < VDEC_MIN_MEMORY)
        return vdec_reject("computeQueue.shared_size", memory->shared_size, VDEC_ERR_MEMORY_SIZE);
    if (!memory->shared)
        return vdec_reject("computeQueue.shared", memory->shared, VDEC_ERR_MEMORY_PTR);
    *out = memory->shared;
    if (svclog()) fprintf(stderr, "[svc]   Videodec2 compute queue -> 0x%llx (%llu bytes)\n",
                          (unsigned long long)*out, (unsigned long long)memory->shared_size);
    return 0;
}
HLE(s_videodec2_release_compute_queue) { return a0 ? 0 : VDEC_ERR_QUEUE; }
HLE(s_videodec2_query_decoder_memory) {
    svc_log("sceVideodec2QueryDecoderMemoryInfo", a0, a1, a2, a3, a4, a5, 9);
    auto* config = (const VdecConfig*)PW(a0); auto* info = (VdecMemory*)PW(a1);
    if (!config || !info) return VDEC_ERR_ARG;
    if (info->size != sizeof(*info)) return VDEC_ERR_STRUCT;
    // Pure sizing query: no compute queue required (#1658).
    uint64_t valid = vdec_validate_config(config, /*require_queue=*/false); if (valid) return valid;

    // max_frame_size is the buffer the GUEST allocates and hands straight back as VdecFrame.data for
    // every decoded picture. That coupling is observed, not assumed: this file reported
    // VDEC_MIN_MEMORY (0x10000) and the title's next VdecFrame.data_size came back as exactly
    // 0x10000.
    //
    // VDEC_MIN_MEMORY was derived as a compute-queue identity page (see its definition), never as a
    // decoder frame estimate, and it does not vary with resolution — a 4K request got the same
    // 64 KiB. That is safe ONLY while vdec_no_picture never writes into frame->data. The moment a
    // real decoder lands (#1688), writing a 1920x1088 NV12 picture into a 64 KiB guest allocation is
    // a ~3 MiB guest-heap overflow, and it would present as a decoder bug rather than as the sizing
    // answer that caused it. Report a size the guest can actually decode into, now, while doing so
    // costs nothing and cannot be mistaken for a decoder defect later.
    //
    // NV12 is the format PS5 video decode delivers (see video_backend.hpp): a full-resolution luma
    // plane plus a half-resolution interleaved chroma plane, each chroma dimension rounded UP —
    // `w*h + 2*ceil(w/2)*ceil(h/2)` — then rounded up to frame_alignment. The exact expression and
    // why the `*3/2` shorthand is NOT used are below, at the computation. An unreasonable request
    // yields an honestly unreasonable size that the guest's own allocator rejects visibly, which is
    // the behaviour we want.
    //
    // Dimensions may legitimately be -1 ("auto"). prosper has no evidence for what a real library
    // reports then, and a level-implied maximum would be invention, so that path keeps the
    // documented floor unchanged and is called out rather than papered over.
    //
    // CONFIDENCE: MED on the NV12 derivation — the format is established and the guest's observed
    // 1920x1088 (macroblock-rounded) request fits it exactly, but whether Sony's library adds
    // per-plane padding or a decoder-private tail is unproven. LOW on the auto-dimension path.
    // The cpu/gpu/shared workspace sizes stay at the floor: nothing here is evidence for those, and
    // inflating them has broken a title before (see VDEC_MIN_MEMORY's own note on the 16 MiB
    // placeholder exhausting a CRI pool).
    // The EXACT NV12 size comes from prosper::video::nv12_bytes(), which is the SAME function the
    // decoder backend uses to check the caller's buffer and to fill it. That shared call is the point:
    // this query tells the guest how large a frame buffer to allocate, and the backend writes into
    // the buffer the guest allocated from that answer — two rules would be a guest buffer sized by
    // one and written by the other, i.e. a heap overflow presenting as a decoder bug. The rule itself,
    // why `w*h*3/2` is wrong for an odd dimension, and the widen-before-arithmetic trap are all
    // written out at the function. (Instrument-trap 34; consolidated on #2571 review D1, which
    // pointed out the property had no test that could see it — `test_videodec2_decode` now has one.)
    constexpr uint32_t VDEC_FRAME_ALIGN = 0x100;
    uint64_t frame_bytes = 0;
    if (config->max_width > 0 && config->max_height > 0) {
        // Use the dimensions the guest ASKED FOR, which are macroblock-rounded (this title requests
        // 1088 for a 1080-line movie). That is not a safety margin to be trimmed later: H.264 codes
        // in 16x16 macroblocks, so a 1080-row picture is physically written as 1088 rows and cropped
        // for display via the SPS frame_cropping fields. Sizing from 1080 would be short by 8 rows
        // times the pitch — a real overflow, not conservatism.
        frame_bytes = prosper::video::nv12_bytes((uint32_t)config->max_width,
                                                 (uint32_t)config->max_height);
    } else {
        // Auto dimensions: the value below is a FLOOR, not a derivation, and the original
        // resolution-independent hazard survives on this path. Nothing in the code distinguishes the
        // two cases at the call site, so say it once rather than let the next reader — or the real
        // decoder that #1688 adds — treat 64 KiB as a computed answer.
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            fprintf(stderr,
                    "[vdec] QueryDecoderMemoryInfo: max_width/max_height are auto (%d x %d), so "
                    "max_frame_size falls back to the 0x%llx FLOOR -- this is not a derived size, and "
                    "a real decoder must not write a picture into it unchecked (#1688)\n",
                    config->max_width, config->max_height, (unsigned long long)VDEC_MIN_MEMORY);
    }
    frame_bytes = (frame_bytes + VDEC_FRAME_ALIGN - 1) & ~(uint64_t)(VDEC_FRAME_ALIGN - 1);

    info->cpu_size = info->gpu_size = info->shared_size = VDEC_MIN_MEMORY;
    info->max_frame_size = frame_bytes > VDEC_MIN_MEMORY ? frame_bytes : VDEC_MIN_MEMORY;
    info->cpu = info->gpu = info->shared = 0;
    info->frame_alignment = VDEC_FRAME_ALIGN; info->reserved = 0;
    return 0;
}
HLE(s_videodec2_create_decoder) {
    svc_log("sceVideodec2CreateDecoder", a0, a1, a2, a3, a4, a5, 9);
    auto* config = (const VdecConfig*)PW(a0); auto* memory = (const VdecMemory*)PW(a1);
    auto* out = (uint64_t*)PW(a2);
    if (!config || !memory || !out) return VDEC_ERR_ARG;
    if (memory->size != sizeof(*memory)) return VDEC_ERR_STRUCT;
    // Building a decoder DOES need a queue to build it on, and a caller that reaches here without one
    // must fail visibly rather than be handed a decoder handle backed by no queue (#1658).
    uint64_t valid = vdec_validate_config(config, /*require_queue=*/true); if (valid) return valid;
    // The other half of the #2270 contract capture: what the title asked the decoder to BE.
    // codec/profile name the bitstream a real decoder must accept; max_width/height and
    // VdecMemory::max_frame_size together pin the output layout arithmetically -- max_frame_size
    // divided by max_width*max_height is bytes-per-pixel, and 1.5 vs 4 is YUV420 vs packed 32-bit.
    if (getenv("PROSPER_VDEC2_CONTRACT"))
        fprintf(stderr,
                "[vdec-contract] create codec=%u profile=%u max=%dx%d dpb=%d input_depth=%u | "
                "mem cpu=%llu gpu=%llu shared=%llu max_frame=%llu align=%u\n",
                config->codec, config->profile, config->max_width, config->max_height,
                config->max_dpb, config->input_depth,
                (unsigned long long)memory->cpu_size, (unsigned long long)memory->gpu_size,
                (unsigned long long)memory->shared_size,
                (unsigned long long)memory->max_frame_size, memory->frame_alignment);
    if (memory->cpu_size < VDEC_MIN_MEMORY || memory->gpu_size < VDEC_MIN_MEMORY ||
        memory->shared_size < VDEC_MIN_MEMORY || memory->max_frame_size < VDEC_MIN_MEMORY)
        return VDEC_ERR_MEMORY_SIZE;
    if (!memory->cpu || !memory->gpu || !memory->shared) return VDEC_ERR_MEMORY_PTR;
    const uint64_t handle = g_handle.fetch_add(1) + 0x10000;
    { std::lock_guard<std::mutex> lk(g_vdec_mx); g_vdec_codecs[handle] = config->codec; }
    *out = handle;
    if (svclog()) fprintf(stderr, "[svc]   Videodec2 decoder -> 0x%llx codec=%u\n",
                          (unsigned long long)handle, config->codec);
    return 0;
}
HLE(s_videodec2_delete_decoder) {
    // Tear the access-unit decoder down with the guest's decoder (#2270). Without this every
    // deleted decoder strands an AVCodecContext, two AVFrames, an AVPacket and the NV12 staging
    // vector for the life of the process.
    //
    // A LEAK, not a correctness bug, and the distinction is worth keeping straight: a stale
    // g_vdec_au_ids entry could only decode against another movie's reference frames if a handle
    // were ever REUSED, and it is not -- g_handle is fetch_add-only and never reset (:66, :529),
    // so the recycled-handle path is unreachable.
    //
    // close_decoder runs OUTSIDE g_vdec_mx: it calls into the video backend, and holding an HLE
    // lock across a backend call is how lock-order problems get built.
    int au_id = -1;
    {
        std::lock_guard<std::mutex> lk(g_vdec_mx);
        if (!g_vdec_codecs.erase(a0)) return VDEC_ERR_DECODER;
        auto it = g_vdec_au.find(a0);
        if (it != g_vdec_au.end()) { au_id = it->second.id; g_vdec_au.erase(it); }
    }
    if (au_id >= 0)
        if (auto* vb = prosper::video::backend()) vb->close_decoder(au_id);
    return 0;
}
HLE(s_videodec2_decode) {
    // This log is CAPPED at 8 access units, and it is the only Videodec2 entry point that caps.
    // Say so in the log itself: once a title actually feeds this path, "8 decodes in the log" reads
    // exactly like "the guest decoded 8 frames and stopped", which is a different and much more
    // alarming finding than "the cap was reached". (Instrument-trap 13 in
    // docs/GAME_COMPAT_ORCHESTRATION.md — a print cap misread as a frequency. Nearly banked as a
    // result on the very first boot that reached here, #1658.)
    if (svclog()) {
        static std::atomic<unsigned> n{0};
        unsigned seq = n.fetch_add(1);
        if (seq < 8) svc_log("sceVideodec2Decode", a0, a1, a2, a3, a4, a5, 7);
        else if (seq == 8)
            fprintf(stderr, "[vdec] sceVideodec2Decode: log capped at 8 access units; further "
                            "calls still execute but are NOT logged (this cap is not a count)\n");
    }
    uint32_t codec;
    { std::lock_guard<std::mutex> lk(g_vdec_mx); auto it = g_vdec_codecs.find(a0);
      if (it == g_vdec_codecs.end()) return VDEC_ERR_DECODER; codec = it->second; }
    auto* input = (const VdecInput*)PW(a1); auto* frame = (VdecFrame*)PW(a2);
    auto* out = (VdecOutput*)PW(a3);
    if (!input || !frame || !out) return VDEC_ERR_ARG;
    if (input->size != sizeof(*input) || frame->size != sizeof(*frame) ||
        out->size != sizeof(*out)) return VDEC_ERR_STRUCT;
    if (input->data_size && !input->data) return VDEC_ERR_ARG;
    if (!frame->data_size) return VDEC_ERR_FRAME_SIZE;
    if (!frame->data) return VDEC_ERR_FRAME_PTR;
    // PROSPER_VDEC2_CONTRACT: what a title actually asks this decoder for (#2270). prosper has no
    // Videodec2 decoder, and the two facts a fix needs -- the output pixel FORMAT enum and the
    // LAYOUT to write into frame->data -- are the two that cannot be read off a struct definition.
    //
    // There is a catch-22 in the middle of it: what the guest does with a DECODED picture cannot be
    // observed while every decode returns no-picture. What CAN be observed is what the guest asked
    // for and what it allocated, and the allocation constrains the layout arithmetically -- a
    // frame buffer of w*h*3/2 is a YUV420/NV12 request, w*h*4 is a packed 32-bit one. That is a
    // derivation from bytes the guest wrote, not an inference from what seems likely.
    if (getenv("PROSPER_VDEC2_CONTRACT")) {
        static std::atomic<unsigned> seq{0};
        const unsigned k = seq.fetch_add(1);
        if (k < 12) {
            const double px = 0.0;
            (void)px;
            // The first bytes of the access unit NAME the bitstream, which the codec field does not
            // when its value is not one we have seen before. H.264 carries a 1-byte NAL header after
            // the start code (SPS = 0x67); HEVC carries a 2-byte one (VPS = 0x40 0x01). Printing the
            // head is what separates "implement HEVC" from "we are reading the wrong field".
            char head[64] = {0};
            if (input->data && input->data_size) {
                const auto* au = (const uint8_t*)(uintptr_t)input->data;
                const uint64_t n = input->data_size < 16 ? input->data_size : 16;
                for (uint64_t i = 0; i < n; ++i)
                    snprintf(head + i * 3, 4, "%02x ", au[i]);
            }
            fprintf(stderr,
                    "[vdec-contract] decode#%u handle=0x%llx au_bytes=%llu frame_buf=%llu "
                    "frame_ptr=0x%llx head=%s\n",
                    k, (unsigned long long)a0, (unsigned long long)input->data_size,
                    (unsigned long long)frame->data_size,
                    (unsigned long long)frame->data, head);
        }
    }
    // Real decode through the backend's access-unit path (#2270).
    //
    // ON BY DEFAULT. This path landed opt-in behind PROSPER_VDEC2_DECODE while the output FORMAT
    // enum was unestablished, reasoning that a wrong constant hands the guest a correctly-decoded
    // picture it reads wrongly. That weighed one unknown field against the entire image and got the
    // trade backwards: with the path off, out->format is 0 AND out->pictures is 0, forever. The
    // guest receives the SAME unestablished format value it would receive with decoding on, plus no
    // image at all, and a title that waits for a frame waits for one that cannot arrive. Enabling
    // this cannot make `format` more wrong than leaving it off already does.
    //
    // PROSPER_VDEC2_NO_DECODE=1 restores the no-picture behaviour, so the A/B that justified this
    // stays reproducible.
    static const bool vdec2_no_decode = getenv("PROSPER_VDEC2_NO_DECODE") != nullptr;
    auto* vb = vdec2_no_decode ? nullptr : prosper::video::backend();
    if (vb) {
        int au_id;
        bool announce = false;
        {
            // open_decoder runs UNDER g_vdec_mx, unlike close_decoder below, and the asymmetry is
            // deliberate rather than an oversight. "Open at most once per open attempt, and never
            // once per access unit" is what the fail-visible path rests on -- one refusal, and not a
            // fresh AVCodecContext for every unit -- and holding the lock across the call is what
            // makes it one. Doing it outside would need a published "opening" state that a
            // concurrent decode on the same handle would read as a refusal. It is safe because the
            // backend's access-unit registry is a LEAF lock: nothing inside the backend re-enters
            // the HLE, so no path acquires the two in the opposite order.
            //
            // Note the invariant is per OPEN ATTEMPT, not per handle for all time: Reset deliberately
            // re-arms it. The BANNER is per handle -- see VdecAu::announced.
            std::lock_guard<std::mutex> lk(g_vdec_mx);
            VdecAu& st = g_vdec_au[a0];
            if (!st.opened) { st.opened = true; st.id = vb->open_decoder(codec); }
            au_id = st.id;
            if (au_id < 0 && !st.announced) { st.announced = true; announce = true; }
        }
        if (au_id < 0) {
            // FAIL-VISIBLE, and this is the whole point of #2270. Announce the refusal once per
            // guest decoder, with the codec AND the head of the access unit: the codec field alone
            // does not name a bitstream (2382845 is not an ordinal anyone can look up), while the
            // first bytes do -- 00 00 00 01 is Annex-B H.264/HEVC, and a VP9 key frame carries the
            // sync code 49 83 42. Whoever reads this log can tell "implement this codec" apart from
            // "we read the wrong field" without another run.
            if (announce) vdec_no_decoder_warning(codec, input, "the backend refused the codec");
        } else {
            vdec_dump_au((const uint8_t*)PW(input->data), input->data_size);
            // The decoder writes STRAIGHT INTO the guest's frame buffer, under its own lock. It used
            // to hand back pointers into its staging buffer for this function to copy from with no
            // lock held, which raced a concurrent Reset/DeleteDecoder freeing exactly that buffer
            // (#2571 review N5). Passing the destination down removes the lifetime question instead
            // of documenting it, and drops a 3.1 MB per-frame copy on the way.
            prosper::video::VideoBackend::AuPicture pic;
            auto* dst = (uint8_t*)PW(frame->data);
            const auto decoded = vb->decode_au(au_id, (const uint8_t*)PW(input->data),
                                               (size_t)input->data_size, dst, frame->data_size, pic);
            using AuResult = prosper::video::VideoBackend::AuResult;
            // A decoded picture that does not fit is the shape a WRONG LAYOUT takes, so it must not
            // be silent. `pic.nv12_bytes` is the exact NV12 derivation the whole output rests on; if
            // that derivation is wrong, what you observe is exactly this -- a buffer the guest sized
            // correctly and we think is too small. Reporting it as "no picture" without a word makes
            // it indistinguishable from a decoder that merely needs more input, which is the benign
            // case it looks like. Rate-limited, and it names both numbers so the ratio is checkable
            // on sight (#2270).
            if (decoded == AuResult::FrameTooSmall) {
                static std::atomic<unsigned> warned{0};
                if (warned.fetch_add(1) < 8)
                    fprintf(stderr,
                            "[vdec2] frame buffer too small: need=%llu (%ux%u NV12) but "
                            "frame->data_size=%llu -- reporting NO PICTURE. If this fires, "
                            "suspect the bytes-per-pixel derivation before the decoder (#2270)\n",
                            (unsigned long long)pic.nv12_bytes, pic.width, pic.height,
                            (unsigned long long)frame->data_size);
            }
            // THE GUARD. A zero-size picture must never become a reported one: `w*h*3/2` is 0 for a
            // 0x0 frame, so the old size check passed it, copied nothing, and answered valid=1 /
            // pictures=1 -- a confident, well-formed "here is your picture" over nothing, which is
            // precisely the false-success shape this whole change exists to remove, re-created one
            // layer down in its own new code. Only reachable through a backend bug; the backend
            // guards it too, and this one covers EVERY backend rather than the one that has it
            // today. (#2571 review N8.)
            if (decoded == AuResult::Decoded && (!pic.width || !pic.height || !pic.nv12_bytes)) {
                static std::atomic<unsigned> warned{0};
                if (warned.fetch_add(1) < 8)
                    fprintf(stderr,
                            "[vdec2] backend reported a DECODED %ux%u picture (%llu bytes) -- that "
                            "is not a picture, and reporting it as one would be #2270 again. "
                            "Answering NO PICTURE.\n",
                            pic.width, pic.height, (unsigned long long)pic.nv12_bytes);
            } else if (decoded == AuResult::Decoded) {
                // ONE line, once per run, naming the exact NV12 layout prosper wrote and where.
                // A guest reads that buffer through T# descriptors whose pitch, height and second-
                // plane offset it chose itself, and the only way to tell a layout disagreement from
                // a sampling defect is to have both numbers on one screen: this line supplies the
                // producer's half, the renderer's [avpchroma] line the consumer's.
                {
                    static std::atomic<bool> said{false};
                    if (!said.exchange(true))
                        fprintf(stderr,
                                "[vdec2] first picture: %ux%u NV12 y_stride=%u uv_stride=%u "
                                "nv12_bytes=%llu -> guest buffer 0x%llx (%llu bytes); chroma plane "
                                "starts at +%llu\n",
                                pic.width, pic.height, pic.y_stride, pic.uv_stride,
                                (unsigned long long)pic.nv12_bytes,
                                (unsigned long long)frame->data,
                                (unsigned long long)frame->data_size,
                                (unsigned long long)((uint64_t)pic.y_stride * pic.height));
                }
                // Dump what the GUEST receives, not what the backend produced: the copy is part of
                // what a reference comparison has to cover.
                vdec_dump_picture(dst, (size_t)pic.nv12_bytes, pic.width, pic.height);
                { std::lock_guard<std::mutex> lk(g_vdec_mx); g_vdec_au[a0].no_picture_run = 0; }
                frame->accepted = 1;
                out->valid = 1; out->error = 0; out->pictures = 1; out->discarded = 0;
                out->codec = codec;
                out->width = pic.width; out->height = pic.height;
                out->pitch = pic.y_stride; out->pitch_bytes = pic.y_stride;
                out->frame = frame->data; out->frame_size = frame->data_size;
                // #2898: remember this picture's SPS/VUI metadata so GetPictureInfo can
                // answer from the real stream. The access-unit bytes are exactly what the
                // guest handed us this call -- no extra I/O, no guessing.
                {
                    prosper::h264::SpsPictureMeta m;
                    bool have = prosper::h264::parse_first_sps(
                        (const uint8_t*)PW(input->data), (size_t)input->data_size, &m);
                    std::lock_guard<std::mutex> lk(g_vdec_mx);
                    VdecPicMeta& e = g_vdec_picmeta[frame->data];
                    if (!e.record) {
                        e.record = calloc(1, kVdecInfoRecordSize);
                        if (e.record) {
                            // *(void**)record = &record[8]: the payload block. calloc
                            // zeroes it, so the C-string at payload+0x08 is empty and
                            // the flag byte at payload+0x30 is 0 (guest default path).
                            *(void**)e.record = (uint8_t*)e.record + 8;
                        }
                    }
                    e.has_meta = have;
                    if (have) e.meta = m;
                    static std::atomic<bool> said{false};
                    if (!said.exchange(true))
                        fprintf(stderr,
                                "[vdec2] picture meta captured for GetPictureInfo (#2898): "
                                "record=%p sps=%s crop=%u[%u,%u,%u,%u] ar=%u(idc=%u sar=%ux%u) "
                                "timing=%u(%u/%u)\n",
                                e.record,
                                have ? "parsed" : "absent-in-AU", m.crop_flag ? 1u : 0u,
                                m.crop[0], m.crop[1], m.crop[2], m.crop[3],
                                m.ar_flag ? 1u : 0u, m.ar_idc, m.sar_w, m.sar_h,
                                m.timing_flag ? 1u : 0u, m.num_units_in_tick, m.time_scale);
                }
                // The one field still unestablished (#2270). Left 0 rather than filled with a
                // plausible constant, because a wrong value here is a correctly-decoded picture
                // the guest misreads, which is silent. PROSPER_VDEC2_FORMAT exists so the first
                // title observed misreading a picture can be swept over candidate values in one
                // sitting instead of one rebuild per candidate; it is a diagnostic, and a value
                // discovered through it belongs in code with the title evidence that found it.
                static const uint32_t format_override = [] {
                    const char* v = getenv("PROSPER_VDEC2_FORMAT");
                    return v ? (uint32_t)strtoul(v, nullptr, 0) : 0u;
                }();
                out->format = format_override;
                return 0;
            }
            // The decoder opened and is consuming units without ever producing a picture. A few of
            // those are correct (a decoder builds reference state before its first frame); an
            // unbounded run of them is #2270 wearing the benign costume, so say so once, loudly.
            unsigned run;
            { std::lock_guard<std::mutex> lk(g_vdec_mx); run = ++g_vdec_au[a0].no_picture_run; }
            if (run == kVdecNoPictureAlarm)
                fprintf(stderr,
                        "[vdec2] sceVideodec2Decode: %u consecutive access units with NO PICTURE "
                        "from an open codec=%u decoder. Each one reported SCE_OK, which a title "
                        "cannot tell apart from \"no frame ready yet\" -- if it is waiting for a "
                        "frame it will wait forever (#2270)\n",
                        run, codec);
        }
    } else if (!vdec2_no_decode) {
        // No backend at all: headless tools and any build without a host video decoder. Same
        // announcement, same reason -- silence here is the defect.
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            vdec_no_decoder_warning(codec, input, "no host video backend is registered");
    }
    frame->accepted = 0; vdec_no_picture(frame, out, codec);
    return 0;
}
HLE(s_videodec2_flush) {
    uint32_t codec;
    { std::lock_guard<std::mutex> lk(g_vdec_mx); auto it = g_vdec_codecs.find(a0);
      if (it == g_vdec_codecs.end()) return VDEC_ERR_DECODER; codec = it->second; }
    auto* frame = (VdecFrame*)PW(a1); auto* out = (VdecOutput*)PW(a2);
    if (!frame || !out) return VDEC_ERR_ARG;
    if (frame->size != sizeof(*frame) || out->size != sizeof(*out)) return VDEC_ERR_STRUCT;
    // "Nothing buffered" is a legitimate answer to a flush and terminates a guest's drain loop, so
    // this is NOT the #2270 false-success shape and must not be shouted about like one. It is still
    // incomplete: a decoder holding reordered pictures loses them here, which costs a movie its tail
    // frames rather than hanging a title. Said once, only when a real decoder is actually attached,
    // so the log reflects a consequence someone can observe (#2562).
    {
        bool has_decoder = false;
        { std::lock_guard<std::mutex> lk(g_vdec_mx);
          auto it = g_vdec_au.find(a0); has_decoder = it != g_vdec_au.end() && it->second.id >= 0; }
        static std::atomic<bool> said{false};
        if (has_decoder && !said.exchange(true))
            fprintf(stderr, "[vdec2] sceVideodec2Flush reports \"no pictures buffered\" without "
                            "draining the decoder: any reordered pictures still held are lost, so a "
                            "movie may end a few frames early (#2562)\n");
    }
    frame->accepted = 0; vdec_no_picture(frame, out, codec);
    return 0;
}
HLE(s_videodec2_reset) {
    // sceVideodec2Reset DISCARDS the decoder's buffered state and leaves the decoder usable. It is
    // neither of its two neighbours, and #2585 was implementing it as the third:
    //
    //   Flush  -- DRAIN.   Carries a VdecFrame/VdecOutput, so it hands buffered pictures back.
    //   Reset  -- DISCARD. Carries neither, so it returns nothing; it throws the state away.
    //   Delete -- DESTROY. The decoder is gone and the handle with it.
    //
    // THE WARRANT IS THE DOMINANCE ARGUMENT BELOW, not the confidence label. A weaker supporting
    // claim -- that Reset must leave the decoder usable, derivable from the export list because
    // libSceVideodec2 has NO re-initialise entry point between Create and Delete -- is CONFIDENCE:
    // HIGH but is NOT load-bearing: the pre-fix code already satisfied it in the guest-visible sense,
    // since the handle stayed valid and the next Decode reopened. Cite the dominance argument.
    //
    // THIS USED TO CLOSE THE BACKEND DECODER and let the next Decode open a fresh one. That forgets
    // the parsed SPS/PPS along with the DPB, and the two are not equally re-suppliable: Videodec2's
    // caller demuxes itself, so whether parameter sets are repeated in-band is title-dependent, and
    // a title that sends them once and resets mid-stream got a decoder that could not decode until
    // the next in-band SPS -- #2270's own hang shape in a new place. Measured on this build's
    // libavcodec with the committed Annex-B asset: a FLUSHED context fed access units with every
    // SPS/PPS stripped decodes 6 pictures, a FRESH one decodes 0. See VaapiBackend::reset_decoder.
    //
    // Which disposition the Sony contract actually specifies for the sequence headers is NOT
    // established (CONFIDENCE: MED) -- and it does not have to be, because flushing DOMINATES. If
    // the contract keeps them, close-and-reopen is broken and flushing is right. If the contract
    // drops them, flushing is NO WORSE: a stream that repeats its parameter sets replaces the
    // retained ones in-band by id, and a stream that does not repeat them is one where retaining is
    // the only thing that decodes at all. (Stated as "no worse" rather than "correct" on purpose --
    // under that reading, retention can make prosper decode where a faithful implementation would
    // not. It does not change the decision.) There is no reading under which closing wins.
    //
    // LOGGED, and that is not incidental. #2585's own text argued the defect was bounded because
    // "no title in this repository's history is recorded calling sceVideodec2Reset at all" -- but
    // this handler had no svc_log call, so no boot could ever have recorded one. That was a fact
    // about the instrument being used as a fact about the titles. `dump_words = 0`: the only
    // argument is a decoder handle, and a handle is >= 0x10000, which svc_ptrish would otherwise
    // mistake for a pointer and try to dump.
    svc_log("sceVideodec2Reset", a0, a1, a2, a3, a4, a5, 0);
    int au_id = -1;
    {
        std::lock_guard<std::mutex> lk(g_vdec_mx);
        if (!g_vdec_codecs.count(a0)) return VDEC_ERR_DECODER;
        auto it = g_vdec_au.find(a0);
        if (it != g_vdec_au.end()) {
            au_id = it->second.id;
            it->second.no_picture_run = 0;
            // A handle with NO live backend decoder re-arms the open so the next Decode retries.
            // That is the refused-codec path, and `announced` is deliberately kept so the refusal
            // does not re-print its banner on every reset cycle (#2571 N3). A LIVE decoder must NOT
            // re-arm: it is about to be flushed in place and never closes, so re-arming would open a
            // second decoder and strand the first.
            if (au_id < 0) it->second.opened = false;
        }
    }
    if (au_id < 0) return 0;
    // Outside the lock: these call into the backend, and holding an HLE lock across a backend call
    // is how lock-order problems get built (same rule as DeleteDecoder above).
    auto* vb = prosper::video::backend();
    if (vb && vb->reset_decoder(au_id)) return 0;   // flushed in place; the decoder stays open
    // FALLBACK -- a backend with no in-place reset (or none registered any more). Close and re-arm,
    // which is what this function used to do unconditionally. It forgets too much rather than too
    // little, which is the right direction to fail: leaving the DPB live would decode the guest's
    // next access units against references it just asked us to forget, and that is a corrupt picture
    // rather than an error. Said once, because a fail-visible backstop nobody can see is not one.
    if (vb) {
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            fprintf(stderr,
                    "[vdec2] sceVideodec2Reset: this backend has no in-place reset, so the decoder "
                    "is being CLOSED and reopened. That discards the parsed sequence headers as well "
                    "as the DPB, so a title whose stream carries its parameter sets only once cannot "
                    "decode again until its next in-band SPS (#2585)\n");
    }
    {
        std::lock_guard<std::mutex> lk(g_vdec_mx);
        auto it = g_vdec_au.find(a0);
        if (it != g_vdec_au.end() && it->second.id == au_id) {
            it->second.id = -1;
            it->second.opened = false;
        }
    }
    if (vb) vb->close_decoder(au_id);
    return 0;
}
// sceVideodec2GetPictureInfo / ...GetAvcPictureInfo. `which` names the entry point so a size mismatch
// identifies itself (#1658).
//
// #2898, now implemented for the generic form. Guest evidence (PPSA06367 x2, PPSA29343 x3 --
// all five sites disassembled):
//   * the caller pre-zeroes its picture-info block and sets its FIRST field to the block
//     size -- 0x78 (Gollum both sites), 0xb8 and 0x58 (Beast) -- the same self-sizing
//     convention every other struct in this library uses. The size is the variant
//     discriminator, so it is honoured rather than matched against one constant.
//   * on return the guest reads +0x20 (a pointer), +0x35/+0x38..+0x44 (crop flag +
//     quad), +0x48/+0x49/+0x4a/+0x4c (aspect flag/idc/SAR pair), +0x55/+0x58/+0x5c
//     (timing flag/num_units_in_tick/time_scale). That set IS H.264 SPS/VUI, and every
//     value comes from the stream we actually decoded (see h264_sps.cpp).
//   * +0x20 points at a decoder-side record; Gollum reads an inline C-string at +0x08 of
//     it and a flag byte at +0x30, Beast only compares it for identity. Our record is a
//     zeroed block with an empty string: flag 0 takes the guest's own default path.
//
// The AVC form (kjrLbcyhEiw) keeps the old behaviour: no title has ever been observed
// calling it, so its layout remains unestablished, and inventing one would be fabricating
// an ABI. The mismatch path still reports the size the caller declared.
static uint64_t vdec_picture_info(const char* which, uint64_t a0, uint64_t a1, uint64_t a2,
                                  uint64_t a3) {
    // Report the FIRST call to each entry point unconditionally, with the later arguments.
    //
    // Relying on a size mismatch alone would have been a weak plan: both forms take the same
    // OutputInfo in a0, so if what distinguishes them is what they WRITE to a later out-param, a0
    // matches, the mismatch branch never fires, and a run set up to capture the AVC layout captures
    // nothing while appearing to work. Printing a1..a3 once costs two lines a boot and cannot miss.
    static std::mutex mx;
    static std::unordered_map<const char*, int> seen;   // `which` is a literal: stable key
    bool first = false;
    { std::lock_guard<std::mutex> lk(mx); first = (seen[which]++ == 0); }
    if (first)
        fprintf(stderr, "[vdec] %s FIRST CALL: a0=0x%llx a1=0x%llx a2=0x%llx a3=0x%llx\n",
                which, (unsigned long long)a0, (unsigned long long)a1,
                (unsigned long long)a2, (unsigned long long)a3);
    auto* out = (const VdecOutput*)PW(a0);
    if (!out) return VDEC_ERR_ARG;
    if (out->size != sizeof(*out)) {
        static std::mutex bad_mx;
        static std::unordered_map<const char*, int> bad;
        bool report = false;
        { std::lock_guard<std::mutex> lk(bad_mx); report = (bad[which]++ < 4); }
        if (report)
            fprintf(stderr, "[vdec] %s: outputInfo size 0x%llx != VdecOutput 0x%zx -- rejected. If this "
                            "is the AVC form, that size IS its layout evidence (#1658).\n",
                    which, (unsigned long long)out->size, sizeof(*out));
        return VDEC_ERR_STRUCT;
    }

    // The AVC form stops here: no title has ever been observed calling it, its layout is
    // unestablished, and inventing one would be fabricating an ABI. It accepts the same
    // OutputInfo and stays permissive about everything else.
    if (strcmp(which, "sceVideodec2GetPictureInfo") != 0) return 0;

    // Generic form (#2898): fill the caller's picture-info from real SPS/VUI metadata.
    auto* pic = PW(a1);
    if (!pic) return VDEC_ERR_ARG;
    // Self-sizing convention: the caller's first field declares the block's length.
    // Observed variants: 0xb8 and 0x78 and 0x58 (PPSA29343's three sites). The fill is
    // tiered (see h264_sps.hpp), so anything from 0x28 up gets every group that fits;
    // below that the guest could not read even the +0x20 pointer back.
    const uint64_t declared = *(const uint64_t*)pic;
    if (declared < 0x28 || declared > 0x1000) {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 4)
            fprintf(stderr, "[vdec] %s: pictureInfo size 0x%llx cannot hold the observed "
                            "field set (smallest tier 0x28; observed variants "
                            "0x58/0x78/0xb8, #2898) -- rejected\n",
                    which, (unsigned long long)declared);
        return VDEC_ERR_STRUCT;
    }

    prosper::h264::SpsPictureMeta meta;
    void* record = nullptr;
    bool have = false;
    // #2898 diagnosis: the guest FREES the block behind pictureInfo+0x20 through its own
    // allocator (measured: prosper's calloc'd record came back as "FMallocBinned3
    // Attempt to free an unrecognized block", same address printed). So whatever lives
    // there must be memory the GUEST allocated. The strongest candidate is the decoded
    // frame buffer itself -- out->frame -- which the guest owns and Beast's identity
    // arrays could key on. PROSPER_VDEC2_PICINFO_ECHO_FRAME=1 tests that reading.
    static const bool echo_frame = [] {
        const char* v = getenv("PROSPER_VDEC2_PICINFO_ECHO_FRAME");
        if (v && *v == '1') {
            fprintf(stderr, "[vdec] %s: PROSPER_VDEC2_PICINFO_ECHO_FRAME armed -- +0x20 "
                            "will echo outputInfo->frame. DIAGNOSTIC ONLY: measured on "
                            "PPSA06367 this faults at image+0x105ed54 (the guest derefs "
                            "*record as a pointer to the payload), which is the evidence "
                            "that falsified the frame-echo reading (#2967)\n",
                    "sceVideodec2GetPictureInfo");
            return true;
        }
        return false;
    }();
    {
        std::lock_guard<std::mutex> lk(g_vdec_mx);
        auto it = g_vdec_picmeta.find(out->frame);
        if (it != g_vdec_picmeta.end()) {
            have = it->second.has_meta;
            meta = it->second.meta;
            record = echo_frame ? (void*)(uintptr_t)out->frame : it->second.record;
        }
    }
    if (!record) {
        // Never decoded through this outputInfo: there is no picture to describe. An
        // error here is honest, and both observed callers cope -- Gollum ignores the
        // return but only calls after a successful Decode, Beast tests the return and
        // takes its error path (its three sites all do `test eax,eax`).
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 4)
            fprintf(stderr, "[vdec] %s: no decoded picture behind frame=0x%llx yet -- "
                            "returning VDEC_ERR_PIPE (#2898)\n",
                    which, (unsigned long long)out->frame);
        return VDEC_ERR_PIPE;
    }
    if (!have) {
        // A picture exists but its AU carried no parseable SPS (parameter sets can arrive
        // out-of-band or in a later unit). Still publish the record pointer -- identity
        // is what Beast needs and Gollum dereferences -- with all optional flags zero,
        // which is the truthful answer "this stream declares none of those".
        static std::atomic<int> said{0};
        if (said.fetch_add(1) < 2)
            fprintf(stderr, "[vdec] %s: no SPS parsed for frame=0x%llx; filling flags as absent "
                            "(#2898)\n", which, (unsigned long long)out->frame);
    }
    if (!prosper::h264::fill_picture_info(meta, record, pic, (size_t)declared))
        return VDEC_ERR_STRUCT;  // unreachable: the guard above admits only tiers the fill handles
    return 0;
}
HLE(s_videodec2_picture_info) {
    return vdec_picture_info("sceVideodec2GetPictureInfo", a0, a1, a2, a3);
}
HLE(s_videodec2_avc_picture_info) {
    return vdec_picture_info("sceVideodec2GetAvcPictureInfo", a0, a1, a2, a3);
}
// sceUserServiceGetEvent(SceUserServiceEvent* ev): the event stream. A real system delivers the initial
// user's LOGIN event once at startup, then reports "no more events" so the game's drain loop terminates.
// The previous (unimplemented) stub returned 0 = "got an event" but left the struct unfilled -> the game
// either drained a garbage event or never saw the login it waits on. SceUserServiceEvent = { int32
// eventType (0=LOGIN,1=LOGOUT); int32 userId }.
// NO_EVENT = 0x80960007 (Kyty Errno.h USER_SERVICE_ERROR_NO_EVENT). We first shipped 0x80960009 —
// a DIFFERENT UserService error — and the game's main-thread drain loop, not recognizing it as
// "no more events", retried GetEvent forever: the frame loop never built frame 2 (gdb-sampled spin,
// 4/6 PC samples inside this function). One wrong errno constant == a full render stall.
HLE(s_user_getevent)  {
    static std::atomic<int> delivered{0};
    svc_log("sceUserServiceGetEvent", a0,a1,a2,a3,a4,a5);
    if (a0 && delivered.exchange(1) == 0) { int32_t* ev = (int32_t*)PW(a0); ev[0] = 0; ev[1] = 1; return 0; }
    return 0x80960007ull;   // SCE_USER_SERVICE_ERROR_NO_EVENT
}
// sceSystemServiceReceiveEvent(SceSystemServiceEvent* ev): the system event stream (resume, launch-app,
// entitlement-update, share-menu, ...). Unregistered, it fell to the return-0 stub = "an event was
// received" while leaving the 8196-byte out-struct (4-byte eventType + 8192-byte union) uninitialized ->
// the guest dispatched on a garbage eventType (same harmful class as s_user_getevent above; imported by
// both Unity targets). The truthful idle answer is NO_EVENT with nothing written. shadPS4
// systemservice.cpp: NO_EVENT = 0x80A10004, PARAMETER (ev==NULL) = 0x80A10003. When a host activity
// was selected, a real shell instead delivers SCE_SYSTEM_SERVICE_EVENT_GAME_INTENT once; the title
// then obtains the opaque launchActivity payload through libSceNpGameIntent.
HLE(s_sysservice_receiveevent) {
    svc_log("sceSystemServiceReceiveEvent", a0,a1,a2,a3,a4,a5);
    if (!a0) return 0x80A10003ull;
    if (g_gameintent_initialized.load(std::memory_order_acquire) && gameintent_activity_id()) {
        bool expected = false;
        if (g_gameintent_event_delivered.compare_exchange_strong(expected, true)) {
            const uint32_t event_type = SYSTEM_SERVICE_EVENT_GAME_INTENT;
            if (svc_write_bytes(a0, &event_type, sizeof(event_type))) return 0;
            g_gameintent_event_delivered.store(false);
            return 0x80A10003ull;
        }
    }
    return 0x80A10004ull;
}
HLE(s_ok)             { return 0; }

// libScePsml: NOT IMPLEMENTED, and it must SAY so rather than answering SCE_OK.
//
// The dispatcher's default for an unregistered NID is `return 0`, which the guest reads as SCE_OK.
// For a query with OUT-PARAMETERS that is not a benign no-op, it is a success claim about data
// prosper never wrote -- so the guest goes on to read whatever the out-parameter's stack slot
// happened to hold. `CLAUDE.md` records the same shape on Metaphor: ReFantazio, where an
// unregistered `sceFontRenderCharGlyphImage` answered SCE_OK, wrote neither out-parameter, and the
// title divided by the resulting zero (#2951).
//
// The First Berserker: Khazan (PPSA20447) is the second title to pay for it, and it multiplies
// instead of dividing. Its `eboot+0xe74b20` does:
//
//     e74b51  mov $0x137,%edi ; call <sceSysmoduleLoadModule>
//     e74b74  call <libScePsml::3WVD91e12ZQ>          ; -> r15d
//     e74b8e  lea -0x48(%rbp),%rdi                    ; an OUT struct, NOT initialised
//     e74b9e  call <libScePsml::+2KpvixvL6E>          ; (&out, buf) -> eax
//     e74ba3  or %r15d,%eax ; jne 0xe74dc2            ; either non-zero -> clean early return
//     e74bac  mov -0x38(%rbp),%rdi                    ; N := out[+0x10]  (uninitialised stack)
//     e74bb5  shl $0x4,%rdi                           ; size = N * 16
//     e74bb9  call <FMemory::Malloc>                  ; ~244 TiB, and the title asserts
//
// Two things make returning an error the right answer here rather than a guess, and both were
// checked rather than assumed -- this project has a recorded trap in the OPPOSITE direction, where
// an error sentinel returned from a VALUE-returning contract was read as data and produced a 2 GiB
// allocation:
//   * The return value is a STATUS, not data: the guest ORs the two results and branches on
//     non-zero. `CONFIDENCE: HIGH` -- disassembled.
//   * The non-zero branch at `eboot+0xe74dc2` is a stack-check plus epilogue plus `ret`. The guest's
//     own code treats this as "the feature is unavailable" and skips it. Nothing is faked and no
//     output is manufactured; prosper is answering truthfully that it does not implement this
//     library, through the only channel the guest reads.
//
// `CONFIDENCE: LOW` on the specific value. libScePsml is absent from the PS5 3.20 reference set
// (this dump requires system software 12.70), so its own error space is not derivable and neither
// are these NIDs' names or signatures. The guest tests only zero versus non-zero, so ENOSYS is a
// truthful placeholder for "not implemented" and not a claim about Sony's encoding. If the real
// error space is ever recovered, replace the value -- not the policy.
HLE(s_psml_unimplemented) { return prosper::hle::kSceKernelErrorENOSYS; }
// ===== libSceAvPlayer (#324/#705): real playback lifecycle over a host video-decode backend =====
// The core owns the guest sceAvPlayer contract + per-player state + the guest event callback and
// pulls decoded frames from a registered VideoBackend (app-side hardware decode). With NO backend
// (headless/tests) it runs a synthetic black-frame lifecycle so the title's video state machine still
// completes and the boot advances. ABI structs mirror the published sceAvPlayer layout (guest is
// SysV LP64 == host on Linux, so guest pointers are read directly).
namespace {
// Guest event callback: void(void* object_ptr, uint32_t event, int32_t source_id, void* data).
using AvpEventCb = void (PROSPER_SYSV_ABI *)(void*, uint32_t, int32_t, void*);
// Guest memory callbacks use the PS5/SysV ABI even when the emulator host is Windows.  Keep their
// types explicit here; calling them as native Windows functions corrupts the argument registers.
using AvpAllocateCb = void* (PROSPER_SYSV_ABI *)(void*, uint32_t, uint32_t);
using AvpDeallocateCb = void (PROSPER_SYSV_ABI *)(void*, void*);
enum : uint32_t { AVP_STOP = 0x01, AVP_READY = 0x02, AVP_PLAY = 0x03, AVP_PAUSE = 0x04, AVP_BUFFERING = 0x05 };
// The WARNING channel of the same callback. Unlike the state events above it carries a payload: the
// fourth callback argument points at a uint32_t notification code, and a guest that is waiting for a
// particular one compares against THAT, not against the event id.
//
// AVP_WARNING_SEEK_COMPLETE is the code sceAvPlayerJumpToTime must publish once the source has
// actually been repositioned. #1974 implemented the seek itself and left this half out, so every
// guest that waits for it burns its own timeout on a seek that already succeeded.
//
// EVIDENCE, in the project's own order of preference:
//   (1) The guest's code. Space Adventure Cobra (PPSA17337) writes 0x806a00a3 into its
//       wait-slot at eboot+0x15400b8, immediately BEFORE calling sceAvPlayerJumpToTime at
//       eboot+0x15400c2, and afterwards blocks in its 15,000 ms event wait at eboot+0x153ce5d
//       (`esi=0x3a98`). Its event callback (eboot+0x153ef30) posts that wait only when the event id
//       is 0x20 AND the payload word equals the awaited code -- for a warning it compares the
//       PAYLOAD, having replaced the id with `*(uint32_t*)eventData` at eboot+0x153ef5c. The
//       matching failure code the same callback special-cases is 0x806a0004.
//   (2) Directly measured: with nothing published, that wait times out at exactly its 15,000 ms
//       budget (15,003 ms between the post-seek frame and sceAvPlayerResume, PROSPER_AVPLOG).
//   (3) Cross-title: of the 42 dumps in the local corpus that import sceAvPlayerJumpToTime
//       (`XC9wM+xULz8`), 40 contain 0x806a00a3 as an immediate and 33 also contain 0x806a0004 --
//       against roughly 0.008 expected occurrences per 32 MB eboot by chance. It is an SDK
//       constant shared across engines, not one title's private token.
// CONFIDENCE: HIGH on the behaviour (read off the guest and measured in both directions).
// CONFIDENCE: MED on the name -- Sony's own symbol for 0x806a00a3 is not in any source available
// here, so the name describes the contract rather than quoting a header.
enum : uint32_t { AVP_WARNING_ID = 0x20 };
enum : uint32_t { AVP_WARNING_SEEK_COMPLETE = 0x806a00a3u };

struct AvpMemAllocator {
    void* obj;
    AvpAllocateCb allocate;
    AvpDeallocateCb deallocate;
    AvpAllocateCb allocate_texture;
    AvpDeallocateCb deallocate_texture;
};
struct AvpFileReplace   { void* obj; void* open; void* close; void* read_offset; void* size; };
struct AvpEventReplace  { void* obj; void* event_callback; };
struct AvpInitData {                 // sceAvPlayerInit
    AvpMemAllocator memory; AvpFileReplace file; AvpEventReplace event;
    uint32_t debug_level; uint32_t base_priority; int32_t num_fb; uint8_t auto_start; uint8_t rsv[3]; const char* default_language;
};
struct AvpThreadInfo {
    uint32_t priority, stack_size; uint64_t affinity; uint8_t reserved[32];
};
struct AvpInitDataEx {               // sceAvPlayerInitEx (note: this_size first)
    uint64_t this_size; AvpMemAllocator memory; AvpFileReplace file; AvpEventReplace event;
    const char* default_language; uint32_t debug_level; uint8_t auto_start; uint8_t rsv[3];
    AvpThreadInfo audio_decoder, video_decoder, demuxer, event_thread, call_queue;
    AvpThreadInfo http_command_processor, http_segment_manager, http_streamlist, file_streaming;
    int32_t num_fb; uint8_t rsv2[4];
};
static_assert(sizeof(AvpThreadInfo) == 48 && sizeof(AvpInitDataEx) == 560 &&
              offsetof(AvpInitDataEx, auto_start) == 116 &&
              offsetof(AvpInitDataEx, num_fb) == 552, "AvPlayerInitDataEx ABI");
static_assert(sizeof(AvpMemAllocator) == 40 && offsetof(AvpInitData, num_fb) == 104 &&
              offsetof(AvpInitData, default_language) == 112 && sizeof(AvpInitData) == 120,
              "AvPlayerInitData ABI");
struct AvpUri           { const char* name; uint32_t length; };
struct AvpSourceDetails { AvpUri uri; uint8_t rsv1[64]; uint32_t source_type; uint8_t rsv2[44]; };
struct AvpFrameInfo {                // sceAvPlayerGetVideoData / GetAudioData out-param
    uint8_t* p_data; uint8_t reserved[4]; uint64_t timestamp;
    uint32_t d0, d1, d2, d3;         // AvPlayerStreamDetails union (16B): video={width,height,aspect,lang}
};
struct AvpVideoEx {                  // AvPlayerVideoEx (in the 80B details union)
    uint32_t width, height; float aspect; uint8_t lang[4]; uint8_t reserved0[4];
    uint32_t crop_left_offset, crop_right_offset, crop_top_offset, crop_bottom_offset, pitch;
    uint8_t luma_bd, chroma_bd, full_range, reserved1[5];
    double framerate; uint32_t colour_primaries, transfer_characteristics; uint8_t reserved2[16];
};
static_assert(sizeof(AvpVideoEx) == 80 &&
              offsetof(AvpVideoEx, crop_left_offset) == 20 &&
              offsetof(AvpVideoEx, crop_right_offset) == 24 &&
              offsetof(AvpVideoEx, crop_top_offset) == 28 &&
              offsetof(AvpVideoEx, crop_bottom_offset) == 32 &&
              offsetof(AvpVideoEx, pitch) == 36 &&
              offsetof(AvpVideoEx, framerate) == 48, "AvPlayerVideoEx ABI");
struct AvpFrameInfoEx {              // sceAvPlayerGetVideoDataEx out-param (larger details union)
    void* p_data; uint8_t reserved[4]; uint64_t timestamp;
    union { AvpVideoEx video; uint8_t raw[80]; } details;
};
struct AvpVideo { uint32_t width, height; float aspect; uint8_t lang[4]; };
struct AvpAudio { uint16_t channels; uint8_t reserved[2]; uint32_t sample_rate, size; uint8_t lang[4]; };
struct AvpAudioEx {
    uint16_t channels; uint8_t reserved[2]; uint32_t sample_rate, size; uint8_t lang[4];
    uint8_t reserved1[64];
};
static_assert(sizeof(AvpAudio) == 16 && sizeof(AvpAudioEx) == 80, "AvPlayerAudio ABI");
struct AvpStreamInfo {
    uint32_t type; uint8_t reserved[4];
    union { AvpVideo video; AvpAudio audio; uint8_t raw[16]; } details; uint64_t duration;
};
struct AvpStreamInfoEx {
    uint64_t this_size; uint32_t type; uint8_t reserved[4];
    union { AvpVideoEx video; AvpAudioEx audio; uint8_t raw[80]; } details; uint64_t duration;
};
static_assert(sizeof(AvpStreamInfo) == 32 && offsetof(AvpStreamInfo, duration) == 24,
              "AvPlayerStreamInfo ABI");
static_assert(sizeof(AvpStreamInfoEx) == 104 && offsetof(AvpStreamInfoEx, duration) == 96,
              "AvPlayerStreamInfoEx ABI");

struct AvpPlayer {
    void* ev_obj = nullptr; AvpEventCb ev_cb = nullptr;
    AvpMemAllocator memory{};
    // The guest's file-replacement table, retained verbatim from sceAvPlayerInit(Ex). When all four
    // entries are present, sceAvPlayerAddSource(Ex) reads the media THROUGH it rather than opening
    // the URI as a host path (#1955) — a title that stores a clip inside a container file has no
    // other way to express the byte range. Reported under PROSPER_AVPLOG; every path that declines
    // to use it reports unconditionally. See avp_add_source.
    AvpFileReplace file{};
    int32_t num_fb = 0;
    bool auto_start = false, have_source = false, synthetic = false;
    bool playing = false, paused = false, stop_fired = false;
    // Set by a successful sceAvPlayerJumpToTime, cleared by the first frame delivered after it.
    // A seek repositions the player and must publish the new position even while the guest holds
    // the player PAUSED — see the delivery gate in s_avp_getvideodata[ex] for the guest evidence.
    bool seek_deliver = false;
    std::string guest_path;
    prosper::video::VideoBackend* backend = nullptr;
    int backend_id = -1;             // >=0 when a host backend is decoding
    uint32_t width = 1920, height = 1080;
    float fps = 30.0f;
    bool has_audio = true;
    uint32_t audio_channels = 2, audio_rate = 48000;
    uint64_t duration_ms = 0;
    uint64_t poll = 0;               // synthetic frames delivered through GetVideoData[Ex]
    uint64_t audio_poll = 0;
    uint64_t active_poll = 0;        // independent progress for IsActive-only playback loops
    uint64_t last_ts_ms = 0;         // newest delivered frame timestamp (sceAvPlayerCurrentTime)
    // When something last ASKED this player for a video frame — the media clock's only input
    // (#1973, see avp_media_played_out). Refreshed by every GetVideoData[Ex] call whatever that call
    // returns, and at every point presentation (re)starts: AddSource, Start, Resume, JumpToTime.
    // Those are also every place `paused` is cleared, which is what keeps paused time — during which
    // nothing is presented and nothing should be pulled — from counting as an absent consumer.
    std::chrono::steady_clock::time_point video_request_at = std::chrono::steady_clock::now();
    // The media-clock anchor for BLOCKING video delivery: the frame with pts P is due at
    // play_start_wall + (P - play_start_pts). The caller is held until due — the real
    // AvPlayer's contract, and the only pacing this guest respects: it sleeps its whole
    // poll budget when a non-blocking gate refuses a frame, which collapsed playback to
    // ~1 frame per sleep (measured 0.4-1.2 fps, #2981 FMV).
    std::chrono::steady_clock::time_point play_start_wall = std::chrono::steady_clock::now();
    uint64_t play_start_pts = 0;
    bool gate_started = false;
    // When the title supplies its AvPlayer texture allocator, decoded NV12 must be written into those
    // guest-visible buffers.  A host vector is invisible to the title's pre-wired texture descriptors.
    std::vector<uint8_t*> texture_frames;
    size_t texture_frame_bytes = 0;
    size_t next_texture_frame = 0;
    std::vector<uint8_t> frame;      // fallback for tests/titles with no texture callbacks
    std::vector<int16_t> audio;      // synthetic silent stereo PCM
};
std::mutex g_avp_mx;
std::unordered_map<uint64_t, AvpPlayer> g_avp;

// PS5 AvPlayer exposes decoded NV12 as a sampled-linear surface. Its physical row pitch is aligned
// to 256 bytes; the valid image width remains separate and the padded pixels are reported through
// AvPlayerVideoEx::crop_right_offset. Keeping those two extents distinct is required by consumers
// that validate an AGC texture footprint before binding its second plane.
bool avp_video_pitch(uint32_t width, uint32_t& pitch) {
    if (!width || width > UINT32_MAX - 255u) return false;
    pitch = (width + 255u) & ~255u;
    return true;
}

bool avp_nv12_bytes(uint32_t pitch, uint32_t height, size_t& bytes) {
    if (!pitch || !height || pitch > SIZE_MAX / height) return false;
    const size_t y_bytes = static_cast<size_t>(pitch) * height;
    const size_t uv_rows = (static_cast<size_t>(height) + 1) / 2;
    if (pitch > SIZE_MAX / uv_rows) return false;
    const size_t uv_bytes = static_cast<size_t>(pitch) * uv_rows;
    if (uv_bytes > SIZE_MAX - y_bytes) return false;
    bytes = y_bytes + uv_bytes;
    return true;
}

void avp_register_frame_layout(uint8_t* base, size_t bytes, uint32_t pitch) {
    gpu::register_guest_linear_texture_layout(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base)), bytes, pitch);
}

// AvPlayerVideoEx publishes `width` as the extent the crop offsets TRIM, not as the visible picture.
//
// The load-bearing evidence is that three shipping titles run on retail PS5 hardware while spelling
// "how wide is the visible picture" two different ways, so Sony's own publication must satisfy both
// at once. Two of the three fields are already pinned by prior live evidence:
//
//   pitch = 2048   forced by R-Type Delta, which builds its luma T# with `pitch` AS the descriptor
//                  width and requires agc_footprint == pitch*height (#1814, R_TYPE_DELTA_STATUS.md).
//   crop_right     forced by GRIS: #1393 padded the pitch with a zero crop and exposed the padded
//     = 128        columns as a right-edge strip, so the crop must describe that padding.
//
// With those pinned, ArcRunner's spelling leaves exactly one free value:
//
//   GRIS       visible = pitch - crop_left - crop_right  ->  2048 - 0 - 128 = 1920   (correct)
//   ArcRunner  visible = width - crop_left - crop_right  ->  1920 - 0 - 128 = 1792   (WRONG)
//
// so `width` must be 2048. Measured on ArcRunner (PPSA21406, #2011): the guest sizes its movie luma
// T# from the published pitch (2048x1080, chroma 1024x540 — half of it) and, while width published
// the visible 1920, sized every converted movie surface 128 columns short at 1792x1080. Publishing
// width == pitch satisfies both spellings at once and leaves GRIS and R-Type unchanged.
//
// The four crop field names mirror H.264's frame_crop_{left,right,top,bottom}_offset, and that is an
// ANALOGY, not the evidence: H.264's coded extent is macroblock-aligned (a 1920x1080 stream codes
// 1920x1088 with crop_bottom, crop_right=0), never memory-pitch-aligned, so the SPS convention alone
// would argue for width=1920/crop_right=0. The real statement is that the crop fields are the only
// ABI channel able to describe buffer padding, so that is what this contract uses them for.
//
// `aspect` stays the VISIBLE display ratio: it is not part of the crop arithmetic, and a consumer
// reads it to letterbox, never to size a surface.
//
// `width == pitch` is only meaningful because every path here is 8-bit (luma_bd/chroma_bd below):
// `pitch` is a BYTE stride and `width` a PIXEL count, and they coincide only at 1 byte per luma
// sample. A 10-bit/P010 source would need the pixel pitch, not the byte pitch, computed here.
//
// CONFIDENCE: MED — derived from three independent shipping consumers that all work on retail
// hardware. No Sony header was consulted. The counter-model this cannot exclude is that
// AvPlayerVideoEx merely EXTENDS AvPlayerVideo (whose `width` is unambiguously visible) and Sony's
// crop fields describe bitstream cropping rather than buffer padding.
bool avp_fill_video_ex(AvpVideoEx& out, uint32_t visible_w, uint32_t visible_h, uint32_t pitch,
                       double framerate) {
    out = {};
    // Every current caller derives `pitch` from the same width it passes, so this cannot trip today.
    // It is a guard rather than an assumption because the underflow would be silent and guest-visible:
    // crop_right_offset is unsigned, so pitch < visible_w publishes ~4 G and every consumer that
    // subtracts it gets a nonsense extent.
    //
    // FAIL-VISIBLE rather than clamping. Clamping to `visible_w` looks like the safe repair and is the
    // more dangerous of the two: the buffer was staged at the REAL pitch, so publishing a larger row
    // stride with crop_right = 0 tells a guest that trusts it to read past the end of every row, and
    // the extents it gets are plausible enough that nobody notices. The ~4 G crop is at least absurd
    // on sight. Since the branch is unreachable from any current caller, refusing costs nothing and
    // keeps a future caller's mistake loud instead of silently corrupting its sampling.
    if (pitch < visible_w) {
        fprintf(stderr,
                "[avp] REFUSING AvPlayerVideoEx: pitch %u < visible width %u -- the staged surface and "
                "the published extent disagree; frame not published\n",
                pitch, visible_w);
        return false;
    }
    out.width = pitch;                 // the extent the crop offsets trim
    out.height = visible_h;            // prosper stages no padded rows
    out.aspect = visible_h ? static_cast<float>(visible_w) / static_cast<float>(visible_h) : 0.0f;
    out.crop_right_offset = pitch - visible_w;
    out.pitch = pitch;
    out.luma_bd = 8;
    out.chroma_bd = 8;
    out.framerate = framerate;
    return true;
}

// The byte values the pitch padding is filled with, and why they are not 0.
//
// `Y=0, U=V=0` is NOT black in either BT.601 or BT.709, limited or full range: it converts to roughly
// (0, 136, 0) — mid green. Since #2011 publishes `width` as the CODED extent, a consumer that reads
// `width` and ignores the crop offsets samples the padded columns, and zero-filled padding would draw
// a bright green stripe down the right edge of every movie frame. Limited-range black is `Y=0x10`
// with `U=V=0x80`, which is what the decoders themselves emit for a black frame (measured on
// ArcRunner's intro movie with PROSPER_DUMP_RAWTEX), so padding filled this way is indistinguishable
// from the picture's own black and costs exactly the same memset.
constexpr uint8_t AVP_PAD_LUMA = 0x10;
constexpr uint8_t AVP_PAD_CHROMA = 0x80;

void avp_release_frame_storage(AvpPlayer& player) {
    if (!player.frame.empty())
        gpu::unregister_guest_linear_texture_layout(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(player.frame.data())));
    player.frame.clear();
}

uint8_t* avp_frame_storage(AvpPlayer& player, size_t bytes, uint32_t pitch) {
    if (player.frame.size() != bytes) {
        avp_release_frame_storage(player);
        player.frame.resize(bytes);
    }
    if (player.frame.empty()) return nullptr;
    avp_register_frame_layout(player.frame.data(), player.frame.size(), pitch);
    return player.frame.data();
}

// Native decoders own their dequeue storage and may move or recycle it on the next pull. Copy the
// visible NV12 rows into a guest texture buffer when one exists.  The host-vector fallback preserves
// the old lifetime guarantee for native tests and titles that omit the replacement callbacks.
bool avp_stage_video(AvpPlayer& player, const prosper::video::VideoFrame& frame,
                     bool extended_layout, uint8_t*& data, uint32_t& pitch) {
    if (!frame.y || !frame.uv || frame.width == 0 || frame.height == 0) return false;
    const size_t y_stride = frame.y_stride ? frame.y_stride : frame.width;
    const size_t uv_stride = frame.uv_stride ? frame.uv_stride : frame.width;
    if (y_stride < frame.width || uv_stride < frame.width) return false;
    const size_t uv_rows = (static_cast<size_t>(frame.height) + 1) / 2;
    if (y_stride > SIZE_MAX / frame.height || uv_stride > SIZE_MAX / uv_rows) return false;
    pitch = frame.width;
    if (extended_layout && !avp_video_pitch(frame.width, pitch)) return false;
    size_t bytes = 0;
    if (!avp_nv12_bytes(pitch, frame.height, bytes)) return false;
    uint8_t* dst = nullptr;
    if (!player.texture_frames.empty()) {
        if (bytes > player.texture_frame_bytes) return false;
        dst = player.texture_frames[player.next_texture_frame++ % player.texture_frames.size()];
    } else {
        dst = avp_frame_storage(player, bytes, pitch);
    }
    if (!dst) return false;
    // Copy visible pixels and fill only the padded tail. Filling the entire frame before copying
    // it would double memory traffic on every decoded frame; the crop still must not expose stale
    // decoder bytes if a title deliberately samples beyond the valid image extent. The fill is
    // limited-range BLACK, not zero — see AVP_PAD_LUMA/AVP_PAD_CHROMA above.
    const size_t padding = static_cast<size_t>(pitch) - frame.width;
    for (uint32_t row = 0; row < frame.height; ++row) {
        memcpy(dst + static_cast<size_t>(row) * pitch,
               frame.y + static_cast<size_t>(row) * y_stride, frame.width);
        if (padding)
            memset(dst + static_cast<size_t>(row) * pitch + frame.width, AVP_PAD_LUMA, padding);
    }
    uint8_t* dst_uv = dst + static_cast<size_t>(pitch) * frame.height;
    for (size_t row = 0; row < uv_rows; ++row) {
        memcpy(dst_uv + row * pitch, frame.uv + row * uv_stride, frame.width);
        if (padding) memset(dst_uv + row * pitch + frame.width, AVP_PAD_CHROMA, padding);
    }
    // Callback storage is registered before the first pull. Refresh it here because the basic API
    // retains its historical tight layout while GetVideoDataEx publishes the padded physical pitch.
    avp_register_frame_layout(dst, bytes, pitch);
    data = dst;
    return true;
}

bool avp_stage_synthetic(AvpPlayer& player, bool extended_layout,
                         uint8_t*& data, uint32_t& pitch) {
    pitch = player.width;
    if (extended_layout && !avp_video_pitch(player.width, pitch)) return false;
    size_t need = 0;
    if (!avp_nv12_bytes(pitch, player.height, need)) return false;
    if (!player.texture_frames.empty()) {
        if (need > player.texture_frame_bytes) return false;
        data = player.texture_frames[player.next_texture_frame++ % player.texture_frames.size()];
        if (!data) return false;
    } else {
        data = avp_frame_storage(player, need, pitch);
        if (!data) return false;
    }
    // Same reason as the padding fill: a zero-filled NV12 frame is mid green, not black. The
    // synthetic path fabricates a blank frame, so it should fabricate a legitimately black one.
    const size_t synth_y_bytes = static_cast<size_t>(pitch) * player.height;
    memset(data, AVP_PAD_LUMA, synth_y_bytes);
    memset(data + synth_y_bytes, AVP_PAD_CHROMA, need - synth_y_bytes);
    avp_register_frame_layout(data, need, pitch);
    return true;
}

bool avp_synth_enabled() { return getenv("PROSPER_AVP_SYNTH_FRAMES") != nullptr; }
uint64_t avp_synth_frames() {
    const char* value = getenv("PROSPER_AVP_SYNTH_FRAMES");
    return value ? strtoull(value, nullptr, 0) : 0;
}
uint64_t avp_synth_duration_ms() {
    const char* value = getenv("PROSPER_AVP_DURATION_MS");
    return value ? strtoull(value, nullptr, 0) : avp_synth_frames() * 33;
}
bool avp_log() { static const bool enabled = getenv("PROSPER_AVPLOG") != nullptr; return enabled; }
// One shared monotonic timeline for every [avp] log line — per-site static t0s made cross-
// referencing deliveries against each other guesswork (#2981 FMV measurement).
uint64_t avp_ms() {
    static const auto t0 = std::chrono::steady_clock::now();
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
}
// PROSPER_AVP_PAUSED_DELIVER — a PRESERVED FALSIFICATION LEVER, NOT A FEATURE, AND NOT A FIX.
// It MUST stay off by default; do not "enable" it and do not promote it into the paused contract.
//
// The shipped contract is the one below: while the guest holds a player paused, GetVideoData/Ex
// deliver nothing. Turning this lever on makes them keep handing out already-decoded frames
// instead. It exists only so the A/B that KILLED a hypothesis stays reproducible, exactly as
// PROSPER_UD_TAIL_ALIGN does for the user-data-tail hypothesis (see CLAUDE.md).
//
// The dead hypothesis (#1599, Asterix & Obelix: Babylon Mission PPSA30490): "the title is stuck in
// Unity's video splash because pause gates frame delivery." Measured, the lever is real but is not
// the cause — it collapses the futile pull census from 14,546 calls to 14 and delivers two further
// frames after the guest's seek, yet sceAvPlayerResume is still never called, all AvPlayer traffic
// then stops, and the output stays uniformly black (max_rgb=0). The actual blocker was that
// sceAvPlayerJumpToTime (NID XC9wM+xULz8) was unimplemented, so the dispatcher's default 0 was read
// as a successful seek: #1949, and prosper/docs/ASTERIX_BABYLON_STATUS.md § Ruled out.
//
// #1949 has since landed and the title now reaches its title screen. Note carefully what that did
// and did not vindicate: a *seek* legitimately publishes its landing frame while paused, and
// s_avp_getvideodata[ex] opens a one-frame window for exactly that (AvpPlayer::seek_deliver). This
// blanket lever is still not that, is still not a fix, and must stay off.
bool avp_paused_deliver() {
    static const bool enabled = getenv("PROSPER_AVP_PAUSED_DELIVER") != nullptr;  // default: false
    return enabled;
}

// ---- the media clock (#1973) ------------------------------------------------------------------
//
// prosper's host decode backends are CONSUMER-DRIVEN: the decode worker fills a small bounded frame
// queue and blocks while it is full, so the decoder only advances when someone pulls. That makes
// "the decoder finished" — until now the player's ONLY end-of-playback signal — unreachable for a
// guest that starts a movie and then never collects a frame. The queue nobody drains never empties,
// VideoBackend::eof() never turns true, and sceAvPlayerIsActive answers 1 forever.
//
// PPSA27624 (Bendy and the Dark Revival) does exactly that: measured on ff72e77c, 14,843
// sceAvPlayerIsActive polls from one call site, ZERO sceAvPlayerGetVideoData[Ex] calls, and no STOP
// event, for a 15.1 s source. The title waits on a movie that cannot end.
//
// A real player is CLOCKED, not consumer-driven. The console's playback timeline runs whether or not
// the application collects frames, so a movie nobody watches still finishes when its media does.
// Model exactly that and nothing more: when NOTHING has asked this player for a video frame for as
// long as the remainder of the media would have taken to play, the media has played out. That is a
// statement about the source's own timeline, not a guess — which is why it may only be made when
// there is a real host decode session with a duration the container actually reported. A synthetic
// session has its own frame budget, and an unknown duration is precisely the case where prosper must
// NOT claim to know when the movie ends.
//
// The "nothing has asked" input is what keeps this off every title that works today. A consumer
// resets the window on EVERY GetVideoData[Ex] call, including calls that return no frame, so a
// starved consumer, or one running far below real time because prosper renders slowly, still ends
// its movie the way it does now: by the queue draining after the decoder finished. This clock can
// only fire where no consumer exists at all — the one case the drain path cannot resolve.
//
// REJECTED ALTERNATIVE, so nobody re-derives it: making video enqueue non-blocking the way audio
// already is (drop the oldest frame when the queue is full) — #1973's own suggested fix. It does let
// decode reach EOF, but it also removes the ONLY thing pacing a movie for every title that works
// today. The decoder would run the whole clip at CPU speed, dropping all but the last few frames, so
// a consuming title's 15-second cutscene becomes six frames and an immediate STOP. The queue's
// backpressure is correct for a decoder; what was missing was a player that knows when its media is
// over. Nothing in the decode backends is touched by this fix, and the same reasoning rules out
// running decode ahead to EOF "bounded by the queue cap" — the bound IS the frame dropping.
//
// CONFIDENCE: HIGH that an unconsumed source must reach the end of its own media. CONFIDENCE: MED on
// the guard band below, which is slop for scheduling rather than a modelled console property.
constexpr uint64_t kAvpMediaClockGraceMs = 250;

bool avp_media_played_out(const AvpPlayer& p, std::chrono::steady_clock::time_point now,
                          uint64_t& idle_ms, uint64_t& remaining_ms) {
    if (!p.backend || p.backend_id < 0 || p.duration_ms == 0) return false;
    remaining_ms = p.duration_ms > p.last_ts_ms ? p.duration_ms - p.last_ts_ms : 0;
    const auto idle =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - p.video_request_at).count();
    idle_ms = idle > 0 ? static_cast<uint64_t>(idle) : 0;
    return idle_ms >= remaining_ms + kAvpMediaClockGraceMs;
}

#if defined(__linux__)
// Linux import stubs swap the guest %fs to the host TCB before entering an HLE handler.  AvPlayer
// event callbacks go the other way and are guest code, so they must run with the caller's guest TCB.
// The entry shims below recover that TCB from the import-stub frame and publish it only for the
// dynamic extent of the HLE call.  Nested AvPlayer calls from the callback preserve the outer value.
thread_local uint64_t t_avp_callback_guest_fs = 0;
struct AvpEntryGuestFsScope {
    uint64_t previous;
    explicit AvpEntryGuestFsScope(uint64_t guest_fs)
        : previous(t_avp_callback_guest_fs) { t_avp_callback_guest_fs = guest_fs; }
    ~AvpEntryGuestFsScope() { t_avp_callback_guest_fs = previous; }
};
struct AvpCallbackGuestFsScope {
    uint64_t saved = 0;
    bool active = false;
    explicit AvpCallbackGuestFsScope(uint64_t guest_fs) {
        if (guest_fs) {
            __asm__ volatile("rdfsbase %0" : "=r"(saved));
            __asm__ volatile("wrfsbase %0" : : "r"(guest_fs));
            active = true;
        }
    }
    ~AvpCallbackGuestFsScope() {
        if (active) __asm__ volatile("wrfsbase %0" : : "r"(saved));
    }
};
#endif

// ---- guest file-replacement reader (#1955) ---------------------------------------------------
// SceAvPlayerFileReplacement is {objectPointer, open, close, readOffset, size} — the order this file
// already declares. Signatures follow the published sceAvPlayer contract: open(obj, path) and
// readOffset(obj, buffer, position, length) return a signed count/status, size(obj) returns the
// media length. CONFIDENCE: MED-HIGH — struct order is confirmed by the ABI static_asserts above and
// by PPSA27624 supplying all four entries, but no live title has yet exercised the call semantics.
using AvpOpenFileCb = int32_t (PROSPER_SYSV_ABI *)(void*, const char*);
using AvpCloseFileCb = int32_t (PROSPER_SYSV_ABI *)(void*);
using AvpReadOffsetFileCb = int32_t (PROSPER_SYSV_ABI *)(void*, uint8_t*, uint64_t, uint32_t);
using AvpSizeFileCb = uint64_t (PROSPER_SYSV_ABI *)(void*);

bool avp_has_file_replacement(const AvpFileReplace& file) {
    return file.open && file.close && file.read_offset && file.size;
}

// A table with SOME entries set but not all four. prosper cannot use it — calling through the null
// member would fault — but it also must not treat it as absence, because absence and "the guest DID
// ask us to read through it and we declined" have opposite meanings for the bytes that get demuxed.
// Split out so the caller can say which one it saw. CONFIDENCE: HIGH (this is a property of the
// pointers prosper was handed, not an inference about the published contract).
bool avp_has_partial_file_replacement(const AvpFileReplace& file) {
    const bool any = file.open || file.close || file.read_offset || file.size;
    return any && !avp_has_file_replacement(file);
}

int32_t avp_file_open(const AvpFileReplace& file, const char* path) {
#if defined(_WIN32)
    return (int32_t)prosper_call_guest_sysv4(reinterpret_cast<uint64_t>(file.open),
                                             reinterpret_cast<uint64_t>(file.obj),
                                             reinterpret_cast<uint64_t>(path), 0, 0);
#else
#if defined(__linux__)
    AvpCallbackGuestFsScope guest_fs(t_avp_callback_guest_fs);
#endif
    return ((AvpOpenFileCb)file.open)(file.obj, path);
#endif
}

int32_t avp_file_close(const AvpFileReplace& file) {
#if defined(_WIN32)
    return (int32_t)prosper_call_guest_sysv4(reinterpret_cast<uint64_t>(file.close),
                                             reinterpret_cast<uint64_t>(file.obj), 0, 0, 0);
#else
#if defined(__linux__)
    AvpCallbackGuestFsScope guest_fs(t_avp_callback_guest_fs);
#endif
    return ((AvpCloseFileCb)file.close)(file.obj);
#endif
}

uint64_t avp_file_size(const AvpFileReplace& file) {
#if defined(_WIN32)
    return prosper_call_guest_sysv4(reinterpret_cast<uint64_t>(file.size),
                                    reinterpret_cast<uint64_t>(file.obj), 0, 0, 0);
#else
#if defined(__linux__)
    AvpCallbackGuestFsScope guest_fs(t_avp_callback_guest_fs);
#endif
    return ((AvpSizeFileCb)file.size)(file.obj);
#endif
}

int32_t avp_file_read(const AvpFileReplace& file, uint8_t* buffer, uint64_t position,
                      uint32_t length) {
#if defined(_WIN32)
    return (int32_t)prosper_call_guest_sysv4(reinterpret_cast<uint64_t>(file.read_offset),
                                             reinterpret_cast<uint64_t>(file.obj),
                                             reinterpret_cast<uint64_t>(buffer), position, length);
#else
#if defined(__linux__)
    AvpCallbackGuestFsScope guest_fs(t_avp_callback_guest_fs);
#endif
    return ((AvpReadOffsetFileCb)file.read_offset)(file.obj, buffer, position, length);
#endif
}

void* avp_allocate_texture(const AvpMemAllocator& memory, uint32_t align, uint32_t size) {
    if (!memory.allocate_texture) return nullptr;
#if defined(_WIN32)
    return reinterpret_cast<void*>(static_cast<uintptr_t>(prosper_call_guest_sysv4(
        reinterpret_cast<uint64_t>(memory.allocate_texture),
        reinterpret_cast<uint64_t>(memory.obj), align, size, 0)));
#elif defined(__linux__)
    AvpCallbackGuestFsScope guest_fs(t_avp_callback_guest_fs);
    return memory.allocate_texture(memory.obj, align, size);
#else
    return memory.allocate_texture(memory.obj, align, size);
#endif
}

void avp_deallocate_texture(const AvpMemAllocator& memory, void* allocation) {
    if (!memory.deallocate_texture || !allocation) return;
#if defined(_WIN32)
    prosper_call_guest_sysv4(reinterpret_cast<uint64_t>(memory.deallocate_texture),
                             reinterpret_cast<uint64_t>(memory.obj),
                             reinterpret_cast<uint64_t>(allocation), 0, 0);
#elif defined(__linux__)
    AvpCallbackGuestFsScope guest_fs(t_avp_callback_guest_fs);
    memory.deallocate_texture(memory.obj, allocation);
#else
    memory.deallocate_texture(memory.obj, allocation);
#endif
}

void avp_release_textures(const AvpMemAllocator& memory,
                          const std::vector<uint8_t*>& textures) {
    for (auto* texture : textures) {
        gpu::unregister_guest_linear_texture_layout(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(texture)));
        avp_deallocate_texture(memory, texture);
    }
}

bool avp_build_textures(const AvpMemAllocator& memory, int32_t requested,
                        uint32_t width, uint32_t height,
                        std::vector<uint8_t*>& textures, size_t& texture_bytes) {
    const bool have_allocate = memory.allocate_texture != nullptr;
    const bool have_deallocate = memory.deallocate_texture != nullptr;
    if (!have_allocate && !have_deallocate) return true; // legacy/test fallback
    uint32_t pitch = 0;
    if (!have_allocate || !have_deallocate || !avp_video_pitch(width, pitch) ||
        !avp_nv12_bytes(pitch, height, texture_bytes) || texture_bytes > UINT32_MAX)
        return false;

    // The public API permits a caller-selected framebuffer count.  Zero requests the normal double
    // buffer; bound hostile values so a malformed guest structure cannot fan out allocations.
    const int count = std::clamp(requested > 0 ? requested : 2, 2, 16);
    constexpr uint32_t alignment = 0x100;
    textures.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        void* allocation = avp_allocate_texture(memory, alignment,
                                                static_cast<uint32_t>(texture_bytes));
        if (!allocation) {
            avp_release_textures(memory, textures);
            textures.clear();
            texture_bytes = 0;
            return false;
        }
        auto* texture = static_cast<uint8_t*>(allocation);
        textures.push_back(texture);
        avp_register_frame_layout(texture, texture_bytes, pitch);
    }
    if (avp_log()) {
        fprintf(stderr,
                "[avp] guest texture buffers requested=%d allocated=%d align=%u bytes=%zu first=%p\n",
                requested, count, alignment, texture_bytes,
                textures.empty() ? nullptr : textures.front());
    }
    return true;
}

// #1955 discriminator: report the guest's file-replacement table verbatim. A title that stores media
// inside a container file can only express the byte range through these callbacks, so which titles
// supply the table — and whether all four entries are there — is a measurement, not an assumption.
// Print all four entries (and "absent" when the table is null); avp_add_source reports what it then
// did with them.
void avp_log_file_replacement(const char* entry, const AvpFileReplace& file) {
    const bool present = file.open || file.close || file.read_offset || file.size;
    fprintf(stderr,
            "[avp] %s file-replacement=%s obj=%p open=%p close=%p read-offset=%p size=%p\n",
            entry, present ? "present" : "absent", file.obj, file.open, file.close,
            file.read_offset, file.size);
}

// Fire the guest event callback. Caller MUST NOT hold g_avp_mx (the callback re-enters AvPlayer HLE).
// `data` is the callback's fourth argument: null for the state events, and a pointer to the
// notification code for the warning channel (see AVP_WARNING_ID). The guest dereferences it during
// the call, so it may live on this frame -- but it must not outlive the call, and nothing here keeps
// it.
void avp_fire_data(void* obj, AvpEventCb cb, uint32_t ev, int32_t source_id, const void* data) {
    if (!cb) return;
    if (svclog() || avp_log()) fprintf(stderr, "[avp] -> event 0x%02x\n", ev);
#if defined(_WIN32)
    prosper_call_guest_sysv4((uint64_t)(uintptr_t)cb, (uint64_t)(uintptr_t)obj, ev,
                             (uint64_t)(uint32_t)source_id, (uint64_t)(uintptr_t)data);
#elif defined(__linux__)
    {
        AvpCallbackGuestFsScope guest_fs(t_avp_callback_guest_fs);
        cb(obj, ev, source_id, const_cast<void*>(data));
    }
#else
    cb(obj, ev, source_id, const_cast<void*>(data));
#endif
    if (avp_log()) fprintf(stderr, "[avp] <- event 0x%02x\n", ev);
}
void avp_fire(void* obj, AvpEventCb cb, uint32_t ev) { avp_fire_data(obj, cb, ev, 0, nullptr); }
} // namespace

HLE(s_avplayer_init) {   // AvPlayerHandle sceAvPlayerInit(AvPlayerInitData*)
    svc_log("sceAvPlayerInit", a0,a1,a2,a3,a4,a5);
    uint64_t h = g_handle.fetch_add(1);
    AvpPlayer p;
    if (auto* d = (const AvpInitData*)PW(a0)) {
        p.memory = d->memory; p.file = d->file; p.num_fb = d->num_fb;
        p.ev_obj = d->event.obj; p.ev_cb = (AvpEventCb)d->event.event_callback;
        p.auto_start = d->auto_start != 0;
        if (avp_log()) {
            fprintf(stderr,
                "[avp] init memory=%p alloc=%p free=%p texture=%p texture-free=%p num-fb=%d auto=%d\n",
                p.memory.obj, (void*)p.memory.allocate, (void*)p.memory.deallocate,
                (void*)p.memory.allocate_texture, (void*)p.memory.deallocate_texture,
                p.num_fb, (int)p.auto_start);
            avp_log_file_replacement("init", p.file);
        }
    }
    { std::lock_guard<std::mutex> lk(g_avp_mx); g_avp[h] = std::move(p); }
    return h;   // non-NULL SceAvPlayerHandle
}
// sceAvPlayerInitEx returns an int32 error code (0 = success) and writes the handle to *out — a
// DIFFERENT ABI from sceAvPlayerInit (which returns the handle). Returning the handle here made the
// game read it as an error and abort the intro (live-captured, PPSA02664). CONFIDENCE: HIGH.
HLE(s_avplayer_initex) {   // s32 sceAvPlayerInitEx(const AvPlayerInitDataEx*, AvPlayerHandle* out)
    svc_log("sceAvPlayerInitEx", a0,a1,a2,a3,a4,a5);
    uint64_t h = g_handle.fetch_add(1);
    AvpPlayer p;
    if (auto* d = (const AvpInitDataEx*)PW(a0)) {
        p.memory = d->memory; p.file = d->file; p.num_fb = d->num_fb;
        p.ev_obj = d->event.obj; p.ev_cb = (AvpEventCb)d->event.event_callback;
        p.auto_start = d->auto_start != 0;
        if (avp_log()) {
            fprintf(stderr,
                "[avp] init-ex memory=%p alloc=%p free=%p texture=%p texture-free=%p num-fb=%d auto=%d\n",
                p.memory.obj, (void*)p.memory.allocate, (void*)p.memory.deallocate,
                (void*)p.memory.allocate_texture, (void*)p.memory.deallocate_texture,
                p.num_fb, (int)p.auto_start);
            avp_log_file_replacement("init-ex", p.file);
        }
    }
    { std::lock_guard<std::mutex> lk(g_avp_mx); g_avp[h] = std::move(p); }
    if (a1) *(uint64_t*)PW(a1) = h;
    return 0;
}
HLE(s_avplayer_isactive) {   // bool sceAvPlayerIsActive(AvPlayerHandle)
    svc_log("sceAvPlayerIsActive", a0,a1,a2,a3,a4,a5);
    void* obj = nullptr; AvpEventCb cb = nullptr; bool fire_stop = false; uint64_t active = 0;
    bool played_out = false; uint64_t clock_idle_ms = 0, clock_remaining_ms = 0, clock_duration_ms = 0;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0); if (it == g_avp.end()) return 0;
        AvpPlayer& p = it->second;
        if (p.playing) {
            const auto now = std::chrono::steady_clock::now();
            // A paused player is active and presents nothing; its media clock is restarted by
            // sceAvPlayerResume, the only exit from this state, so no time is accumulated here.
            if (p.paused) return 1;
            p.active_poll++;
            bool still = p.backend && p.backend_id >= 0
                             ? !p.backend->eof(p.backend_id)
                             : p.synthetic && p.active_poll < avp_synth_frames();
            // #1973: a consumer-driven decoder cannot report the end of a source nobody consumes.
            if (still && avp_media_played_out(p, now, clock_idle_ms, clock_remaining_ms)) {
                still = false;
                played_out = true;
                clock_duration_ms = p.duration_ms;
            }
            if (still) { active = 1; }
            else if (!p.stop_fired) { p.playing = false; p.stop_fired = true; fire_stop = true; obj = p.ev_obj; cb = p.ev_cb; }
        }
    }
    if (played_out) {
        // Loud by design, like the JumpToTime failure above: playback that ends on the clock rather
        // than by delivering its frames is a title consuming nothing, and must never read as an
        // ordinary drain in a default run.
        fprintf(stderr,
                "[avp] handle=0x%llx: the %llu ms source played out on the media clock -- nothing "
                "requested a video frame for %llu ms and %llu ms of the media remained (#1973)\n",
                (unsigned long long)a0, (unsigned long long)clock_duration_ms,
                (unsigned long long)clock_idle_ms, (unsigned long long)clock_remaining_ms);
    }
    if (fire_stop) avp_fire(obj, cb, AVP_STOP);   // playback complete -> game advances past the video
    return active;
}
HLE(s_avp_postinit)   { svc_log("sceAvPlayerPostInit", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_avp_setlogcb)   { svc_log("sceAvPlayerSetLogCallback", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_avp_streamcount) {   // s32 sceAvPlayerStreamCount(handle)
    svc_log("sceAvPlayerStreamCount", a0,a1,a2,a3,a4,a5);
    std::lock_guard<std::mutex> lk(g_avp_mx);
    auto it = g_avp.find(a0);
    int count = it != g_avp.end() && it->second.have_source
                    ? (it->second.has_audio ? 2 : 1) : 0;
    if (avp_log()) fprintf(stderr, "[avp] stream-count handle=0x%llx -> %d\n",
                           (unsigned long long)a0, count);
    return count;
}
HLE(s_avp_getstreaminfo) { // s32 sceAvPlayerGetStreamInfo(handle, stream_id, out)
    svc_log("sceAvPlayerGetStreamInfo", a0,a1,a2,a3,a4,a5);
    auto* out = (AvpStreamInfo*)PW(a2); if (!out) return 0x806a0001ull;
    std::lock_guard<std::mutex> lk(g_avp_mx);
    auto it = g_avp.find(a0); if (it == g_avp.end() || !it->second.have_source) return 0x806a0001ull;
    const AvpPlayer& p = it->second;
    if (a1 >= (p.has_audio ? 2u : 1u)) return 0x806a0001ull;
    *out = {}; out->type = a1 == 0 ? 1u : 2u;
    if (a1 == 0) {
        // The basic AvPlayerVideo has NO pitch and NO crop fields, so its `width` must be the VISIBLE
        // extent — there is no channel here through which a consumer could subtract padding. That is
        // why it deliberately differs from the Ex struct's coded `width` (see avp_fill_video_ex): the
        // two structs answer the same question through different field sets, and the basic path also
        // stages tight rows to match.
        out->details.video.width = p.width; out->details.video.height = p.height;
        out->details.video.aspect = (float)p.width / (float)p.height;
    } else {
        out->details.audio.channels = (uint16_t)p.audio_channels;
        out->details.audio.sample_rate = p.audio_rate;
        out->details.audio.size = 1024 * p.audio_channels * sizeof(int16_t);
    }
    out->duration = p.duration_ms;
    return 0;
}
HLE(s_avp_getstreaminfoex) { // s32 sceAvPlayerGetStreamInfoEx(handle, stream_id, out)
    svc_log("sceAvPlayerGetStreamInfoEx", a0,a1,a2,a3,a4,a5);
    auto* out = (AvpStreamInfoEx*)PW(a2); if (!out) return 0x806a0001ull;
    std::lock_guard<std::mutex> lk(g_avp_mx);
    auto it = g_avp.find(a0); if (it == g_avp.end() || !it->second.have_source) return 0x806a0001ull;
    const AvpPlayer& p = it->second; const uint64_t caller_size = out->this_size;
    if (a1 >= (p.has_audio ? 2u : 1u)) return 0x806a0001ull;
    *out = {}; out->this_size = caller_size; out->type = a1 == 0 ? 1u : 2u;
    if (a1 == 0) {
        uint32_t pitch = 0;
        if (!avp_video_pitch(p.width, pitch)) return 0x806a0001ull;
        if (!avp_fill_video_ex(out->details.video, p.width, p.height, pitch, p.fps))
            return 0x806a0001ull;
    } else {
        out->details.audio.channels = (uint16_t)p.audio_channels;
        out->details.audio.sample_rate = p.audio_rate;
        out->details.audio.size = 1024 * p.audio_channels * sizeof(int16_t);
    }
    out->duration = p.duration_ms;
    return 0;
}
HLE(s_avp_stream_ok) { svc_log("sceAvPlayerStreamControl", a0,a1,a2,a3,a4,a5); return 0; }
// Begin a source: resolve/open it, fire READY, and (auto_start) begin playback with PLAY.
static uint64_t avp_add_source(uint64_t handle, const char* guest_path) {
    if (!guest_path || !*guest_path) return 0x806a0001ull;

    if (avp_log()) {
        fprintf(stderr, "[avp] add source request handle=0x%llx guest='%s'\n",
                (unsigned long long)handle, guest_path);
        fflush(stderr);
    }

    prosper::video::VideoBackend* old_backend = nullptr;
    int old_backend_id = -1;
    AvpMemAllocator memory{};
    AvpFileReplace file{};
    int32_t requested_textures = 0;
    std::vector<uint8_t*> old_textures;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(handle); if (it == g_avp.end()) return 0x80000000ull;
        AvpPlayer& p = it->second;
        file = p.file;
        old_backend = p.backend;
        old_backend_id = p.backend_id;
        memory = p.memory;
        requested_textures = p.num_fb;
        old_textures = std::move(p.texture_frames);
        avp_release_frame_storage(p);
        p.texture_frame_bytes = 0;
        p.next_texture_frame = 0;
        p.have_source = false; p.synthetic = false; p.playing = false; p.paused = false;
        p.backend = nullptr; p.backend_id = -1;
    }
    if (old_backend && old_backend_id >= 0) old_backend->close(old_backend_id);
    avp_release_textures(memory, old_textures);

    // #1955: when the guest supplies a file-replacement table, IT decides which bytes of the named
    // file are the media. A title that stores a clip at an offset inside a container has no other way
    // to express that — SceAvPlayerSourceDetails carries no byte range — so opening the host path at
    // offset 0 demuxes the wrong bytes. Read the media through the guest's own reader instead.
    //
    // The read happens HERE, on the guest thread that called sceAvPlayerAddSource(Ex), because these
    // callbacks are guest code and need that thread's TCB; the decode worker is a prosper-owned host
    // thread and must never call them. The whole media is therefore buffered up front.
    //
    // EVERY branch below that declines the guest's reader falls back to opening the host path at
    // offset 0 — which is exactly the #1955 defect when the guest asked us to read a byte range. So
    // none of them may be silent: the messages here are deliberately NOT gated on PROSPER_AVPLOG,
    // because a default run has to be able to say which branch it took. (The success line stays
    // gated; it is the only one that is not a degradation.)
    std::vector<uint8_t> guest_media;
    if (avp_has_partial_file_replacement(file)) {
        // Not the same thing as no table. The guest asked to be read through, and prosper cannot
        // oblige a table it would have to call a null member of, so it is about to demux the host
        // file from offset 0 — the wrong bytes whenever the media is embedded.
        fprintf(stderr,
                "[avp] the guest file-replacement table for '%s' is INCOMPLETE "
                "(open=%p close=%p read-offset=%p size=%p); ignoring it and falling back to the "
                "host path (#1955)\n",
                guest_path, file.open, file.close, file.read_offset, file.size);
    }
    if (avp_has_file_replacement(file)) {
        const int32_t opened = avp_file_open(file, guest_path);
        const uint64_t media_bytes = opened >= 0 ? avp_file_size(file) : 0;
        if (opened < 0) {
            fprintf(stderr,
                    "[avp] guest file-replacement open('%s') failed: %d -> falling back to the host "
                    "path (#1955)\n", guest_path, (int)opened);
        } else if (avp_log()) {
            fprintf(stderr, "[avp] guest file-replacement reader: open('%s') -> %d size=%llu bytes\n",
                    guest_path, (int)opened, (unsigned long long)media_bytes);
        }
        if (opened >= 0) {
            // Bound the buffer: a guest that answers with a whole multi-hundred-megabyte container
            // is not describing a clip, and silently allocating that is worse than saying so.
            constexpr uint64_t kMaxGuestMediaBytes = 256ull * 1024 * 1024;
            if (media_bytes == 0 || media_bytes > kMaxGuestMediaBytes) {
                fprintf(stderr,
                        "[avp] guest file-replacement reader for '%s' reports %llu bytes; refusing to "
                        "buffer it and falling back to the host path (#1955)\n",
                        guest_path, (unsigned long long)media_bytes);
            } else {
                // The cap bounds the request, not the machine: a host that cannot spare the buffer
                // still throws here, and an uncaught bad_alloc inside an HLE handler ends the
                // process rather than the movie. Degrade to the same loud fallback as every other
                // branch. (Noted in #1975's review of #1974; it is a failure mode of exactly this
                // block, so it is fixed here rather than left to the buffering optimisation.)
                uint64_t position = 0;
                bool read_ok = true;
                try {
                    guest_media.resize(static_cast<size_t>(media_bytes));
                } catch (const std::bad_alloc&) {
                    fprintf(stderr,
                            "[avp] could not allocate %llu bytes for the guest file-replacement "
                            "media of '%s'; falling back to the host path (#1955)\n",
                            (unsigned long long)media_bytes, guest_path);
                    read_ok = false;
                }
                // `read_ok &&` is load-bearing, not defensive: without it a failed resize leaves an
                // EMPTY vector and the loop hands `data() + 0` to the guest's reader to write into.
                while (read_ok && position < media_bytes) {
                    const uint64_t remaining = media_bytes - position;
                    const uint32_t chunk = static_cast<uint32_t>(
                        std::min<uint64_t>(remaining, 1u << 20));
                    const int32_t got = avp_file_read(file, guest_media.data() + position, position,
                                                      chunk);
                    if (got <= 0) {
                        fprintf(stderr,
                                "[avp] guest file-replacement read('%s') failed at offset %llu: %d "
                                "-> falling back to the host path (#1955)\n",
                                guest_path, (unsigned long long)position, (int)got);
                        read_ok = false;
                        break;
                    }
                    // A count LARGER than the buffer space offered is the one answer that must never
                    // be believed. Taken at face value it advances `position` past `media_bytes`, the
                    // loop exits with `read_ok` still true, and the zero-filled tail of the buffer is
                    // handed to the demuxer AS IF it were the complete media — a silent truncation
                    // that surfaces as a corrupt stream far from here. It also means the callee may
                    // already have written past the space it was given, which prosper cannot undo but
                    // must not build on. Refuse the media and say so.
                    if (static_cast<uint32_t>(got) > chunk) {
                        fprintf(stderr,
                                "[avp] guest file-replacement read('%s') at offset %llu returned %d "
                                "for a %u-byte request; refusing the media and falling back to the "
                                "host path (#1955)\n",
                                guest_path, (unsigned long long)position, (int)got, chunk);
                        read_ok = false;
                        break;
                    }
                    position += static_cast<uint64_t>(got);
                }
                // With the over-count refused above, a loop that ran to completion can only have
                // landed exactly on media_bytes. Assert it rather than assume it: a future edit to
                // the arithmetic that stops short would otherwise deliver a short buffer as a
                // complete one. (`read_ok` is already false on the allocation and read failures, so
                // this reports only the case those two did not.)
                if (read_ok && position != media_bytes) {
                    fprintf(stderr,
                            "[avp] guest file-replacement read('%s') assembled %llu of %llu bytes; "
                            "refusing the media and falling back to the host path (#1955)\n",
                            guest_path, (unsigned long long)position,
                            (unsigned long long)media_bytes);
                    read_ok = false;
                }
                if (!read_ok) guest_media.clear();
            }
            avp_file_close(file);
        }
    }

    const std::string host_path = resolve_guest_path(guest_path);
    auto* selected_backend = prosper::video::backend();
    if (avp_log()) {
        fprintf(stderr, "[avp] opening source host='%s' backend=%d guest-media=%zu bytes\n",
                host_path.c_str(), selected_backend != nullptr, guest_media.size());
        fflush(stderr);
    }
    prosper::video::StreamInfo stream{};
    int backend_id = -1;
    if (selected_backend && !guest_media.empty()) {
        backend_id = selected_backend->open_memory(host_path, guest_media.data(),
                                                   guest_media.size());
        // A backend with no in-memory path, or media the demuxer rejects, must not silently lose the
        // titles that work today: fall back to the host file exactly as before.
        if (backend_id < 0)
            fprintf(stderr,
                    "[avp] the guest-supplied media for '%s' could not be demuxed; falling back to "
                    "the host path (#1955)\n", guest_path);
    }
    if (backend_id < 0 && selected_backend) backend_id = selected_backend->open(host_path);
    if (backend_id >= 0 && !selected_backend->info(backend_id, stream)) {
        selected_backend->close(backend_id);
        backend_id = -1;
    }
    // #1105 test lever: force a total source-open failure to exercise the graceful immediate-EOF path
    // (an undecodable source) without needing a corrupt media file. Off by default.
    if (backend_id >= 0 && getenv("PROSPER_AVP_FORCE_FAIL")) {
        selected_backend->close(backend_id);
        backend_id = -1;
    }
    const bool synthetic = backend_id < 0 && avp_synth_enabled();
    if (backend_id < 0 && !synthetic) {
        // #1105: a source that cannot be opened (corrupt file, missing codec, demux error) must not hang
        // the title. Returning an error with no lifecycle event leaves a title whose intro state machine
        // waits for the movie to finish deadlocked in its black scene (the #320 shape). Instead set up an
        // EMPTY source and drive it to a graceful immediate EOF: fire AVP_READY (so the title starts it),
        // and the first IsActive/GetVideoData poll finds no backend and no synthetic frames (:506 / :760)
        // and fires AVP_STOP — the game skips the unplayable movie and proceeds.
        if (avp_log()) fprintf(stderr, "[avp] source open failed guest='%s' host='%s' backend=%d -> "
                               "graceful skip (empty source, immediate EOF)\n",
                               guest_path, host_path.c_str(), selected_backend != nullptr);
        void* obj = nullptr; AvpEventCb cb = nullptr; bool play = false, player_missing = false;
        {
            std::lock_guard<std::mutex> lk(g_avp_mx);
            auto it = g_avp.find(handle);
            if (it == g_avp.end()) {
                player_missing = true;
            } else {
                AvpPlayer& p = it->second;
                p.guest_path = guest_path;
                p.have_source = true; p.synthetic = false;
                p.poll = 0; p.audio_poll = 0; p.active_poll = 0; p.last_ts_ms = 0;
                p.paused = false; p.stop_fired = false; p.seek_deliver = false;
                p.video_request_at = std::chrono::steady_clock::now();
                p.play_start_wall = p.video_request_at; p.play_start_pts = 0; p.gate_started = false;
                p.texture_frames.clear(); p.texture_frame_bytes = 0; p.next_texture_frame = 0;
                p.backend = nullptr; p.backend_id = -1;
                // Non-zero placeholder dims/fps: a title that queries GetStreamInfo up-front (before it
                // discovers the immediate EOF) then gets a benign aspect (w/h) and never a 0x0 surface or
                // a divide-by-zero. The source still delivers zero frames and EOFs on the first poll.
                p.width = 1920; p.height = 1080; p.fps = 30.0f;
                p.has_audio = false; p.audio_channels = 0; p.audio_rate = 0; p.duration_ms = 0;
                obj = p.ev_obj; cb = p.ev_cb;
                if (p.auto_start) { p.playing = true; play = true; }
            }
        }
        if (player_missing) return 0x80000000ull;
        avp_fire(obj, cb, AVP_READY);
        if (play) avp_fire(obj, cb, AVP_PLAY);
        return 0;
    }

    const uint32_t video_width = synthetic ? 1920u : stream.width;
    const uint32_t video_height = synthetic ? 1080u : stream.height;
    std::vector<uint8_t*> textures;
    size_t texture_bytes = 0;
    if (!avp_build_textures(memory, requested_textures, video_width, video_height,
                            textures, texture_bytes)) {
        if (backend_id >= 0) selected_backend->close(backend_id);
        if (avp_log()) fprintf(stderr,
            "[avp] source rejected: guest texture allocation failed (%ux%u num-fb=%d)\n",
            video_width, video_height, requested_textures);
        return 0x806a0001ull;
    }

    void* obj = nullptr; AvpEventCb cb = nullptr; bool play = false, player_missing = false;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(handle);
        if (it == g_avp.end()) {
            player_missing = true;
        } else {
            AvpPlayer& p = it->second;
            p.guest_path = guest_path;
            p.have_source = true; p.synthetic = synthetic;
            p.poll = 0; p.audio_poll = 0; p.active_poll = 0; p.last_ts_ms = 0; p.paused = false;
            p.stop_fired = false; p.seek_deliver = false;
            p.video_request_at = std::chrono::steady_clock::now();
            p.play_start_wall = p.video_request_at; p.play_start_pts = 0; p.gate_started = false;
            p.texture_frames = std::move(textures);
            p.texture_frame_bytes = texture_bytes;
            p.next_texture_frame = 0;
            p.backend = backend_id >= 0 ? selected_backend : nullptr;
            p.backend_id = backend_id;
            if (synthetic) {
                p.width = 1920; p.height = 1080; p.fps = 30.0f;
                p.has_audio = true; p.audio_channels = 2; p.audio_rate = 48000;
                p.duration_ms = avp_synth_duration_ms();
            } else {
                p.width = stream.width; p.height = stream.height; p.fps = stream.fps;
                p.has_audio = stream.has_audio; p.audio_channels = stream.audio_channels;
                p.audio_rate = stream.audio_rate; p.duration_ms = stream.duration_us / 1000;
            }
            obj = p.ev_obj; cb = p.ev_cb;
            if (p.auto_start) { p.playing = true; play = true; }
        }
    }
    if (player_missing) {
        if (backend_id >= 0) selected_backend->close(backend_id);
        avp_release_textures(memory, textures);
        return 0x80000000ull;
    }
    avp_fire(obj, cb, AVP_READY);
    if (play)  avp_fire(obj, cb, AVP_PLAY);
    // The duration is the media clock's only input besides delivered frames (#1973); log it so a run
    // shows whether this source has a timeline prosper can reason about at all.
    if (avp_log()) fprintf(stderr,
                           "[avp] add source handle=0x%llx guest='%s' host='%s' mode=%s auto_start=%d "
                           "duration=%llums\n",
                           (unsigned long long)handle, guest_path, host_path.c_str(),
                           synthetic ? "synthetic-explicit" : "native", (int)play,
                           (unsigned long long)(synthetic ? avp_synth_duration_ms()
                                                          : stream.duration_us / 1000));
    return 0;
}
HLE(s_avp_addsource)   {   // s32 sceAvPlayerAddSource(handle, const char* filename)
    svc_log("sceAvPlayerAddSource", a0,a1,a2,a3,a4,a5);
    return avp_add_source(a0, (const char*)PW(a1));
}
HLE(s_avp_addsourceex) {   // s32 sceAvPlayerAddSourceEx(handle, AvPlayerUriType, AvPlayerSourceDetails*)
    svc_log("sceAvPlayerAddSourceEx", a0,a1,a2,a3,a4,a5);
    const char* path = nullptr;
    if (auto* d = (const AvpSourceDetails*)PW(a2)) path = d->uri.name;
    return avp_add_source(a0, path);
}
HLE(s_avp_start) {   // s32 sceAvPlayerStart(handle) — used when auto_start is false
    svc_log("sceAvPlayerStart", a0,a1,a2,a3,a4,a5);
    void* obj = nullptr; AvpEventCb cb = nullptr; bool play = false;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0); if (it == g_avp.end()) return 0x80000000ull;
        AvpPlayer& p = it->second;
        if (p.have_source && !p.playing && !p.stop_fired) {
            p.playing = true; p.paused = false; play = true; obj = p.ev_obj; cb = p.ev_cb;
            p.video_request_at = std::chrono::steady_clock::now();
        }
    }
    if (play) avp_fire(obj, cb, AVP_PLAY);
    return 0;
}
HLE(s_avp_pause) {   // s32 sceAvPlayerPause(handle)
    svc_log("sceAvPlayerPause", a0,a1,a2,a3,a4,a5);
    void* obj = nullptr; AvpEventCb cb = nullptr; bool notify = false;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0); if (it == g_avp.end()) return 0x806a0001ull;
        if (it->second.playing && !it->second.paused) {
            it->second.paused = true;
            obj = it->second.ev_obj; cb = it->second.ev_cb; notify = true;
        }
    }
    if (notify) avp_fire(obj, cb, AVP_PAUSE);
    return 0;
}
HLE(s_avp_resume) {   // s32 sceAvPlayerResume(handle)
    svc_log("sceAvPlayerResume", a0,a1,a2,a3,a4,a5);
    void* obj = nullptr; AvpEventCb cb = nullptr; bool notify = false;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0); if (it == g_avp.end()) return 0x806a0001ull;
        it->second.seek_deliver = false;   // ordinary delivery resumes; the seek window is over
        if (it->second.playing && it->second.paused) {
            it->second.paused = false;
            obj = it->second.ev_obj; cb = it->second.ev_cb; notify = true;
            // Presentation restarts now; the media clock must measure from here, not from whenever
            // the player was paused.
            it->second.video_request_at = std::chrono::steady_clock::now();
        }
    }
    if (notify) avp_fire(obj, cb, AVP_PLAY);
    return 0;
}
// s32 sceAvPlayerJumpToTime(SceAvPlayerHandle, uint64_t offset_ms) — reposition playback (#1949).
//
// UNITS: milliseconds. PPSA30490 builds the argument as a seconds double scaled by 1000
// (eboot+0x15dc197 .. 0x15dc1ec) and then compares the resulting value directly against the
// millisecond SceAvPlayerFrameInfoEx::timeStamp that sceAvPlayerGetVideoDataEx returns, and against
// sceAvPlayerCurrentTime's millisecond position. CONFIDENCE: HIGH.
//
// WHY EVERY FAILURE PATH RETURNS NON-ZERO: this NID used to be unregistered, so the dispatcher's
// default 0 reached the guest as SCE_OK. PPSA30490 pre-sets its own failure state, tests only
// `eax == 0`, and on zero enters a wait for the player's position to move before it calls
// sceAvPlayerResume. A "successful" seek that moved nothing therefore deadlocked the title in its
// video splash forever, while the guest had a perfectly good `failed to jump while seeking` branch
// it could never reach. A truthful error is strictly better than a silent no-op here.
HLE(s_avp_jumptotime) {
    svc_log("sceAvPlayerJumpToTime", a0,a1,a2,a3,a4,a5);
    const uint64_t target_ms = a1;
    const char* reason = nullptr;
    void* ev_obj = nullptr; AvpEventCb ev_cb = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0);
        if (it == g_avp.end()) {
            reason = "unknown player handle";
        } else {
            AvpPlayer& p = it->second;
            // The backend seek runs under g_avp_mx on purpose: it tears down and recreates the
            // decode worker, and a concurrent GetVideoData/IsActive that observed the intermediate
            // end-of-decode state would fire a spurious AVP_STOP at the guest.
            if (!p.have_source) {
                reason = "no source is attached to this player";
            } else if (!p.backend || p.backend_id < 0) {
                // Includes the #1105 graceful-skip empty source and any headless/synthetic session:
                // there is no decoder to move, so say so instead of reporting a seek that cannot exist.
                reason = "the source has no host decoder to reposition";
            } else if (target_ms > UINT64_MAX / 1000ull) {
                reason = "the requested position overflows a microsecond position";
            } else if (!p.backend->seek(p.backend_id, target_ms * 1000ull)) {
                reason = "the host video backend could not seek this source";
            } else {
                // The player is now at the requested time even though no frame has been pulled yet,
                // so sceAvPlayerCurrentTime must already report it. A seek deliberately does NOT
                // change play/pause state: the observed guest sequence is pause -> jump -> resume.
                // CONFIDENCE: LOW on seeking a player that has already reported AVP_STOP — no title
                // does it, so nothing here revives one; it stays stopped with a repositioned source.
                p.last_ts_ms = target_ms;
                p.seek_deliver = true;
                // Re-anchor the delivery gate at the new position: without this, post-seek
                // frames deliver un-paced against the stale pre-seek anchor until the media
                // clock climbs back over it (#2989 re-review).
                p.play_start_wall = std::chrono::steady_clock::now();
                p.play_start_pts = target_ms;
                p.gate_started = true;
                // The source was just repositioned: the media clock measures the remainder from the
                // new position, starting now.
                p.video_request_at = std::chrono::steady_clock::now();
                ev_obj = p.ev_obj; ev_cb = p.ev_cb;
            }
        }
    }
    if (reason) {
        // A FAILED seek publishes no warning at all, and the reason is NOT that `0x806a0004` would
        // be a lie -- it is this surface's truthful failure signal, and the guests that listen for
        // it are entitled to hear it.
        //
        // The reason is that prosper's failure set above is WIDER than "the seek was refused". It
        // includes "the source has no host decoder to reposition" -- the #1105 graceful-skip empty
        // source and any headless or synthetic session -- which are benign states, not errors the
        // title should latch. And a listener does latch: PPSA25009's event callback
        // (eboot+0x13e46a0) dispatches three warning codes at +0x13e4a53, and its `0x806a0004` arm
        // at +0x13e4a77 stores a pointer into the player object and sets a byte beside it, i.e. a
        // PERSISTENT error state rather than a transient notice. Publishing that code because
        // prosper had no decoder would permanently mark a player the title was merely skipping an
        // unplayable movie on.
        //
        // Narrowing the failure set so the genuinely-refused case could announce itself is a real
        // improvement and is deliberately not attempted here; it needs a title that waits on the
        // failure arm to verify against. (Analysis from review of #3348.)
        //
        // Loud by design (not gated on PROSPER_AVPLOG): a guest-visible seek failure is exactly the
        // kind of gap that must not read as "handled" in a default run.
        fprintf(stderr, "[avp] sceAvPlayerJumpToTime handle=0x%llx target=%llu ms FAILED: %s\n",
                (unsigned long long)a0, (unsigned long long)target_ms, reason);
        return 0x806a0001ull;   // SCE_AVPLAYER_ERROR_INVALID_PARAMS, as used across this surface
    }
    // Repositioning is only half of the contract: the guest is told the seek LANDED through the
    // event callback's warning channel, and until it hears that it is entitled to sit in its own
    // wait. Published outside g_avp_mx, like every other event -- the callback re-enters AvPlayer HLE.
    //
    // Deliberately synchronous rather than deferred to a worker. Two arguments for that are TRUE
    // but cover only part of the population, and neither is what this rests on:
    //
    //   * "the guests arm their wait slot before calling" -- PPSA17337 writes its awaited code at
    //     eboot+0x15400b8, one instruction group before the call at +0x15400c2. Two titles observed.
    //   * "the join is a COUNTING semaphore, so an early post is stored rather than lost" -- true
    //     where it applies: that guest's callback posts with `lock xadd` at eboot+0x153f041 and its
    //     waiter sleeps only when the pre-decrement value was <= 0. But that is ONE binary's
    //     encoding, and it generalises no further than the first argument did. Measured across the
    //     corpus: of the 42 dumps importing this NID, the exact arm-and-join encoding appears in
    //     **11** -- one byte-identical Unity plugin. PPSA25009 is one of the 31 without it.
    //
    // What covers all 42 is weaker in form and stronger in reach, and it is a DEGRADATION bound
    // rather than a safety proof: if a guest's join is an EDGE, an early post is simply lost, and
    // that guest then waits exactly the timeout it already pays today with no notification at all.
    // So publishing synchronously is never worse than the status quo for any importer, and is
    // strictly better for the ones whose join can store it. That is the claim this code makes.
    //
    // SCOPE THE BOUND PRECISELY, because it is narrower than it first reads: it is about the WAIT.
    // An edge-triggered join loses the wake, and the guest then waits out the timeout it already
    // pays today -- but losing the wake does NOT stop the callback BODY from running, and that body
    // is demonstrably not inert. PPSA25009's handler dispatches three warning codes at
    // eboot+0x13e4a53 and its 0x806a00a3 arm at +0x13e4b5e walks a per-stream table doing real work.
    // So side effects inside an edge-triggered guest's callback are OUTSIDE this bound and remain
    // the open residual, which is the same one the PR body records: nobody can say from prosper's
    // code what any of the other importers' handlers do with a notification they have never had.
    //
    // Re-entrancy is outside the bound too, and separately covered rather than overlooked: prosper
    // has always fired guest callbacks synchronously from HLE entry points (see :2360 and :2378,
    // sceAvPlayerPause and sceAvPlayerResume), and PPSA17337's own pause -> jump -> resume sequence
    // already re-enters twice per seek on the pre-existing paths.
    //
    // Publishing here also keeps the notification ordered strictly after the reposition it reports,
    // which a deferred worker could not guarantee. (Population measured in review of #3348.)
    const uint32_t seek_complete = AVP_WARNING_SEEK_COMPLETE;
    avp_fire_data(ev_obj, ev_cb, AVP_WARNING_ID, 0, &seek_complete);
    if (avp_log()) fprintf(stderr, "[avp] jump-to-time handle=0x%llx target=%llu ms -> ok\n",
                           (unsigned long long)a0, (unsigned long long)target_ms);
    return 0;
}
// bool sceAvPlayerGetVideoData(handle, AvPlayerFrameInfo*) — deliver the next decoded frame (NV12).
// BLOCKING media-clock gate: hold the caller until the frame's PTS is due. The real AvPlayer
// delivers a frame when its PTS reaches the media clock; this guest sleeps its whole poll budget
// when a call returns "no frame yet", so a non-blocking refusal collapsed playback to ~1 frame
// per sleep (measured 0.4-1.2 fps, #2981 FMV). g_avp_mx is HELD by the caller for the whole
// wait — deliberate: no player state (stop/pause/seek/destroy) can change mid-wait, so the
// only exit is due time; the 5 s cap keeps a broken anchor from hanging the video thread.
void avp_await_due(AvpPlayer& p, uint64_t pts_ms) {
    if (!p.gate_started) return;   // no anchor yet: the first delivery arms the clock
    const auto due_wall = p.play_start_wall + std::chrono::milliseconds(
        pts_ms > p.play_start_pts ? pts_ms - p.play_start_pts : 0);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= due_wall || now >= deadline) return;
        // Sleep the remaining wait, clamped: a long first sleep keeps the loop cheap, and the
        // final steps shrink so the wakeup lands within ~1 ms of due (an 8 ms flat quantum
        // overshot every frame by ~4 ms and cost ~10% of the playback rate).
        const auto remaining = due_wall - now;
        std::this_thread::sleep_for(
            remaining > std::chrono::milliseconds(8) ? std::chrono::milliseconds(8) : remaining);
    }
}


HLE(s_avp_getvideodata) {
    svc_log("sceAvPlayerGetVideoData", a0,a1,a2,a3,a4,a5);
    auto* fi = (AvpFrameInfo*)PW(a1); if (!fi) return 0;
    void* obj = nullptr; AvpEventCb cb = nullptr; bool fire_stop = false; uint64_t result = 0;
    prosper::video::VideoBackend* log_backend = nullptr; int log_backend_id = -1;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0);
        // Paused gates delivery, EXCEPT in the window opened by a successful sceAvPlayerJumpToTime.
        // A seek repositions the player and must publish the new position while it is still paused:
        // PPSA30490 pauses, seeks, then pulls frames until the player's timestamp moves and only
        // then calls sceAvPlayerResume (eboot+0x15dc45c loop -> +0x15dc0c1). A player that delivered
        // nothing while paused could never let that guest leave its splash. The window closes on the
        // first delivered frame. avp_paused_deliver() is the separate default-off #1599 falsification
        // lever documented above, not a feature; it never changes a default run.
        if (it == g_avp.end()) return 0;
        // A consumer exists (#1973). Record that BEFORE any refusal below: a pull that returns
        // nothing because the player is paused or because decode has not caught up is still a title
        // asking for frames, and the media clock must never mistake it for one that never asks.
        it->second.video_request_at = std::chrono::steady_clock::now();
        if (!it->second.playing ||
            (it->second.paused && !it->second.seek_deliver && !avp_paused_deliver())) return 0;
        AvpPlayer& p = it->second;
        log_backend = p.backend; log_backend_id = p.backend_id;
        if (auto* b = p.backend; b && p.backend_id >= 0) {
            prosper::video::VideoFrame vf;
            // Media-clock gate, same as GetVideoDataEx above (#2981 FMV speed).
            const bool gated = p.gate_started && b->can_peek_video();
            bool have = gated ? b->peek_video(p.backend_id, vf) : true;
            if (have && gated && !p.seek_deliver) {
                // BLOCKING media-clock gate: hold until the frame's PTS is due (see the
                // Ex handler; measured 0.4-1.2 fps with a non-blocking gate on this guest).
                avp_await_due(p, vf.pts_us / 1000);
            }
            if (have && b->next_video(p.backend_id, vf)) {
                p.play_start_wall = std::chrono::steady_clock::now();
                p.play_start_pts = vf.pts_us / 1000;
                p.gate_started = true;
                uint8_t* staged = nullptr;
                uint32_t pitch = 0;
                if (avp_stage_video(p, vf, false, staged, pitch)) {
                    fi->p_data = staged;
                    fi->timestamp = vf.pts_us / 1000;
                    p.last_ts_ms = fi->timestamp;
                    p.seek_deliver = false;
                    fi->d0 = vf.width; fi->d1 = vf.height; fi->d2 = 0; fi->d3 = 0;
                    result = 1;
                }
            } else if (b->eof(p.backend_id) && !p.stop_fired) {
                p.playing = false; p.stop_fired = true; fire_stop = true; obj = p.ev_obj; cb = p.ev_cb;
            }
        } else if (p.synthetic && p.poll < avp_synth_frames()) {
            // Synthetic (headless): a black NV12 frame at the default dimensions.  Advance on
            // frame delivery: Astro Bot never polls IsActive, so an IsActive-only counter made EOF
            // and the STOP event unreachable even though it continuously pulled video frames.
            uint8_t* staged = nullptr;
            uint32_t pitch = 0;
            if (avp_stage_synthetic(p, false, staged, pitch)) {
                fi->p_data = staged; fi->timestamp = p.poll * 33ull; // ~30fps PTS (ms)
                p.last_ts_ms = fi->timestamp;
                fi->d0 = p.width; fi->d1 = p.height; fi->d2 = 0; fi->d3 = 0;
                p.poll++; result = 1;
            }
        } else if (!p.stop_fired) {
            p.playing = false; p.stop_fired = true; fire_stop = true; obj = p.ev_obj; cb = p.ev_cb;
        }
    }
    if (fire_stop) avp_fire(obj, cb, AVP_STOP);
    if (avp_log()) fprintf(stderr, "[avp] video handle=0x%llx result=%llu data=%p stop=%d\n",
                           (unsigned long long)a0, (unsigned long long)result,
                           result ? fi->p_data : nullptr, (int)fire_stop);
    return result;
}
// bool sceAvPlayerGetVideoDataEx(handle, AvPlayerFrameInfoEx*) — the game uses this Ex variant.
HLE(s_avp_getvideodataex) {
    svc_log("sceAvPlayerGetVideoDataEx", a0,a1,a2,a3,a4,a5);
    auto* fi = (AvpFrameInfoEx*)PW(a1); if (!fi) return 0;
    void* obj = nullptr; AvpEventCb cb = nullptr; bool fire_stop = false; uint64_t result = 0;
    uint64_t media_snapshot = UINT64_MAX;
    prosper::video::VideoBackend* log_backend = nullptr; int log_backend_id = -1;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0);
        // See s_avp_getvideodata: paused gates delivery except inside the post-seek window, and any
        // request — answered or refused — proves a consumer exists for the media clock (#1973).
        if (it == g_avp.end()) return 0;
        it->second.video_request_at = std::chrono::steady_clock::now();
        if (!it->second.playing ||
            (it->second.paused && !it->second.seek_deliver && !avp_paused_deliver())) return 0;
        AvpPlayer& p = it->second;
        log_backend = p.backend; log_backend_id = p.backend_id;
        if (auto* b = p.backend; b && p.backend_id >= 0) {
            prosper::video::VideoFrame vf;
            // Media-clock gate: the real AvPlayer owns the clock and refuses frames whose PTS is
            // not due yet. Measured without a gate, GRIS's 73.4 s intro delivered all 1761 frames
            // in 19.0 s of wall (3.86x) — this title presents every frame it receives, so the
            // pacing must live HERE, not in the guest. Peek first: next_video pops, and a frame
            // refused after the pop would be lost.
            // The PTS gate needs a non-destructive peek. A backend without one keeps ungated
            // delivery (the pre-gate behavior) — refusing frames it cannot re-inspect would
            // deliver nothing at all (#2989 review, blocking 1).
            const bool gated = p.gate_started && b->can_peek_video();
            bool have = gated ? b->peek_video(p.backend_id, vf) : true;
            if (have && gated && !p.seek_deliver) {
                // BLOCKING media-clock gate: hold until the frame's PTS is due (see the
                // non-Ex handler; measured 0.4-1.2 fps with a non-blocking gate on this guest).
                avp_await_due(p, vf.pts_us / 1000);
            }
            if (have && b->next_video(p.backend_id, vf)) {
                p.play_start_wall = std::chrono::steady_clock::now();
                p.play_start_pts = vf.pts_us / 1000;
                p.gate_started = true;
                media_snapshot = p.play_start_pts;
                uint8_t* staged = nullptr;
                uint32_t pitch = 0;
                // Publish the frame only if its metadata is publishable: a refused fill would
                // otherwise hand the guest a staged pointer described by a zeroed AvPlayerVideoEx.
                if (avp_stage_video(p, vf, true, staged, pitch) &&
                    avp_fill_video_ex(fi->details.video, vf.width, vf.height, pitch, p.fps)) {
                    fi->p_data = staged;
                    fi->timestamp = vf.pts_us / 1000;
                    p.last_ts_ms = fi->timestamp;
                    p.seek_deliver = false;
                    result = 1;
                }
            } else if (b->eof(p.backend_id) && !p.stop_fired) {
                p.playing = false; p.stop_fired = true; fire_stop = true; obj = p.ev_obj; cb = p.ev_cb;
                if (avp_log())
                    fprintf(stderr, "[avp] video-ex EOF handle=0x%llx decoder_dropped=%llu\n",
                            (unsigned long long)a0,
                            (unsigned long long)b->video_frames_dropped(p.backend_id));
            }
        } else if (p.synthetic && p.poll < avp_synth_frames()) {
            uint8_t* staged = nullptr;
            uint32_t pitch = 0;
            if (avp_stage_synthetic(p, true, staged, pitch) &&
                avp_fill_video_ex(fi->details.video, p.width, p.height, pitch, p.fps)) {
                fi->p_data = staged; fi->timestamp = p.poll * 33ull;
                p.last_ts_ms = fi->timestamp;
                p.poll++; result = 1;
            }
        } else if (!p.stop_fired) {
            p.playing = false; p.stop_fired = true; fire_stop = true; obj = p.ev_obj; cb = p.ev_cb;
        }
    }
    if (fire_stop) avp_fire(obj, cb, AVP_STOP);
    if (avp_log()) {
        static std::atomic<uint64_t> delivered{0};
        if (result)
            fprintf(stderr, "[avp] video-ex handle=0x%llx DELIVER #%llu t=%llums pts=%llums media=%llums\n",
                    (unsigned long long)a0,
                    (unsigned long long)delivered.fetch_add(1) + 1,
                    (unsigned long long)avp_ms(), (unsigned long long)fi->timestamp,
                    (unsigned long long)media_snapshot);
        else
            fprintf(stderr, "[avp] video-ex handle=0x%llx empty t=%llums media=%llums q=%d\n",
                    (unsigned long long)a0, (unsigned long long)avp_ms(),
                    (unsigned long long)media_snapshot,
                    log_backend ? log_backend->video_queue_depth(log_backend_id) : -2);
    }
    return result;
}
HLE(s_avp_getaudiodata)   {   // bool sceAvPlayerGetAudioData(handle, AvPlayerFrameInfo*)
    svc_log("sceAvPlayerGetAudioData", a0,a1,a2,a3,a4,a5);
    std::lock_guard<std::mutex> lk(g_avp_mx);
    auto it = g_avp.find(a0);
    if (it == g_avp.end() || !it->second.playing || it->second.paused) return 0;
    auto* fi = (AvpFrameInfo*)PW(a1); if (!fi) return 0;
    AvpPlayer& p = it->second;
    if (auto* b = p.backend; b && p.backend_id >= 0) {
        prosper::video::AudioFrame af;
        if (!b->next_audio(p.backend_id, af)) return 0;
        if (!af.pcm || af.channels == 0 || af.samples == 0 ||
            af.samples > SIZE_MAX / af.channels) return 0;
        const size_t sample_count = static_cast<size_t>(af.samples) * af.channels;
        if (sample_count > SIZE_MAX / sizeof(int16_t)) return 0;
        p.audio.resize(sample_count);
        memcpy(p.audio.data(), af.pcm, sample_count * sizeof(int16_t));
        fi->p_data = reinterpret_cast<uint8_t*>(p.audio.data()); fi->timestamp = af.pts_us / 1000;
        if (fi->timestamp > p.last_ts_ms) p.last_ts_ms = fi->timestamp;
        AvpAudio details{(uint16_t)af.channels, {}, af.sample_rate,
                         (uint32_t)(af.samples * af.channels * sizeof(int16_t)), {}};
        memcpy(&fi->d0, &details, sizeof(details));
        if (avp_log()) {
            // A real counter: this field was a literal 0, so every line read "#0" and the one thing
            // the field exists to answer -- how much audio has actually been handed over -- could
            // only be recovered by counting log lines.
            static std::atomic<uint64_t> audio_delivered{0};
            fprintf(stderr, "[avp] audio-deliver handle=0x%llx #%llu t=%llums pts=%llums samples=%u\n",
                    (unsigned long long)a0,
                    (unsigned long long)audio_delivered.fetch_add(1) + 1,
                    (unsigned long long)avp_ms(),
                    (unsigned long long)fi->timestamp, af.samples);
        }
        return 1;
    }
    if (!p.synthetic || p.audio_poll >= avp_synth_frames()) return 0;
    constexpr uint32_t samples = 1024, channels = 2;
    if (p.audio.size() != samples * channels) p.audio.assign(samples * channels, 0);
    fi->p_data = (uint8_t*)p.audio.data();
    fi->timestamp = p.audio_poll * 21ull; // 1024/48 kHz, milliseconds
    if (fi->timestamp > p.last_ts_ms) p.last_ts_ms = fi->timestamp;
    AvpAudio details{2, {}, 48000, samples * channels * sizeof(int16_t), {}};
    memcpy(&fi->d0, &details, sizeof(details));
    p.audio_poll++;
    if (avp_log()) fprintf(stderr, "[avp] audio handle=0x%llx result=1 poll=%llu\n",
                           (unsigned long long)a0, (unsigned long long)p.audio_poll);
    return 1;
}
// u64 sceAvPlayerCurrentTime(handle) — current playback position in MILLISECONDS. Returns the
// newest delivered frame timestamp (video, or audio when audio leads), 0 before the first frame or
// for an unknown handle. Alex Kidd DX's pre-level cutscene runner polls this to pace/finish the
// sequence; the old unresolved-import 0 held its fade-to-black over an already-running level
// forever (#320). Fires no guest callbacks, so no AVP_CALLBACK_ENTRY shim is needed.
HLE(s_avp_current_time) {
    svc_log("sceAvPlayerCurrentTime", a0,a1,a2,a3,a4,a5);
    std::lock_guard<std::mutex> lk(g_avp_mx);
    auto it = g_avp.find(a0);
    const uint64_t now_ms = it == g_avp.end() ? 0 : it->second.last_ts_ms;
    if (avp_log()) fprintf(stderr, "[avp] current-time handle=0x%llx -> %llu ms\n",
                           (unsigned long long)a0, (unsigned long long)now_ms);
    return now_ms;
}
HLE(s_avp_stop) {   // s32 sceAvPlayerStop(handle)
    svc_log("sceAvPlayerStop", a0,a1,a2,a3,a4,a5);
    void* obj = nullptr; AvpEventCb cb = nullptr; bool fire_stop = false;
    prosper::video::VideoBackend* backend = nullptr; int backend_id = -1;
    AvpMemAllocator memory{};
    std::vector<uint8_t*> textures;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0); if (it == g_avp.end()) return 0;
        AvpPlayer& p = it->second;
        backend = p.backend; backend_id = p.backend_id;
        memory = p.memory;
        textures = std::move(p.texture_frames);
        avp_release_frame_storage(p);
        p.texture_frame_bytes = 0; p.next_texture_frame = 0;
        p.backend = nullptr; p.backend_id = -1; p.have_source = false; p.synthetic = false;
        p.seek_deliver = false;
        if (p.playing && !p.stop_fired) {
            p.playing = false; p.paused = false; p.stop_fired = true;
            fire_stop = true; obj = p.ev_obj; cb = p.ev_cb;
        }
    }
    if (backend && backend_id >= 0) backend->close(backend_id);
    avp_release_textures(memory, textures);
    if (fire_stop) avp_fire(obj, cb, AVP_STOP);
    return 0;
}
HLE(s_avp_close) {   // s32 sceAvPlayerClose(handle)
    svc_log("sceAvPlayerClose", a0,a1,a2,a3,a4,a5);
    prosper::video::VideoBackend* backend = nullptr; int backend_id = -1;
    AvpMemAllocator memory{};
    std::vector<uint8_t*> textures;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0);
        if (it != g_avp.end()) {
            backend = it->second.backend; backend_id = it->second.backend_id;
            memory = it->second.memory;
            textures = std::move(it->second.texture_frames);
            avp_release_frame_storage(it->second);
            g_avp.erase(it);
        }
    }
    if (backend && backend_id >= 0) backend->close(backend_id);
    avp_release_textures(memory, textures);
    return 0;
}

#ifndef _WIN32
// Pass the HLE-entry stack pointer to callback-producing AvPlayer handlers.  On Linux it identifies
// the guest TCB saved by the import stub; on macOS callback_guest_fs_from_entry_stack() fails closed
// to zero because guest TLS uses trap emulation rather than a hardware %fs switch.
#define AVP_CALLBACK_ENTRY(entry, target, handler) \
    extern "C" uint64_t target(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t); \
    PROSPER_ASM_TRAMPOLINE(entry, target) \
    extern "C" void entry(); \
    extern "C" uint64_t target(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, \
                                uint64_t a4, uint64_t a5, uint64_t entry_rsp) { \
        const uint64_t guest_fs = callback_guest_fs_from_entry_stack(entry_rsp); \
        if (avp_log() && guest_fs) { \
            const uint64_t guest_ra = *(const uint64_t*)(uintptr_t)(entry_rsp + 0x30); \
            /* The title-facing AvPlayer adapters use a frame pointer plus two saved registers; */ \
            /* retain their caller as a diagnostic without changing normal logging or behavior. */ \
            const uint64_t guest_caller = *(const uint64_t*)(uintptr_t)(entry_rsp + 0x50); \
            fprintf(stderr, "[avp] call %s guest_ra=0x%llx guest_caller=0x%llx guest_fs=1\n", \
                    #handler, (unsigned long long)guest_ra, (unsigned long long)guest_caller); \
        } \
        (void)guest_fs; \
        /* Only Linux swaps hardware %fs at the import boundary. */ \
        PROSPER_AVP_ENTRY_SCOPE(guest_fs) \
        return handler(a0, a1, a2, a3, a4, a5); \
    }

#if defined(__linux__)
#define PROSPER_AVP_ENTRY_SCOPE(guest_fs) AvpEntryGuestFsScope avp_entry_guest_fs_scope(guest_fs);
#else
#define PROSPER_AVP_ENTRY_SCOPE(guest_fs)
#endif

AVP_CALLBACK_ENTRY(s_avp_addsource_entry, s_avp_addsource_entry_c, s_avp_addsource)
AVP_CALLBACK_ENTRY(s_avp_addsourceex_entry, s_avp_addsourceex_entry_c, s_avp_addsourceex)
AVP_CALLBACK_ENTRY(s_avp_start_entry, s_avp_start_entry_c, s_avp_start)
AVP_CALLBACK_ENTRY(s_avplayer_isactive_entry, s_avplayer_isactive_entry_c, s_avplayer_isactive)
AVP_CALLBACK_ENTRY(s_avp_getvideodata_entry, s_avp_getvideodata_entry_c, s_avp_getvideodata)
AVP_CALLBACK_ENTRY(s_avp_getvideodataex_entry, s_avp_getvideodataex_entry_c, s_avp_getvideodataex)
AVP_CALLBACK_ENTRY(s_avp_stop_entry, s_avp_stop_entry_c, s_avp_stop)
AVP_CALLBACK_ENTRY(s_avp_close_entry, s_avp_close_entry_c, s_avp_close)
AVP_CALLBACK_ENTRY(s_avp_pause_entry, s_avp_pause_entry_c, s_avp_pause)
AVP_CALLBACK_ENTRY(s_avp_resume_entry, s_avp_resume_entry_c, s_avp_resume)

#undef PROSPER_AVP_ENTRY_SCOPE
#undef AVP_CALLBACK_ENTRY
#endif
// sceUserServiceGetGamePresets(userId, presets): MUST return success (0). The Unity engine's
// per-controller connection check (eboot 0x14707e0, reached from the pad "reset" path 0x1470ca0)
// calls this and treats ANY non-zero user-service return as "controller invalid" — it then clears the
// pad's connected flag (eboot+0x201d150) EVERY FRAME, so scePadGetLoginUserIdList-based enumeration
// reports 0 controllers and the game never calls scePadOpen. i.e. this one wrong errno silently killed
// ALL gamepad input in The Messenger (#234). The prior 0x80960006 was chosen to avoid the game reading
// an untouched (garbage) out-struct; instead return 0 AND zero the payload: the out-struct's first field
// is its byte size (the caller sets it, e.g. 0x30) — zero the bytes after it (bounded) for clean default
// presets. CONFIDENCE: HIGH — root-caused via HWBP/HWWATCH on the eboot connection flag; verified fix
// makes scePadOpen + scePadReadState fire and input register.
HLE(s_gamepresets) {
    if (a1) { uint32_t sz = *(uint32_t*)PW(a1); if (sz > 8 && sz <= 0x400) memset((char*)PW(a1) + 8, 0, sz - 8); }
    return 0;
}

// --- NP / online: an honest OFFLINE, SIGNED-OUT console (#306). --------------------------------
// The DOLL front-end boot flow stalls at UE4's InstallBundleManager PatchCheck because the Np
// sign-in queries returned success-with-garbage-out: "success" from sceNpGetOnlineId told the
// game a user IS signed in, pushing its patch/entitlement check onto online branches that then
// wait forever on fake Http/WebApi handles (docs/DOLL_LOADING_PROGRESSION.md §3). A real console
// with no PSN sign-in answers these with SCE_NP_ERROR_SIGNED_OUT so the flow resolves to its
// offline path (UE4 PatchCheck -> NoLoggedInUser).
// Error space verified against shadPS4 np_error.h (PS4-inherited; identical export names+NIDs in
// the PS5 3.20 libSceNpManager stub table). SIGNED_OUT = 0x80550006. CONFIDENCE: HIGH on the PS4
// semantic (shadPS4 returns exactly this from sceNpGetOnlineId/sceNpGetAccountIdA when no user is
// signed in), MED that the PS5 errno value is unchanged (same 0x8055 Np facility, same API).
static constexpr uint64_t NP_ERR_SIGNED_OUT = 0x80550006ull;   // SCE_NP_ERROR_SIGNED_OUT
HLE(s_np_state)       { if (a1) *(int32_t*)PW(a1) = 1; return 0; }           // SCE_NP_STATE_SIGNED_OUT
HLE(s_np_reach)       { if (a1) *(int32_t*)PW(a1) = 0; return 0; }
// sceNpGetAccountIdA(userId, u64* accountId): signed-out consoles zero the id AND return the
// signed-out error (shadPS4 np_manager.cpp:579 does exactly this). The previous success+0 was
// contradictory garbage ("you have a user; their account id is 0").
HLE(s_np_accountid)   { if (a1) *(uint64_t*)PW(a1) = 0; return NP_ERR_SIGNED_OUT; }
// sceNpGetAccountCountryA: signed-out, out-struct untouched (matching the sibling getters). Was returning
// SUCCESS with a zeroed country code -> the guest reads a blank as a valid region key and takes an
// age/region/store path (the #306 success-masks-offline wedge) instead of the clean offline branch.
HLE(s_np_country)     { return NP_ERR_SIGNED_OUT; }
// sceNpGetOnlineId(userId, SceNpOnlineId* out): the signed-out error, out untouched (shadPS4
// np_manager.cpp:618). The unimplemented success+garbage here is what faked the sign-in.
HLE(s_np_getonlineid) { svc_log("sceNpGetOnlineId", a0,a1,a2,a3,a4,a5); return NP_ERR_SIGNED_OUT; }
// sceNpGetNpId(userId, SceNpId* out): the signed-out error, out untouched (shadPS4 np_manager.cpp) — the
// most common identity getter. Was MISSING -> the return-0 stub told the guest a valid online identity
// existed (a garbage ~40-byte SceNpId), pushing it onto online/entitlement branches that dead-end. This
// was a direct hole in the #306 signed-out fix (the sibling getters below were done, this one wasn't).
HLE(s_np_getnpid)     { svc_log("sceNpGetNpId", a0,a1,a2,a3,a4,a5); return NP_ERR_SIGNED_OUT; }
// sceNpCheckNpAvailability / ...A / CheckNpReachability: an honest signed-out console. Were MISSING -> the
// return-0 stub answered "PSN is available", pushing the guest onto online branches that then wait forever
// (the #306 wedge class). NOTE: the async poll/request pairing (sceNpPollAsync / CreateAsyncRequest) is
// deliberately left for a follow-up -- its exact completion flow needs a live PROSPER_SVCLOG capture.
HLE(s_np_check_avail) { svc_log("sceNpCheckNpAvailability", a0,a1,a2,a3,a4,a5); return NP_ERR_SIGNED_OUT; }
// The local network libraries allocate opaque contexts even on a disconnected console; connection
// state is reported separately through NetCtl/NP. Returning generic success (0) from these ID-returning
// constructors instead creates an invalid context and makes their owner's initialization fail. The
// signatures and positive-return contract agree with the PS5 3.20 symbol table and the independently
// implemented SDK surface; no online identity or connectivity is fabricated here.
namespace {
std::atomic<int32_t> g_net_pool_id{1};
std::atomic<int32_t> g_ssl_context_id{1};
std::atomic<int32_t> g_http2_context_id{1};
std::atomic<int32_t> g_npweb_context_id{1};
std::atomic<int32_t> g_npweb_user_context_id{1001};
}
HLE(s_net_pool_create) {
    svc_log("sceNetPoolCreate", a0,a1,a2,a3,a4,a5);
    if (!svc_ptrish(a0) || (int32_t)a1 <= 0) return 0x80410116ull; // SCE_NET_ERROR_EINVAL (0x80410100 | BSD EINVAL 22, #3300)
    return (uint64_t)(uint32_t)g_net_pool_id.fetch_add(1);
}
HLE(s_ssl_init) {
    svc_log("sceSslInit", a0,a1,a2,a3,a4,a5);
    if (!a0) return 0x8094000cull; // SCE_SSL_ERROR_OUT_OF_SIZE
    return (uint64_t)(uint32_t)g_ssl_context_id.fetch_add(1);
}
HLE(s_http2_init) {
    svc_log("sceHttp2Init", a0,a1,a2,a3,a4,a5);
    return (uint64_t)(uint32_t)g_http2_context_id.fetch_add(1);
}
HLE(s_npweb_init) {
    svc_log("sceNpWebApi2Initialize", a0,a1,a2,a3,a4,a5);
    return (uint64_t)(uint32_t)g_npweb_context_id.fetch_add(1);
}
HLE(s_npweb_create_user_context) {
    svc_log("sceNpWebApi2CreateUserContext", a0,a1,a2,a3,a4,a5);
    return (uint64_t)(uint32_t)g_npweb_user_context_id.fetch_add(1);
}
HLE(s_netctl_getresult) {
    svc_log("sceNetCtlGetResult", a0,a1,a2,a3,a4,a5);
    if (!svc_ptrish(a1)) return 0x80412103ull; // SCE_NET_CTL_ERROR_INVALID_ADDR
    *(int32_t*)PW(a1) = 0;
    return 0;
}
// sceNpHasSignedUp(userId, bool* hasSignedUp): Sonic passes an ODD output address (..cdf), proving
// the result is one byte rather than int32. An offline initial user has no NP signup in this headless
// profile. Success + false lets the caller select its signed-out branch without inventing an online
// identity; the old success-with-untouched-stack answer made that branch random. CONFIDENCE: HIGH
// on the ABI and byte width (live PS5 title trace), MED on success+false for the offline profile.
HLE(s_np_has_signed_up) {
    svc_log("sceNpHasSignedUp", a0,a1,a2,a3,a4,a5);
    if (!svc_ptrish(a1)) return 0x80550003ull; // SCE_NP_ERROR_INVALID_ARGUMENT
    *(uint8_t*)PW(a1) = 0;
    return 0;
}

// --- mouse (report a device that exists but has no input; pad -> hle_pad.cpp real backend) ---
HLE(s_open)           { return g_handle++; }                                 // sceMouseOpen -> handle
// sceMouseRead(handle, SceMouseData*, num) returns the number of mouse events read. SceMouseData
// (~0x18 bytes) is NOT ScePadData — sharing the pad stub returned one "valid" entry whose memset
// overran a single-entry mouse buffer, and the game consumed a phantom mouse event every call. No
// mouse attached: zero one entry defensively, report 0 events.
HLE(s_mouse_read)     { if (a1) memset(PW(a1), 0, 0x18); return 0; }

// --- libSceIme keyboard API (#186) ---
// PPSA02664 polls this every frame in its input loop. We have no physical PS5 keyboard, so we report a
// consistent "no keyboard connected" state: Open/Update succeed, GetResourceId returns an empty
// resource array (no keyboards), GetInfo reports a disconnected device — the game then uses controller
// / on-screen input instead of waiting on a keyboard. Struct layouts + field offsets verified against
// shadPS4 src/core/libraries/ime/ime_common.h; error codes from ime_error.h. Note sceImeKeyboardOpen
// returns an Error (0 = SCE_OK), NOT a handle — so success is 0. CONFIDENCE: HIGH on the layouts; MED
// on the device/status enum "disconnected" == 0.
//   OrbisImeKeyboardResourceIdArray: user_id@0(s32) resource_id[5]@4(u32)                     size 24
//   OrbisImeKeyboardInfo: user_id@0 device@4 type@8 repeat_delay@12 repeat_rate@16 status@20 rsv[12]@24  size 36
// --- IME keyboard event injection (issue #1093) -------------------------------------------------
// PPSA02664 (Alex Kidd) reads its "press any button" / menu input ONLY through the IME keyboard
// path: it opens a keyboard and calls sceImeUpdate(handler) every frame, and never touches
// libScePad or sceUserService. sceImeUpdate must invoke handler(arg, &event) for each queued
// keyboard event; the old stub returned without calling, delivered no input, and the title never
// advanced (injected pad input was inert — the game wasn't reading the pad).
//
// Event ABI derived directly from the guest handler at eboot+0xf2c540 (PPSA02664 disassembly):
//   handler(rdi=arg, rsi=SceImeEvent*)
//   SceImeEvent:  +0x00 u32 id   (0x101 = KEY_DOWN, 0x102 = KEY_UP; 0x103..0x106 = other kbd events)
//                 +0x08 u16 keycode  (USB HID usage id; indexes the guest's keycode->bit table)
//                 +0x0a u16 (modifier/char)   +0x0c u32 (status)
//   KEY_DOWN sets the guest's current + newly-pressed keyboard bitmasks; KEY_UP clears them.
// HID keycodes verified against the guest's table: Enter=0x28, Space=0x2c, Z=0x1d, A=0x04.
// CONFIDENCE: HIGH on the id/keycode/offset layout (read from the guest handler + its HID table).
namespace {
struct ImeKeyEvent { uint16_t hid; bool down; };
// OrbisImeKeyboardParam, shared by the PS4 compatibility API exposed on PS5. The callback context is
// supplied at open time, while sceImeUpdate supplies the callback function used for that pump. Keeping
// only the callback and dropping `arg` happened to work for handlers that ignored rdi, but faults any
// normal C++ thunk that expects its object pointer there.
struct ImeKeyboardParam {
    uint32_t option;
    uint32_t reserved1;
    uint64_t arg;
    uint64_t handler;
    uint64_t reserved2;
};
static_assert(sizeof(ImeKeyboardParam) == 0x20 && offsetof(ImeKeyboardParam, arg) == 0x08 &&
              offsetof(ImeKeyboardParam, handler) == 0x10);
struct ImeKeyboardContext {
    bool open = false;
    int32_t user_id = -1;
    uint64_t arg = 0;
    uint64_t handler = 0;
};
std::mutex g_ime_mx;
std::deque<ImeKeyEvent> g_ime_queue;
ImeKeyboardContext g_ime_keyboard;

void ime_deliver(uint64_t handler, uint64_t arg, const ImeKeyEvent& ev, uint64_t guest_fs) {
    if (!handler) return;
    alignas(16) uint8_t e[0x40] = {0};
    *(uint32_t*)(e + 0x00) = ev.down ? 0x101u : 0x102u;
    *(uint16_t*)(e + 0x08) = ev.hid;
    // The guest handler is guest code that reads guest TLS (its per-thread allocator etc.). But
    // sceImeUpdate is an HLE handler, so the import stub already swapped this thread to HOST %fs before
    // it ran (#1155). Calling the guest handler directly here therefore ran it on host %fs, so its
    // thread-local reads hit host glibc garbage and the first allocation it drives (the menu->gameplay
    // transition on a key press) faulted at a near-null address — Evergate's SIGSEGV at eboot+0xaf8431
    // (#1286). Restore the caller's guest %fs (recovered from the import-stub frame) for the duration of
    // the call, then swap back to host %fs. Linux-only: only Linux swaps hardware %fs at the import
    // boundary (guest_fs is 0 on Windows/macOS -> unchanged direct call).
#if defined(__linux__) && !defined(__APPLE__)
    uint64_t saved_fs = 0; bool swapped = false;
    if (guest_fs) {
        __asm__ volatile("rdfsbase %0" : "=r"(saved_fs));
        __asm__ volatile("wrfsbase %0" : : "r"(guest_fs));
        swapped = true;
    }
#else
    (void)guest_fs;
#endif
    // The handler is GUEST code, so it reads its arguments per the SysV ABI (rdi, rsi). On Windows the
    // host is MS x64 (rcx, rdx) and PROSPER_SYSV_ABI is empty on every platform (dispatch.hpp:27) --
    // the guest<->host conversion lives in the import-stub trampoline, which only covers the
    // guest->host direction. A raw call here therefore handed the guest whatever happened to be in
    // rdi/rsi. Observed on Blue Prince (PPSA25009): the guest read rsi == 0x28, the autokey's own
    // default Enter HID, as a pointer and faulted at eboot+0x137c063 (`mov eax,[rsi]`), after which
    // the title spun on `sce::Agc::suspendPoint` forever with a black screen. Use the same host->guest
    // trampoline the AvPlayer callbacks in this file already use.
    // Only GUEST code needs the trampoline. prosper's own tests install a HOST function as the
    // handler to observe delivery and the event shape; that address is already the host ABI and must
    // not be entered through the guest call path (doing so segfaults ime_input). Classify by module
    // aperture, the same way the rest of the tree names guest addresses.
    // CONFIDENCE: HIGH -- an IL2CPP title's handler is AOT code in the eboot or a PRX, and the
    // runtime-PRX pool is labelled too; there is no observed guest handler outside a module.
#ifdef _WIN32
    if (std::strcmp(prosper::guest_module_name((uint64_t)(uintptr_t)handler), "mapped/host") != 0) {
        prosper_call_guest_sysv4((uint64_t)(uintptr_t)handler, arg, (uint64_t)(uintptr_t)e, 0, 0);
    } else {
        // Unclassified target: prosper's own test double, or a guest handler somewhere this
        // classifier does not know about. The second case would take the host ABI and silently hand
        // the guest wrong arguments -- the exact defect this code fixes -- so say so rather than
        // reintroduce it quietly. Bounded; a host test double trips it at most a few times.
        static std::atomic<int> unclassified{0};
        if (unclassified.fetch_add(1) < 4)
            fprintf(stderr, "[ime] handler 0x%llx is outside every guest module aperture; calling it "
                            "directly. If this is guest code it will receive MS-x64 arguments and "
                            "misread them (#2136).\n", (unsigned long long)(uintptr_t)handler);
        ((void (PROSPER_SYSV_ABI *)(uint64_t, void*))(uintptr_t)handler)(arg, e);
    }
#else
    ((void (PROSPER_SYSV_ABI *)(uint64_t, void*))(uintptr_t)handler)(arg, e);
#endif
#if defined(__linux__) && !defined(__APPLE__)
    if (swapped) __asm__ volatile("wrfsbase %0" : : "r"(saved_fs));
#endif
}

// PROSPER_IME_AUTOKEY=1: deliver a repeating Enter down/up pulse so headless routes advance a
// "press any button" / menu that reads the IME keyboard. Diagnostic/verification lever; the real
// per-key input comes from a frontend via ime_push_key (below). PROSPER_IME_AUTOKEY_HID overrides
// the HID usage (default 0x28 Enter) and accepts a comma-separated LIST (e.g. "0x28,0x2c,0x1d") that
// is cycled one key per press-window — so a single headless run can probe which key a dialog accepts.
// Cadence: down, then up two ticks later, then the next key in the list two windows later.
void ime_autokey_tick(uint64_t handler, uint64_t arg, uint64_t guest_fs) {
    static const int on = [] { const char* e = getenv("PROSPER_IME_AUTOKEY");
                               return e && strtol(e, nullptr, 0) != 0 ? 1 : 0; }();
    if (!on) return;
    static const std::vector<uint16_t> hids = [] {
        std::vector<uint16_t> v;
        const char* e = getenv("PROSPER_IME_AUTOKEY_HID");
        if (e && *e) for (const char* p = e; *p; ) {
            char* end = nullptr; long val = strtol(p, &end, 0);
            if (end == p) break;
            v.push_back((uint16_t)val);
            p = (*end == ',') ? end + 1 : end;
        }
        if (v.empty()) v.push_back(0x28);   // default Enter
        return v;
    }();
    static std::atomic<uint64_t> tick{0};
    uint64_t t = tick.fetch_add(1);
    const uint16_t hid = hids[(t / 90) % hids.size()];
    if ((t % 90) == 0) {
        ime_deliver(handler, arg, {hid, true}, guest_fs);
        if (svclog()) fprintf(stderr, "[ime-autokey] deliver hid=0x%x down\n", hid);
    } else if ((t % 90) == 2) {
        ime_deliver(handler, arg, {hid, false}, guest_fs);
    }
}

// PROSPER_IME_SCRIPT: deterministic headless keyboard input, anchored to the first sceImeUpdate
// call carrying a non-null handler. Entries are `fN:HID` or `fA-B:HID`, separated by semicolons or
// newlines; HID accepts decimal or 0x-prefixed USB usage ids. A point entry is held for two update
// ticks. Prefix the value with '@' to load the same text from a route file. Unlike AUTOKEY this is a
// finite route: transitions are emitted only when a key enters/leaves an active window.
std::string ime_script_text(const char* source, std::string& error) {
    if (!source || !*source) return {};
    if (*source != '@') return source;
    FILE* f = fopen(source + 1, "rb");
    if (!f) { error = std::string("cannot open route file: ") + (source + 1); return {}; }
    std::string text;
    char buf[4096];
    while (size_t n = fread(buf, 1, sizeof buf, f)) text.append(buf, n);
    if (ferror(f)) error = std::string("cannot read route file: ") + (source + 1);
    fclose(f);
    return text;
}

bool ime_script_u64(const char* p, char** tail, uint64_t& value) {
    if (!p || !((*p >= '0' && *p <= '9'))) { if (tail) *tail = (char*)p; return false; }
    errno = 0;
    char* end = nullptr;
    const int base = p[0] == '0' && (p[1] == 'x' || p[1] == 'X') ? 16 : 10;
    const unsigned long long parsed = strtoull(p, &end, base);
    if (end == p || errno == ERANGE || parsed > UINT64_MAX) { if (tail) *tail = end; return false; }
    if (tail) *tail = end;
    value = (uint64_t)parsed;
    return true;
}

bool parse_ime_script_impl(const std::string& text, std::vector<ImeScriptWindow>& out,
                           std::string& error) {
    out.clear();
    size_t pos = 0;
    while (pos < text.size()) {
        size_t end = text.find_first_of(";\n\r", pos);
        if (end == std::string::npos) end = text.size();
        std::string item = text.substr(pos, end - pos);
        pos = end + 1;
        if (size_t hash = item.find('#'); hash != std::string::npos) item.resize(hash);
        size_t first_nonspace = item.find_first_not_of(" \t");
        if (first_nonspace == std::string::npos) continue;
        size_t last_nonspace = item.find_last_not_of(" \t");
        item = item.substr(first_nonspace, last_nonspace - first_nonspace + 1);
        if (item.empty()) continue;
        if (item[0] != 'f') { error = "entry must start with f: " + item; return false; }
        const char* p = item.c_str() + 1;
        char* tail = nullptr;
        uint64_t first = 0;
        if (!ime_script_u64(p, &tail, first)) { error = "invalid frame anchor: " + item; return false; }
        uint64_t last = first == UINT64_MAX ? first : first + 1;
        // Point entries hold down for two update ticks (or one at the representational maximum).
        if (*tail == '-') {
            p = tail + 1;
            if (!ime_script_u64(p, &tail, last) || last < first) {
                error = "invalid frame range: " + item; return false;
            }
        }
        if (*tail != ':') { error = "missing HID separator: " + item; return false; }
        p = tail + 1;
        uint64_t hid = 0;
        if (!ime_script_u64(p, &tail, hid)) { error = "invalid HID usage: " + item; return false; }
        while (*tail == ' ' || *tail == '\t') ++tail;
        if (*tail || hid == 0 || hid > UINT16_MAX) {
            error = "invalid HID usage: " + item; return false;
        }
        out.push_back({first, last, (uint16_t)hid});
    }
    error.clear();
    return true;
}

class ImeScriptRuntime {
public:
    ImeScriptRuntime() {
        const char* source = getenv("PROSPER_IME_SCRIPT");
        if (!source || !*source) return;
        std::string error;
        const std::string text = ime_script_text(source, error);
        if (error.empty()) parse_ime_script_impl(text, entries_, error);
        if (!error.empty()) fprintf(stderr, "[ime] PROSPER_IME_SCRIPT: %s\n", error.c_str());
    }

    void tick(uint64_t handler, uint64_t arg, uint64_t guest_fs) {
        if (!handler || entries_.empty()) return;
        bool become_dispatcher = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::vector<uint16_t> next;
            for (const auto& e : entries_)
                if (tick_ >= e.first && tick_ <= e.last &&
                    std::find(next.begin(), next.end(), e.hid) == next.end()) next.push_back(e.hid);
            for (uint16_t hid : active_)
                if (std::find(next.begin(), next.end(), hid) == next.end())
                    pending_.push_back({handler, arg, guest_fs, {hid, false}});
            for (uint16_t hid : next)
                if (std::find(active_.begin(), active_.end(), hid) == active_.end())
                    pending_.push_back({handler, arg, guest_fs, {hid, true}});
            active_ = std::move(next);
            ++tick_;
            if (!dispatching_ && !pending_.empty()) { dispatching_ = true; become_dispatcher = true; }
        }
        if (!become_dispatcher) return;
        for (;;) {
            Pending transition;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (pending_.empty()) { dispatching_ = false; return; }
                transition = pending_.front();
                pending_.pop_front();
            }
            // Never retain the runtime mutex across a call into guest code. A reentrant tick queues
            // behind transitions already published by this tick; this outer dispatcher drains it.
            ime_deliver(transition.handler, transition.arg, transition.ev, transition.guest_fs);
        }
    }

private:
    struct Pending { uint64_t handler = 0, arg = 0, guest_fs = 0; ImeKeyEvent ev{}; };
    std::vector<ImeScriptWindow> entries_;
    std::vector<uint16_t> active_;
    std::deque<Pending> pending_;
    uint64_t tick_ = 0;
    bool dispatching_ = false;
    std::mutex mutex_;
};

void ime_script_tick(uint64_t handler, uint64_t arg, uint64_t guest_fs) {
    static ImeScriptRuntime runtime;
    runtime.tick(handler, arg, guest_fs);
}

// Shared sceImeUpdate body: fire the autokey pulse (if enabled) then drain the queued keys, dispatching
// each on the caller's guest %fs (guest_fs == 0 => no swap, the Windows/macOS path). The queue mutex is
// released before every guest handler call (never held across guest code).
void ime_update_run(uint64_t handler, uint64_t guest_fs) {
    uint64_t arg = 0;
    {
        std::lock_guard<std::mutex> lk(g_ime_mx);
        if (g_ime_keyboard.open) arg = g_ime_keyboard.arg;
    }
    ime_autokey_tick(handler, arg, guest_fs);        // handler = the guest event handler
    ime_script_tick(handler, arg, guest_fs);
    for (;;) {
        ImeKeyEvent ev;
        { std::lock_guard<std::mutex> lk(g_ime_mx);
          if (g_ime_queue.empty()) break;
          ev = g_ime_queue.front(); g_ime_queue.pop_front(); }
        ime_deliver(handler, arg, ev, guest_fs);
    }
}
} // namespace

bool parse_ime_script_route(const std::string& text, std::vector<ImeScriptWindow>& out,
                            std::string* error) {
    std::string local_error;
    const bool ok = parse_ime_script_impl(text, out, local_error);
    if (error) *error = local_error;
    return ok;
}

// Frontend seam (declared in ime_input.hpp): push a keyboard key (USB HID usage id) into the IME
// queue; drained on the next sceImeUpdate. Thread-safe (called from the frontend's input thread).
// Mirrors pad_set_backend's role for libScePad.
void ime_push_key(uint16_t hid_usage, bool down) {
    std::lock_guard<std::mutex> lk(g_ime_mx);
    g_ime_queue.push_back({hid_usage, down});
}

HLE(s_ime_kbd_open) {
    svc_log("sceImeKeyboardOpen", a0,a1,a2,a3,a4,a5);
    ImeKeyboardParam param{};
    // Some older bring-up callers omitted the optional keyboard parameter entirely; retain their
    // successful no-input behavior. A supplied pointer, however, must be readable because its `arg`
    // is part of the callback ABI and remains live until KeyboardClose/re-open.
    if (a1 && !svc_copy_bytes(a1, &param, sizeof param)) return 0x80BC0031ull;
    std::lock_guard<std::mutex> lk(g_ime_mx);
    g_ime_keyboard = {true, (int32_t)a0, param.arg, param.handler};
    return 0;
}
HLE(s_ime_kbd_close) {
    svc_log("sceImeKeyboardClose", a0,a1,a2,a3,a4,a5);
    std::lock_guard<std::mutex> lk(g_ime_mx);
    if (g_ime_keyboard.open && g_ime_keyboard.user_id == (int32_t)a0) g_ime_keyboard = {};
    return 0;
}
// sceImeUpdate invokes the guest event handler (guest code), so it must restore the caller's guest %fs
// around each dispatch (see ime_deliver / #1286). On Linux/macOS the import stub swapped this thread to
// host %fs, so recover the caller's guest %fs from the import-stub frame via the entry-stack trampoline
// (as the AvPlayer callbacks do) and hand it to ime_update_run. PROSPER_ASM_TRAMPOLINE is POSIX-only, so
// Windows/MinGW (which never swaps hardware %fs at the import boundary) registers a plain handler that
// dispatches directly with guest_fs == 0. callback_guest_fs_from_entry_stack also returns 0 for any
// non-guest frame, so the plain-tail-jump macOS path is likewise an unchanged direct call.
#ifndef _WIN32
extern "C" uint64_t s_ime_update_c(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                   uint64_t a4, uint64_t a5, uint64_t entry_rsp) {
    svc_log("sceImeUpdate", a0,a1,a2,a3,a4,a5);
    ime_update_run(a0, prosper::callback_guest_fs_from_entry_stack(entry_rsp));
    return 0;
}
PROSPER_ASM_TRAMPOLINE(s_ime_update_entry, s_ime_update_c)
extern "C" void s_ime_update_entry();
#else
HLE(s_ime_update) {   // Windows/MinGW: no import-boundary %fs swap -> dispatch directly (guest_fs = 0)
    svc_log("sceImeUpdate", a0,a1,a2,a3,a4,a5);
    ime_update_run(a0, 0);
    return 0;
}
#endif
// sceImeKeyboardGetResourceId(userId, OrbisImeKeyboardResourceIdArray* out): report the user's keyboard
// resource ids. Headless -> none (all-zero). A registered PlatformUi (a windowed frontend with a host
// keyboard) reports its own ids, so a title enables keyboard input (#347).
HLE(s_ime_kbd_resid) {
    svc_log("sceImeKeyboardGetResourceId", a0,a1,a2,a3,a4,a5);
    if (!a1) return 0x80BC0031ull;   // ORBIS_IME_ERROR_INVALID_ADDRESS
    uint8_t* p = (uint8_t*)PW(a1);
    *(int32_t*)(p + 0) = (int32_t)a0;                         // user_id echoes the caller
    uint32_t ids[5] = {0, 0, 0, 0, 0};
    int n = 0;
    if (auto* ui = platform_ui()) n = ui->keyboardResourceIds((int32_t)a0, ids, 5);
    if (n < 0) n = 0; if (n > 5) n = 5;
    for (int i = 0; i < 5; i++) *(uint32_t*)(p + 4 + i * 4) = (i < n) ? ids[i] : 0;
    return 0;
}
// sceImeKeyboardGetInfo(resourceId, OrbisImeKeyboardInfo* info): connected device info when a frontend
// reports a keyboard, else a disconnected (no-device) info.
HLE(s_ime_kbd_info) {
    svc_log("sceImeKeyboardGetInfo", a0,a1,a2,a3,a4,a5);
    if (!a1) return 0x80BC0031ull;   // ORBIS_IME_ERROR_INVALID_ADDRESS
    memset(PW(a1), 0, 36);           // default: device=0 type=0 repeat=0 status=0(disconnected)
    uint8_t* p = (uint8_t*)PW(a1);
    *(int32_t*)(p + 0) = 1;          // user_id = default user (matches sceUserServiceGetInitialUser)
    bool connected = false;
    if (auto* ui = platform_ui()) { uint32_t ids[5]; connected = ui->keyboardResourceIds(1, ids, 5) > 0; }
    if (connected) {
        *(uint32_t*)(p + 4)  = 1;    // device  = 1 (a keyboard is present)
        *(uint32_t*)(p + 20) = 1;    // status  = 1 (connected)
    }
    return 0;
}

// --- app content ---
namespace {
constexpr uint64_t APP_CONTENT_ERROR_PARAMETER = 0x80D90002ull;
constexpr uint64_t APP_CONTENT_ERROR_BUSY = 0x80D90003ull;
constexpr uint64_t APP_CONTENT_ERROR_NOT_FOUND = 0x80D90005ull;
constexpr uint64_t APP_CONTENT_ERROR_DRM_NO_ENTITLEMENT = 0x80D90007ull;
constexpr uint64_t NP_ENTITLEMENT_ERROR_PARAMETER = 0x817D0002ull;
constexpr uint64_t NP_ENTITLEMENT_ERROR_NO_ENTITLEMENT = 0x817D0007ull;
// dlc_emu 0.3's NP list export compares listNum against 0x9c4 before touching either output.
constexpr uint32_t NP_ENTITLEMENT_ADDCONT_LIST_MAX = 2500;

// Direct PS5 guest evidence pins these boundaries. Sonic allocates exactly 0x1c bytes per
// NpEntitlementAccess list entry and passes a 20-byte label to AddcontMount; Crisis Core independently
// allocates the same stride and consumes exactly 16 key bytes. AppContent's inherited info is the
// label+download-status prefix. Sonic directly
// reads `download_status` at +0x18 and accepts INSTALLED=4. Package types PSAC=2 / PSAL=3 and the
// NP entitlement errno space are pinned to the dlc_emu producer at 43ae9aa (CONFIDENCE: HIGH).
struct AppContentAddcontInfo {
    char entitlement_label[20];
    uint32_t download_status;
};
struct NpAddcontEntitlementInfo {
    char entitlement_label[20];
    uint32_t package_type;
    uint32_t download_status;
};
static_assert(sizeof(AppContentAddcontInfo) == 24, "SceAppContentAddcontInfo ABI");
static_assert(offsetof(AppContentAddcontInfo, download_status) == 20,
              "SceAppContentAddcontInfo status offset");
static_assert(sizeof(NpAddcontEntitlementInfo) == 28,
              "SceNpEntitlementAccessAddcontEntitlementInfo ABI");
static_assert(offsetof(NpAddcontEntitlementInfo, package_type) == 20 &&
              offsetof(NpAddcontEntitlementInfo, download_status) == 24,
              "Np add-content info field offsets");

bool appcontent_valid_label_text(const char* text, size_t size) {
    if (!size || size > 16) return false;
    for (size_t i = 0; i < size; ++i) {
        const char ch = text[i];
        if (!((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
              (ch >= '0' && ch <= '9')))
            return false;
    }
    return true;
}

bool appcontent_read_label(uint64_t address, std::string& label) {
    if (!svc_ptrish(address)) return false;
    char raw[20];
    if (!svc_copy_bytes(address, raw, sizeof(raw))) return false;
    size_t size = 0;
    while (size < 17 && raw[size]) ++size;
    if (size == 17 || raw[17] || raw[18] || raw[19] ||
        !appcontent_valid_label_text(raw, size)) return false;
    label.assign(raw, size);
    return true;
}

AppContentAddcontInfo appcontent_info(const InstalledAddcontent& entry) {
    AppContentAddcontInfo info{};
    std::memcpy(info.entitlement_label, entry.entitlement_label.c_str(),
                entry.entitlement_label.size() + 1);
    info.download_status = entry.download_status;
    return info;
}

NpAddcontEntitlementInfo npent_info(const InstalledAddcontent& entry) {
    NpAddcontEntitlementInfo info{};
    std::memcpy(info.entitlement_label, entry.entitlement_label.c_str(),
                entry.entitlement_label.size() + 1);
    info.package_type = entry.package_type;
    info.download_status = entry.download_status;
    return info;
}

const InstalledAddcontent* find_addcontent(const AddcontentInventorySnapshot& inventory,
                                           uint32_t service_label,
                                           const std::string& entitlement_label) {
    for (const InstalledAddcontent& entry : inventory.entries) {
        if ((entry.service_label == -1 ||
             static_cast<uint32_t>(entry.service_label) == service_label) &&
            entry.entitlement_label == entitlement_label) return &entry;
    }
    return nullptr;
}

template <typename GuestInfo, typename MakeInfo>
uint64_t addcontent_info_list(uint64_t list_address, uint64_t list_num,
                              uint64_t hit_num_address, uint32_t service_label,
                              uint64_t parameter_error, uint64_t inventory_error,
                              uint64_t max_list_num, MakeInfo make_info) {
    if (list_num > max_list_num) return parameter_error;
    const AddcontentInventorySnapshot inventory = addcontent_inventory_snapshot();
    if (inventory.state == AddcontentInventoryState::Invalid) return inventory_error;

    std::vector<const InstalledAddcontent*> matching;
    for (const InstalledAddcontent& entry : inventory.entries)
        if (entry.service_label == -1 ||
            static_cast<uint32_t>(entry.service_label) == service_label)
            matching.push_back(&entry);

    const uint32_t capacity = static_cast<uint32_t>(list_num);
    const uint32_t total = static_cast<uint32_t>(matching.size());
    // Both producer list APIs define NULL-list or zero-capacity as a count query, where hitNum is
    // mandatory. On a real list call hitNum is optional.
    if (list_address == 0 || capacity == 0) {
        if (!svc_ptrish(hit_num_address) ||
            !svc_write_bytes(hit_num_address, &total, sizeof(total))) return parameter_error;
        return 0;
    }
    const uint32_t written = static_cast<uint32_t>(std::min<size_t>(capacity, matching.size()));
    if (!svc_ptrish(list_address) ||
        written > (UINT64_MAX - list_address) / sizeof(GuestInfo)) return parameter_error;
    for (uint32_t i = 0; i < written; ++i) {
        const GuestInfo info = make_info(*matching[i]);
        if (!svc_write_bytes(list_address + uint64_t{i} * sizeof(GuestInfo),
                             &info, sizeof(info))) return parameter_error;
    }
    // The dlc_emu producer/reference writes min(listNum,total) entries but reports the full total in
    // hitNum for both AppContent and NpEntitlementAccess. Pin: drakmor/dlc_emu@43ae9aa,
    // dlc_content.cpp:1941-1965 and 2357-2385. This lets a short caller discover and retry the count.
    if (hit_num_address && !svc_write_bytes(hit_num_address, &total, sizeof(total)))
        return parameter_error;
    return 0;
}
} // namespace

// sceAppContentAppParamGetInt(SceAppContentAppParamId paramId, int32_t* value)
//   paramId 0 = SKU_FLAG, 1..4 = USER_DEFINED_PARAM_1..4.
//
// Every one of these is a property the installed application declares about ITSELF in its own
// sce_sys/param.json, so all of them are answered from that local declaration and nothing else. The
// previous stub hardcoded SKU_FLAG to FULL and answered every user-defined param with 0, and it is
// the second half that gated GTA V's (PPSA04263) story mode: the title reads USER_DEFINED_PARAM_1 —
// which its own param.json declares as 9 — stores it, and tests bit 3 to decide that its installed
// content needs no online entitlement lookup. Answering 0 cleared that bit and sent it down the
// online path instead, where sceNpEntitlementAccessGetSkuFlag left the guest's own pre-seeded TRIAL
// value standing and the title offered story mode as a purchase. That chain was only half the story:
// the param read was itself skipped, because sceSysmoduleIsLoaded reported every module loaded and
// GTA V therefore never ran its own sceAppContentInitialize. Fixed separately in #2002 (the query
// now answers from prosper's real load history); both halves are needed, and a three-arm A/B on
// #1873 shows neither alone opens the gate. The declared values matter well
// beyond that gate: Crisis Core (PPSA07809) uses USER_DEFINED_PARAM_2 as a bounded content-variant
// index, and Little Nightmares II/III use USER_DEFINED_PARAM_1 as an index into their own
// add-content list.
//
// param.json omits a userDefinedParamN the publishing tool left at its default, so an absent key is a
// declared zero. With no parseable param.json there is no declaration at all and the query fails
// rather than inventing one — a value invented here is indistinguishable from a real declaration to
// every caller. CONFIDENCE: HIGH.
HLE(s_appcontent_int) {
    // Logged because this is the head of a causal chain that is otherwise invisible: a title stores
    // a user-defined param in its own state and acts on it much later, so "what did prosper answer
    // here" is the question a stalled content gate needs answered first.
    svc_log("sceAppContentAppParamGetInt", a0,a1,a2,a3,a4,a5);
    const int32_t param_id = (int32_t)a0;
    if (param_id < 0 || param_id > 4 || !svc_ptrish(a1)) return APP_CONTENT_ERROR_PARAMETER;
    const AppParamDeclaration decl = app_param_declaration();
    if (!decl.declared) return APP_CONTENT_ERROR_NOT_FOUND;
    int32_t value = 0;
    if (param_id == 0) {
        if (!decl.sku_flag) return APP_CONTENT_ERROR_NOT_FOUND;
        value = (int32_t)*decl.sku_flag;
    } else {
        value = decl.user_param[param_id - 1];
    }
    return svc_write_bytes(a1, &value, sizeof(value)) ? 0 : APP_CONTENT_ERROR_PARAMETER;
}

// sceSystemServiceParamGetInt(SceSystemServiceParamId paramId, int32_t* value): a0=paramId, a1=value.
// The system-settings a blanket-zero stub gives are mostly harmless, EXCEPT the LANGUAGE (paramId 1):
// value 0 = SCE_SYSTEM_PARAM_LANG_JAPANESE, so games localise their UI/text to Japanese. Default to
// US English (SCE_SYSTEM_PARAM_LANG_ENGLISH_US = 1) instead. Configurable via PROSPER_SYS_LANG, which
// takes the Sony SCE_SYSTEM_PARAM_LANG_* enum (0=ja, 1=en-US, 2=fr, 4=de, 5=it, 9=ko, 18=en-GB, …).
// Date/time-format params (2/3) default to the US convention: date enum 2 = MM/DD/YYYY and
// time enum 0 = 12-hour. Date enum 0 is YYYYMMDD, despite the old comment claiming it was US.
HLE(s_syss_param_int) {
    int32_t paramId = (int32_t)a0;           // a0 = paramId, a1 = int32_t* value out (matches s_appcontent_int)
    int32_t val = 0;
    if (paramId == 1) {                      // SCE_SYSTEM_SERVICE_PARAM_ID_LANG
        val = 1;                             // SCE_SYSTEM_PARAM_LANG_ENGLISH_US
        if (const char* e = getenv("PROSPER_SYS_LANG")) val = (int32_t)strtol(e, nullptr, 0);
    } else if (paramId == 2) {               // SCE_SYSTEM_SERVICE_PARAM_ID_DATE_FORMAT
        val = 2;                             // SCE_SYSTEM_PARAM_DATE_FORMAT_MMDDYYYY
    } else if (paramId == 1000) {            // SCE_SYSTEM_SERVICE_PARAM_ID_ENTER_BUTTON_ASSIGN
        // Cross = confirm (Western default). Previously fell through to val=0 = Circle, which inverts
        // ✕/○ confirm/cancel and on-screen button prompts relative to the en-US locale we present.
        val = 1;                             // Cross (shadPS4 default; Circle=0)
        if (const char* e = getenv("PROSPER_ENTER_BUTTON")) val = (int32_t)strtol(e, nullptr, 0);
    }
    if (a1) *(int32_t*)PW(a1) = val;
    return 0;
}

// sceSystemServiceGetStatus(SceSystemServiceStatus* status) — the out-struct is ARG 0 (single-arg
// call). This was aliased to s_appcontent_int, which returned success while writing 4 bytes through
// a1 — whatever stale value the caller left in RSI — and left the real status struct uninitialized.
// Layout cross-checked against Kyty LibSystemService.cpp:83: int32 eventNum @0; bool
// isSystemUiOverlaid @4, isInBackgroundExecution @5, isCpuMode7CpuNormal @6 (defaults TRUE),
// isGameLiveStreamingOnAir @7, isOutOfVrPlayArea @8; 12 bytes with tail padding.
namespace {
// ErrorDialog is system-owned UI. Platform glue uses SystemService status' background -> foreground
// transition to resume requests after it closes, independently of the CommonDialog FINISHED value.
// A real backend may stay active across many status samples; the headless backend dismisses during
// Open and must still expose one observable enter/return pair instead of collapsing both edges into
// a permanent foreground value. Oregon Trail #1606 positive-controlled this exact contract twice:
// the account-completion writer ran only after both SystemService samples. No title address or
// title identity participates in this state machine.
enum ErrorDialogBackgroundPhase : unsigned {
    ErrorDialogIdle = 0,
    ErrorDialogHeadlessEnter,
    ErrorDialogHeadlessReturn,
    ErrorDialogBackedEnter,
    ErrorDialogBackedActive,
    ErrorDialogBackedReturn,
};
std::atomic<unsigned> g_error_dialog_background_phase{ErrorDialogIdle};

void begin_error_dialog_background(bool backed) {
    g_error_dialog_background_phase.store(
        backed ? ErrorDialogBackedEnter : ErrorDialogHeadlessEnter, std::memory_order_release);
}

void finish_error_dialog_background() {
    unsigned phase = g_error_dialog_background_phase.load(std::memory_order_acquire);
    for (;;) {
        unsigned next = phase;
        if (phase == ErrorDialogBackedEnter) next = ErrorDialogHeadlessEnter;
        else if (phase == ErrorDialogBackedActive) next = ErrorDialogBackedReturn;
        else return;
        if (g_error_dialog_background_phase.compare_exchange_weak(
                phase, next, std::memory_order_acq_rel, std::memory_order_acquire)) return;
    }
}

uint8_t sample_error_dialog_background() {
    unsigned phase = g_error_dialog_background_phase.load(std::memory_order_acquire);
    for (;;) {
        unsigned next = phase;
        uint8_t background = 0;
        const char* edge = nullptr;
        switch (phase) {
        case ErrorDialogHeadlessEnter:
            next = ErrorDialogHeadlessReturn; background = 1; edge = "enter"; break;
        case ErrorDialogHeadlessReturn:
            next = ErrorDialogIdle; background = 0; edge = "return"; break;
        case ErrorDialogBackedEnter:
            next = ErrorDialogBackedActive; background = 1; edge = "enter"; break;
        case ErrorDialogBackedActive:
            return 1;
        case ErrorDialogBackedReturn:
            next = ErrorDialogIdle; background = 0; edge = "return"; break;
        default:
            return 0;
        }
        if (!g_error_dialog_background_phase.compare_exchange_weak(
                phase, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
            continue;
        }
        fprintf(stderr, "[svc] ErrorDialog system lifecycle background=%u phase=%s\n",
                (unsigned)background, edge);
        return background;
    }
}
}

HLE(s_syss_getstatus) {
    auto* st = (uint8_t*)PW(a0);
    if (!st) return 0x80A10003ull;   // SYSTEM_SERVICE_ERROR_PARAMETER (Kyty Errno.h:382)
    memset(st, 0, 12);
    // event_num is the first field. Titles use it to decide how many times to call ReceiveEvent;
    // advertising an activity only in ReceiveEvent leaves that function unreachable.
    if (g_gameintent_initialized.load(std::memory_order_acquire) &&
        !g_gameintent_event_delivered.load(std::memory_order_acquire) &&
        gameintent_activity_id()) {
        *(int32_t*)st = 1;
    }
    st[5] = sample_error_dialog_background();
    st[6] = 1;                       // isCpuMode7CpuNormal = true
    return 0;
}

// sceSystemServiceGetDisplaySafeAreaInfo(SceSystemServiceDisplaySafeAreaInfo* info) — single-arg
// call, out-struct is ARG 0: { float ratio; uint8_t reserved[128]; } (shadPS4 systemservice.h).
// The default unimplemented stub returned 0 (SUCCESS) but left `ratio` uninitialized (typically 0.0
// from a fresh guest heap block). A safe-area ratio of 0 makes UE4's PS5 viewport code compute a
// DEGENERATE (zero-area) title-safe rect: the game scales its render/UI viewport by the ratio, and a
// zero ratio collapses the visible region — nothing to rasterize, so the RHI submits setup/compute
// but never geometry. Real hardware always reports ratio=1.0 for a display with overscan disabled
// (the modern default); shadPS4 hard-codes 1.0f. Fill ratio=1.0 and zero the reserved tail.
// CONFIDENCE: MED — struct + 1.0f contract confirmed against shadPS4; that a 0.0 ratio is what
// gates DOLL's scene draws is a hypothesis under test, but returning success with an unfilled
// out-struct is a bug regardless (same class as GetStatus / ParamGetString above).
HLE(s_syss_safearea) {
    auto* info = (uint8_t*)PW(a0);
    if (!info) return 0x80A10003ull;   // SYSTEM_SERVICE_ERROR_PARAMETER
    memset(info, 0, 0x84);             // sizeof {float + uint8_t[128]} = 132
    *(float*)info = 1.0f;              // ratio = full display, no overscan inset
    return 0;
}

// sceSystemServiceReportAbnormalTermination(cause) — the guest reporting a crash to the system.
//
// Registered so the line is READABLE. Unregistered, the dispatcher default fired as one opaque
// `unimplemented: libSceSystemService::3s8cHiCBKBE -> returning 0` among a dozen others, and
// resolving it needed a hand lookup against the PS5 3.20 stub table:
//   libSceSystemService.c:3976  sprx_dlsym(__handle, "3s8cHiCBKBE",
//                                          &__ptr_sceSystemServiceReportAbnormalTermination)
//
// IT DOES NOT MEAN THE TITLE IS FAILING NOW, and an earlier revision of this handler assumed it did.
// Independent review of #3120 opened five archived Tactics Ogre (PPSA03839) runs: the call is in ALL
// of them, always at line 8, during boot before the first frame -- including the run that then
// rendered 40,936 frames over 470 s and reached the tutorial battle, and the runs that stalled at
// ~frame 993. `prosper_on_unimpl` logs only on FIRST invocation, so that is the title's one call.
// It therefore has zero discriminating power: it is something this title does unconditionally at
// startup. One reading consistent with all five logs, offered as hypothesis and not finding: the API
// may report that the PREVIOUS session ended abnormally, and these runs are killed by `timeout`.
//
// So the default is to LOG AND CONTINUE. Stopping is opt-in via PROSPER_ABNORMAL_TERMINATION_STOP.
// Stopping by default would have been a live regression rather than a theoretical one: 23 of the 56
// local dumps import this NID, four of them rung-6 guarded (The Messenger, Blasphemous 2, Alex Kidd,
// Blue Prince), and `prosper_stop_requested()` is polled only by prosper-app -- so the snapshot
// matrix, which drives boot_trace and screenshot, cannot see the breakage at all. It would have
// failed only in a human's hands.
//
// SCOPE, stated because the opt-in is not uniform across frontends. prosper-app winds its run-loop
// down (main.cpp) and run_entry() never observes the stop, so the guest thread is not torn down. On
// screenshot and boot_trace nothing polls it, so the flag does not stop those runs -- and note it is
// not inert there either: prosper_wait_while_paused() returns false permanently once stop latches,
// which mutes the SDL3 audio sink in a frontend that keeps running.
//
// The return stays 0, identical to the dispatcher default, so no guest observes a different answer.
// The argument is logged raw; no layout for it is confirmed against any primary source.
// CONFIDENCE: HIGH on the NID identity. LOW on the argument, and on what a call implies about the
// title's state.
HLE(s_syss_report_abnormal_termination) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    // Read per call rather than caching in a static: this fires at most once in a run, so there is
    // nothing to optimise, and a cached sample would make the opt-in untestable in-process.
    const bool stop_run = getenv("PROSPER_ABNORMAL_TERMINATION_STOP") != nullptr;
    fprintf(stderr,
            "[prosper] GUEST REPORTED ABNORMAL TERMINATION: "
            "sceSystemServiceReportAbnormalTermination(0x%llx). The title has decided it is "
            "crashing -- this is the guest's own verdict, not prosper's. %s\n",
            (unsigned long long)a0,
            stop_run ? "PROSPER_ABNORMAL_TERMINATION_STOP is set; stopping the run."
                     : "Continuing: this title may call it at boot and run fine. Set "
                       "PROSPER_ABNORMAL_TERMINATION_STOP=1 to stop on it.");
    fflush(stderr);
    if (stop_run) prosper::prosper_request_stop();
    return 0;
}

// sceSystemServiceGetHdrToneMapLuminance(out). Kyty models the output as three floats in this order:
// max-full-frame, max, and min tone-map luminance. The imported function previously fell through to
// success-without-output, so display setup consumed poisoned values. Use Kyty's 80/1000/0 values as
// a deterministic fallback; this does not advertise or implement an HDR presentation path.
// CONFIDENCE: MED on the Kyty-derived 12-byte layout/order; LOW on the real-hardware values.
struct SysHdrToneMapLuminance {
    float max_full_frame;
    float max;
    float min;
};
static_assert(sizeof(SysHdrToneMapLuminance) == 12, "Kyty-modeled HDR tone-map luminance layout");
HLE(s_syss_hdr_luminance) {
    auto* luminance = (SysHdrToneMapLuminance*)PW(a0);
    if (!luminance) return 0x80A10003ull;   // SYSTEM_SERVICE_ERROR_PARAMETER
    luminance->max_full_frame = 80.0f;
    luminance->max = 1000.0f;
    luminance->min = 0.0f;
    return 0;
}

// sceAppContentTemporaryDataMount2(option, SceAppContentMountPoint* mp) — mount the app's temp-data
// area and write its guest path into mp. SceAppContentMountPoint = char data[16] (shadPS4
// app_content.h; PS4/PS5 identical). shadPS4 writes exactly "/temp0\0" and returns 0. Our previous
// behavior (unimplemented stub -> return 0 = SUCCESS with mp untouched) made the game treat 16 bytes
// of uninitialized memory as its temp-data mount path and build file paths from it. We write EXACTLY
// 7 bytes ("/temp0" + NUL) — never the full 16, never more (cf. the f_fstat oversized-write lesson).
// hle_file.cpp translates the /temp0 prefix to a host-backed directory so subsequent I/O works.
HLE(s_appcontent_tmpmount2) {
    char* mp = (char*)PW(a1);
    if (!mp) return 0x80D90002ull;              // SCE_APP_CONTENT_ERROR_PARAMETER
    memcpy(mp, "/temp0", 7);
    return 0;
}
// sceAppContentTemporaryDataGetAvailableSpaceKb(mp, uint64_t* kb): temp0 is a 1 GiB scratch area on
// PS5; report it all free (shadPS4 does the same). The stub's return-0-only left *kb garbage — a
// value games use to size caches/allocations.
HLE(s_appcontent_tmpspace) { if (a1) *(uint64_t*)PW(a1) = 1048576ull; return 0; }

// sceAppContentGetAddcontInfoList(serviceLabel, Info* list, u32 listNum, u32* hitNum). A zero-capacity
// call is the count query; a shorter caller buffer receives only its capacity while hitNum reports the
// full matching count. With no manifest the retail no-DLC answer remains SUCCESS/hitNum=0. A malformed host
// inventory fails rather than masquerading as a valid empty install.
HLE(s_appcontent_addcont_list) {
    if (a0 > UINT32_MAX) return APP_CONTENT_ERROR_PARAMETER;
    return addcontent_info_list<AppContentAddcontInfo>(
        a1, a2, a3, static_cast<uint32_t>(a0), APP_CONTENT_ERROR_PARAMETER,
        APP_CONTENT_ERROR_DRM_NO_ENTITLEMENT, UINT32_MAX, appcontent_info);
}

HLE(s_appcontent_addcont_info) {
    std::string label;
    if (a0 > UINT32_MAX || !appcontent_read_label(a1, label) || !svc_ptrish(a2))
        return APP_CONTENT_ERROR_PARAMETER;
    const AddcontentInventorySnapshot inventory = addcontent_inventory_snapshot();
    if (inventory.state == AddcontentInventoryState::Invalid)
        return APP_CONTENT_ERROR_DRM_NO_ENTITLEMENT;
    const InstalledAddcontent* entry = find_addcontent(inventory, static_cast<uint32_t>(a0), label);
    if (!entry) return APP_CONTENT_ERROR_DRM_NO_ENTITLEMENT;
    const AppContentAddcontInfo info = appcontent_info(*entry);
    return svc_write_bytes(a2, &info, sizeof(info)) ? 0 : APP_CONTENT_ERROR_PARAMETER;
}

HLE(s_appcontent_entitlement_key) {
    std::string label;
    if (a0 > UINT32_MAX || !appcontent_read_label(a1, label) || !svc_ptrish(a2))
        return APP_CONTENT_ERROR_PARAMETER;
    const AddcontentInventorySnapshot inventory = addcontent_inventory_snapshot();
    const InstalledAddcontent* entry = inventory.state == AddcontentInventoryState::Ready
        ? find_addcontent(inventory, static_cast<uint32_t>(a0), label) : nullptr;
    if (!entry) return APP_CONTENT_ERROR_DRM_NO_ENTITLEMENT;
    return svc_write_bytes(a2, entry->entitlement_key.data(), entry->entitlement_key.size())
        ? 0 : APP_CONTENT_ERROR_PARAMETER;
}

HLE(s_appcontent_addcont_mount) {
    std::string label;
    if (a0 > UINT32_MAX || !appcontent_read_label(a1, label) || !svc_ptrish(a2))
        return APP_CONTENT_ERROR_PARAMETER;
    switch (addcontent_mount(static_cast<uint32_t>(a0), label, a2, svc_write_bytes)) {
    case AddcontentMountResult::NotFound: return APP_CONTENT_ERROR_NOT_FOUND;
    case AddcontentMountResult::Busy: return APP_CONTENT_ERROR_BUSY;
    case AddcontentMountResult::OutputError: return APP_CONTENT_ERROR_PARAMETER;
    case AddcontentMountResult::Mounted: return 0;
    }
    return APP_CONTENT_ERROR_NOT_FOUND;
}

// sceAppContentAddcontUnmount(const SceAppContentMountPoint* mountPoint) — release a claim taken by
// sceAppContentAddcontMount, identified by the 16-byte mount-point object that call handed back.
//
// Unregistered, this reached the dispatcher's `return 0`, which for this contract is SCE_OK: the
// guest was told the unmount succeeded while `entry.mounted` stayed set, so the NEXT mount of the
// same add-content returned BUSY and kept doing so forever. The title has no way to recover —
// from its point of view it released the entry — and locally-present content becomes unreachable.
// It presents as "the game refuses to load DLC it loaded a minute ago", arbitrarily far from here.
//
// This grants nothing. It clears a flag prosper itself set, and an unrecognised mount point frees
// nothing and returns NOT_FOUND. No ownership or entitlement answer is reachable from this path.
//
// The argument is the mount point rather than the entitlement label, mirroring the Mount pair
// (Mount takes serviceLabel + entitlementLabel and WRITES the mount point; Unmount takes the mount
// point back). CONFIDENCE: MED on the argument — it is the inherited PS4 AppContent shape and no
// local dump exercises a mount/unmount cycle, so it is not confirmed against a live guest. The
// residual risk is bounded by direction: if a title passes something else, the lookup fails and the
// guest gets NOT_FOUND, which is fail-visible and still strictly better than the silent success it
// gets today. What would settle it: a boot of a title whose dump declares `dlc_emu.ini`, with the
// argument registers traced at this NID.
HLE(s_appcontent_addcont_unmount) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    if (!svc_ptrish(a0)) return APP_CONTENT_ERROR_PARAMETER;
    // SceAppContentMountPoint is a 16-byte NUL-padded char array. Read it fault-contained: a bad
    // pointer must be a parameter error, not a fault, and not a success.
    char raw[16];
    if (!svc_copy_bytes(a0, raw, sizeof(raw))) return APP_CONTENT_ERROR_PARAMETER;
    const size_t len = strnlen(raw, sizeof(raw));
    switch (addcontent_unmount(std::string_view(raw, len))) {
    case AddcontentUnmountResult::Unmounted:  return 0;
    case AddcontentUnmountResult::NotMounted: return APP_CONTENT_ERROR_NOT_FOUND;
    }
    return APP_CONTENT_ERROR_NOT_FOUND;
}

// sceSystemServiceParamGetString(paramId, char* buf, size_t bufSize): fetch a system string parameter
// (e.g. the console/user nickname). The default unimplemented stub returned 0 (SUCCESS) but never wrote
// the buffer, so the game read whatever uninitialized bytes were there as a "valid" string and derefed
// into it (null-deref crash in managed code during scene load). Match this file's policy: write a valid
// NUL-terminated string and report success. CONFIDENCE: MED — signature (paramId, buf, size) is the
// documented Sony ABI; an empty string is a safe, defined default when we have no real system value.
HLE(s_param_string)   { if (a1 && a2) ((char*)PW(a1))[0] = '\0'; return 0; }

// Message-dialog LIFECYCLE (#144). Status enum: NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3.
// The old handler returned FINISHED(3) UNCONDITIONALLY — including before any Open, where the real
// API reports NONE/INITIALIZED — so a guest polling GetStatus as a guard saw "dialog already done"
// at the wrong stage and skipped/duplicated its dialog logic. We track the real transitions:
// Initialize -> INITIALIZED, Open -> auto-dismiss to FINISHED (headless: no interactive UI, so the
// game's "wait until dismissed" loop still exits immediately), Close/Terminate -> back to NONE.
// GetResult -> zeroed struct = OK/no button pressed.
// A registered PlatformUi gets first refusal at Open; if it takes the dialog, status/result/close
// route there (real message box). Otherwise the core auto-dismisses headlessly. `g_msgdialog_backed`
// records which path Open chose. See platform_ui.hpp (#347).
namespace { std::atomic<int> g_msgdialog_status{0 /*NONE*/}; std::atomic<int> g_msgdialog_backed{0}; }
HLE(s_dialog_initialize) { g_msgdialog_backed.store(0); g_msgdialog_status.store(1 /*INITIALIZED*/); return 0; }
HLE(s_dialog_open) {
    if (auto* ui = platform_ui(); ui && ui->msgDialogOpen(a0)) { g_msgdialog_backed.store(1); return 0; }
    g_msgdialog_backed.store(0);
    g_msgdialog_status.store(3 /*FINISHED (auto-dismiss)*/);
    return 0;
}
HLE(s_dialog_close)      { if (g_msgdialog_backed.exchange(0)) { if (auto* ui = platform_ui()) ui->msgDialogClose(); } g_msgdialog_status.store(0 /*NONE*/); return 0; }
HLE(s_dialog_terminate)  { if (g_msgdialog_backed.exchange(0)) { if (auto* ui = platform_ui()) ui->msgDialogClose(); } g_msgdialog_status.store(0 /*NONE*/); return 0; }
HLE(s_dialog_status) {
    if (g_msgdialog_backed.load()) { if (auto* ui = platform_ui()) return (uint64_t)(unsigned)ui->msgDialogStatus(); }
    return (uint64_t)(unsigned)g_msgdialog_status.load();
}
// SceMsgDialogResult = { u32 mode; u32 result; u32 buttonId; char reserved[32] } = 0x2C bytes
// (shadPS4 msgdialog_ui.h DialogResult). Was memset(0x30) — 4 bytes PAST the caller's struct,
// exactly the f_fstat oversized-write class this file warns about.
HLE(s_dialog_result) {
    if (g_msgdialog_backed.load()) { if (auto* ui = platform_ui()) { (void)ui->msgDialogResult(a0); return 0; } }
    if (a0) memset(PW(a0), 0, 0x2C);
    return 0;
}

// --- libSceSaveDataDialog ---------------------------------------------------------------
// Uses the common-dialog status enum: NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3. A generic
// success stub is not safe for UpdateStatus: Dead Cells polled it 1,135 times in 16 seconds because
// zero means NONE, leaving its level transition in a permanent wait (#768). The headless policy
// matches MsgDialog above: Open auto-dismisses to FINISHED so the guest can consume a neutral result.
//
// PS4/PS5 SDK layout used below (also used by shadPS4): param.mode is u32 at +0x34 and param.userData
// is a pointer at +0x70. Result begins with mode/result/buttonId/pad followed by three pointers. The
// dirName/param pointer slots are caller-owned output destinations, not fields for the service to
// replace. Write only mode/result/buttonId/userData and preserve every padding, pointer, and reserved
// byte. A PlatformUi that accepts Open remains the intended owner, but every callback obtains a
// registry lease first: unregistering the backend safely abandons the dialog instead of dereferencing
// a stale non-owning pointer or rerouting the dialog to a replacement backend.
namespace {
    std::atomic<int> g_savedatadialog_status{0 /*NONE*/};
    std::atomic<uint32_t> g_savedatadialog_mode{0 /*INVALID*/};
    std::atomic<uint64_t> g_savedatadialog_user_data{0};
    std::atomic<PlatformUi*> g_savedatadialog_ui{nullptr};

    void savedlg_close_owner() {
        PlatformUi* expected = g_savedatadialog_ui.exchange(nullptr);
        if (!expected) return;
        auto ui = platform_ui_lease(expected);
        if (ui) ui->saveDataDialogClose();
    }

    PlatformUiLease savedlg_owner_lease(PlatformUi*& expected) {
        expected = g_savedatadialog_ui.load(std::memory_order_acquire);
        if (!expected) return {};
        return platform_ui_lease(expected);
    }

    void savedlg_abandon_owner(PlatformUi* expected) {
        if (expected && g_savedatadialog_ui.compare_exchange_strong(expected, nullptr))
            g_savedatadialog_status.store(3 /*FINISHED: safe headless fallback*/);
    }
}
HLE(s_savedlg_initialize) {
    savedlg_close_owner();
    g_savedatadialog_mode.store(0);
    g_savedatadialog_user_data.store(0);
    g_savedatadialog_status.store(1 /*INITIALIZED*/);
    return 0;
}
HLE(s_savedlg_open) {
    savedlg_close_owner();
    if (a0 && a0 <= UINT64_MAX - 0x78) {
        uint32_t mode = 0;
        uint64_t user_data = 0;
        (void)svc_copy_bytes(a0 + 0x34, &mode, sizeof mode);
        (void)svc_copy_bytes(a0 + 0x70, &user_data, sizeof user_data);
        g_savedatadialog_mode.store(mode);
        g_savedatadialog_user_data.store(user_data);
    } else {
        g_savedatadialog_mode.store(0);
        g_savedatadialog_user_data.store(0);
    }
    auto ui = platform_ui_lease();
    if (ui && ui->saveDataDialogOpen(a0)) {
        g_savedatadialog_ui.store(ui.get(), std::memory_order_release);
        return 0;
    }
    g_savedatadialog_status.store(3 /*FINISHED (auto-dismiss)*/);
    return 0;
}
HLE(s_savedlg_status) {
    PlatformUi* expected = nullptr;
    auto ui = savedlg_owner_lease(expected);
    if (expected && ui)
        return (uint64_t)(unsigned)ui->saveDataDialogStatus();
    savedlg_abandon_owner(expected);
    return (uint64_t)(unsigned)g_savedatadialog_status.load();
}
HLE(s_savedlg_result) {
    PlatformUi* expected = nullptr;
    auto ui = savedlg_owner_lease(expected);
    if (expected && ui) {
        (void)ui->saveDataDialogResult(a0);
        return 0;
    }
    savedlg_abandon_owner(expected);
    if (a0 && a0 <= UINT64_MAX - 0x20) {
        const uint32_t mode = g_savedatadialog_mode.load();
        const uint32_t ok = 0 /*CommonDialogResult::OK*/;
        const uint32_t invalid = 0 /*ButtonId::INVALID*/;
        const uint64_t user_data = g_savedatadialog_user_data.load();
        (void)svc_write_bytes(a0 + 0x00, &mode, sizeof mode);
        (void)svc_write_bytes(a0 + 0x04, &ok, sizeof ok);
        (void)svc_write_bytes(a0 + 0x08, &invalid, sizeof invalid);
        (void)svc_write_bytes(a0 + 0x20, &user_data, sizeof user_data);
    }
    return 0;
}
HLE(s_savedlg_close) {
    savedlg_close_owner();
    g_savedatadialog_status.store(0 /*NONE*/);
    return 0;
}
HLE(s_savedlg_terminate) {
    savedlg_close_owner();
    g_savedatadialog_status.store(0 /*NONE*/);
    g_savedatadialog_mode.store(0);
    g_savedatadialog_user_data.store(0);
    return 0;
}
HLE(s_savedlg_ready) { return 1; }
HLE(s_savedlg_progress_inc) {
    PlatformUi* expected = nullptr;
    auto ui = savedlg_owner_lease(expected);
    if (expected && ui)
        ui->saveDataDialogProgressBarInc((uint32_t)a0, (uint32_t)a1);
    else
        savedlg_abandon_owner(expected);
    return 0;
}
HLE(s_savedlg_progress_set) {
    PlatformUi* expected = nullptr;
    auto ui = savedlg_owner_lease(expected);
    if (expected && ui)
        ui->saveDataDialogProgressBarSetValue((uint32_t)a0, (uint32_t)a1);
    else
        savedlg_abandon_owner(expected);
    return 0;
}

// --- libSceImeDialog (on-screen text-entry dialog) (#191). We have no keyboard UI, so the dialog
// auto-completes: Init -> FINISHED immediately, so the game's "poll GetStatus until Finished" loop
// exits at once instead of hanging on a dialog that never appears; GetResult reports endStatus =
// OK/ENTER with the (unchanged/empty) input buffer; Term/Abort return to NONE.
// IMPORTANT: sceImeDialogGetStatus returns the IME-dialog's OWN enum (OrbisImeDialogStatus: NONE=0,
// RUNNING=1, FINISHED=2) — NOT the 4-value SceCommonDialogStatus that MsgDialog/ErrorDialog use
// (…RUNNING=2, FINISHED=3). Verified in shadPS4 ime_dialog.cpp. Returning 3 here made a game's poll
// loop never see Finished(2). We write only the 4-byte endStatus at GetResult offset 0 (the field the
// game branches on), never more, so a wrong tail-field guess can't corrupt the caller's struct.
// A registered PlatformUi (the app frontend) gets first refusal on the dialog: if it takes it
// (imeDialogOpen -> true), status/result/close route there so a real text field is shown; otherwise
// the core auto-completes headlessly (below). `g_imedialog_backed` records which path Init chose so a
// later poll/result/term goes to the same place. See platform_ui.hpp (#347).
namespace { std::atomic<int> g_imedialog_status{0 /*NONE*/}; std::atomic<int> g_imedialog_backed{0}; }
HLE(s_imedlg_init) {
    if (auto* ui = platform_ui(); ui && ui->imeDialogOpen(a0, a1)) { g_imedialog_backed.store(1); return 0; }
    g_imedialog_backed.store(0);
    g_imedialog_status.store(2 /*OrbisImeDialogStatus::Finished — auto-complete, no keyboard UI*/);
    return 0;
}
HLE(s_imedlg_status) {
    if (g_imedialog_backed.load()) { if (auto* ui = platform_ui()) return (uint64_t)(unsigned)ui->imeDialogStatus(); }
    return (uint64_t)(unsigned)g_imedialog_status.load();
}
HLE(s_imedlg_result) {
    if (g_imedialog_backed.load()) { if (auto* ui = platform_ui()) { (void)ui->imeDialogResult(a0); return 0; } }
    if (a0) *(int32_t*)PW(a0) = 0 /*SCE_IME_DIALOG_END_STATUS_OK*/;
    return 0;
}
HLE(s_imedlg_term)  { if (g_imedialog_backed.exchange(0)) { if (auto* ui = platform_ui()) ui->imeDialogClose(); } g_imedialog_status.store(0 /*NONE*/); return 0; }
HLE(s_imedlg_abort) { if (g_imedialog_backed.exchange(0)) { if (auto* ui = platform_ui()) ui->imeDialogClose(); } g_imedialog_status.store(0 /*NONE*/); return 0; }

// --- libSceNpTrophy2 (PS5 trophy system) — the DOLL 34.6 GB OOM (issue #213 diagnosis). ---------
// The guest's trophy bring-up (eboot+0xdbcb43..0xdbcc2e, gdb-captured live) calls
// sceNpTrophy2GetGameInfo(ctx, handle, out*, 0) — NID 4IzqhhUQ3nk named via nid_hash brute force —
// then grows TWO arrays from the out-struct's counts (u32 at out+0x4: 32-byte entries; a second
// 0x520-byte-entry array) and calls a sibling (y3zHpdZO6ME, unnamed) to fill them. The generic
// unimplemented stub returned 0 = SUCCESS with the out-struct UNWRITTEN, so the engine consumed
// heap garbage as a trophy count: gdb-captured count 0x408bd000 -> a 34,644,492,288-byte TArray
// grow ("Ran out of memory allocating 34644492288 bytes") — and when the garbage happened to be
// allocatable-huge instead, the minutes-long zero-fill starved the RenderThread until UE's
// "GameThread timed out waiting for RenderThread after 120.00 secs" watchdog killed the boot.
// Without a trophy backend the honest answer is FAILURE: a negative return takes the caller's
// clean invalid path (eboot+0xdbd239/0xdbd242: mark the trophy config unavailable, continue) —
// exactly the state a real console reports with no signed-in user. Only the SIGN of the return is
// consumed by this caller; the exact NpTrophy2 error space is unverified (no Kyty/shadPS4/stub
// reference), so the value is chosen inside the documented SCE_NP_TROPHY (0x8055xxxx) range.
// CONFIDENCE: HIGH that failure beats success+garbage; LOW on the specific error constant.
HLE(s_nptrophy2_unavailable) { return 0x80551500ull; }

// ===== Issue #232: the Sony services DOLL's level-load flow polls (PlayGo / SaveData / =========
// ===== NpTrophy2 lifecycle / Share). All NID<->name pairs verified against the PS5 3.20 ========
// ===== library stub tables (PS5-3.20_Libs/libSce{PlayGo,SaveData.native,NpTrophy2,Share}.c). ===
//
// DOLL's game-side workers (DollLevelPreloader / SaveLoadUpdate / DLCDataUpdate / ShareUpdate)
// gate the per-frame boot-flow state machine at eboot+0x5044740 on these services answering.
// Bare unimpl->0 stubs returned SUCCESS with every out-param unfilled (the recurring
// success+garbage-out bug class), so e.g. scePlayGoOpen "succeeded" without ever writing the
// handle the game then queries loci with.

// --- libScePlayGo: report ALL content installed and locus-local. --------------------------------
// PS4-inherited API (identical exported names on PS5 3.20); shapes cross-checked against shadPS4
// playgo.cpp + playgo_types.h and Kyty. A disc/fully-installed title is exactly this state on real
// hardware, so "everything present" is the truthful answer for our complete dump.
// Error space 0x80B2000x (shadPS4 playgo_types.h). CONFIDENCE: HIGH (two agreeing PS4 references,
// PS4-inherited surface).
static constexpr uint64_t PLAYGO_ERR_BAD_POINTER = 0x80B2000Aull;
static constexpr uint64_t PLAYGO_ERR_BAD_SIZE    = 0x80B2000Bull;
static constexpr uint64_t PLAYGO_ERR_BAD_CHUNK_ID = 0x80B2000Cull;

// Most PS5 dumps do not include sce_sys/playgo-chunk.dat, but UE IoStore preserves the same chunk
// ids in paired pakchunk<N>-*.utoc/.ucas files. Keep the PlayGo answers internally consistent with
// the content that is actually present: GetChunkId enumerates these ids and GetLocus/GetProgress
// reject everything else.
// Returning LOCAL_FAST for every possible u16 made DOLL probe through its 1000-id safety cap and left
// its optional-content state unresolved even though pakchunk1 had mounted successfully (#1373).
static std::vector<uint16_t> discover_playgo_chunks() {
    namespace fs = std::filesystem;
    std::vector<uint16_t> chunks;
    std::vector<fs::path> pak_dirs;
    bool saw_iostore_index = false;
    std::error_code ec;
    const fs::path app0(resolve_guest_path("/app0"));

    auto lower = [](std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return value;
    };
    auto child_named = [&](const fs::path& parent, const char* wanted) -> fs::path {
        const std::string wanted_lower = lower(wanted);
        std::error_code iter_ec;
        for (fs::directory_iterator it(parent, iter_ec), end; !iter_ec && it != end;
             it.increment(iter_ec)) {
            if (it->is_directory(iter_ec) && lower(it->path().filename().string()) == wanted_lower)
                return it->path();
            iter_ec.clear();
        }
        return {};
    };
    auto add_project_paks = [&](const fs::path& project) {
        const fs::path content = child_named(project, "content");
        if (content.empty()) return;
        const fs::path paks = child_named(content, "paks");
        if (!paks.empty()) pak_dirs.push_back(paks);
    };

    add_project_paks(app0); // /app0/Content/Paks
    for (fs::directory_iterator it(app0, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->is_directory(ec)) add_project_paks(it->path()); // /app0/<Project>/Content/Paks
        ec.clear();
    }

    for (const fs::path& paks : pak_dirs) {
        std::vector<std::string> nonempty_files;
        std::error_code iter_ec;
        for (fs::directory_iterator it(paks, iter_ec), end; !iter_ec && it != end;
             it.increment(iter_ec)) {
            const bool regular = it->is_regular_file(iter_ec);
            if (iter_ec || !regular) {
                iter_ec.clear();
                continue;
            }
            const uintmax_t size = it->file_size(iter_ec);
            if (iter_ec || size == 0) {
                iter_ec.clear();
                continue;
            }
            nonempty_files.push_back(lower(it->path().filename().string()));
        }
        std::sort(nonempty_files.begin(), nonempty_files.end());
        nonempty_files.erase(std::unique(nonempty_files.begin(), nonempty_files.end()),
                             nonempty_files.end());
        for (const std::string& name : nonempty_files) {
            constexpr const char* prefix = "pakchunk";
            constexpr size_t prefix_len = 8;
            if (name.compare(0, prefix_len, prefix) != 0 ||
                name.size() < prefix_len + 2 + 5 ||
                name.compare(name.size() - 5, 5, ".utoc") != 0)
                continue;
            saw_iostore_index = true;
            const std::string data_name = name.substr(0, name.size() - 5) + ".ucas";
            if (!std::binary_search(nonempty_files.begin(), nonempty_files.end(), data_name))
                continue;
            size_t pos = prefix_len;
            uint32_t id = 0;
            while (pos < name.size() && std::isdigit(static_cast<unsigned char>(name[pos]))) {
                id = id * 10 + static_cast<unsigned>(name[pos++] - '0');
                if (id > UINT16_MAX) break;
            }
            if (pos == prefix_len || pos >= name.size() || name[pos] != '-' || id > UINT16_MAX)
                continue;
            chunks.push_back(static_cast<uint16_t>(id));
        }
    }
    std::sort(chunks.begin(), chunks.end());
    chunks.erase(std::unique(chunks.begin(), chunks.end()), chunks.end());
    if (chunks.empty() && !saw_iostore_index)
        chunks.push_back(0); // compatibility fallback for non-IoStore titles
    return chunks;
}

static std::vector<uint16_t> g_playgo_chunks{0};
static bool playgo_has_chunk(uint16_t id) {
    return std::binary_search(g_playgo_chunks.begin(), g_playgo_chunks.end(), id);
}

HLE(s_playgo_init)  { svc_log("scePlayGoInitialize", a0,a1,a2,a3,a4,a5);
                      g_playgo_chunks = discover_playgo_chunks();
                      if (svclog()) {
                          std::fprintf(stderr, "[svc] PlayGo discovered %zu installed chunk(s):",
                                       g_playgo_chunks.size());
                          for (uint16_t id : g_playgo_chunks) std::fprintf(stderr, " %u", id);
                          std::fputc('\n', stderr);
                      }
                      return 0; }
HLE(s_playgo_term)  { return 0; }
// scePlayGoOpen(u32* outHandle, const void* param): the handle the whole API is keyed on. The
// unimpl stub's success-with-unfilled-handle left the game querying loci with garbage.
HLE(s_playgo_open)  { svc_log("scePlayGoOpen", a0,a1,a2,a3,a4,a5);
                      if (!a0) return PLAYGO_ERR_BAD_POINTER;
                      *(uint32_t*)PW(a0) = 1; return 0; }
HLE(s_playgo_close) { return 0; }
// scePlayGoGetLocus(h, const u16* chunkIds, u32 n, s8* outLoci): installed chunks are LOCAL_FAST.
HLE(s_playgo_getlocus) { svc_log("scePlayGoGetLocus", a0,a1,a2,a3,a4,a5, 2);
                         if (!a1 || !a3) return PLAYGO_ERR_BAD_POINTER;
                         if (!(uint32_t)a2) return PLAYGO_ERR_BAD_SIZE;
                         const auto* ids = (const uint16_t*)PW(a1);
                         auto* loci = (int8_t*)PW(a3);
                         for (uint32_t i = 0; i < (uint32_t)a2; ++i) {
                             if (!playgo_has_chunk(ids[i])) {
                                 loci[i] = 0; // SCE_PLAYGO_LOCUS_NOT_DOWNLOADED
                                 return PLAYGO_ERR_BAD_CHUNK_ID;
                             }
                             loci[i] = 3; // SCE_PLAYGO_LOCUS_LOCAL_FAST
                         }
                         return 0; }
// scePlayGoGetProgress(h, chunkIds, n, OrbisPlayGoProgress* out): one struct {u64 progressSize;
// u64 totalSize} summed over the queried chunks; fully-installed == progressSize==totalSize!=0.
HLE(s_playgo_getprogress) { svc_log("scePlayGoGetProgress", a0,a1,a2,a3,a4,a5, 2);
                            if (!a1 || !a3) return PLAYGO_ERR_BAD_POINTER;
                            if (!(uint32_t)a2) return PLAYGO_ERR_BAD_SIZE;
                            const auto* ids = (const uint16_t*)PW(a1);
                            for (uint32_t i = 0; i < (uint32_t)a2; ++i)
                                if (!playgo_has_chunk(ids[i])) return PLAYGO_ERR_BAD_CHUNK_ID;
                            uint64_t* p = (uint64_t*)PW(a3);
                            p[0] = p[1] = (uint64_t)(uint32_t)a2 << 20;  // 1 MiB/chunk, done==total
                            return 0; }
// scePlayGoGetToDoList(h, OrbisPlayGoToDo* list, u32 n, u32* outEntries): nothing left to install.
HLE(s_playgo_gettodo) { if (!a3) return PLAYGO_ERR_BAD_POINTER; *(uint32_t*)PW(a3) = 0; return 0; }
HLE(s_playgo_settodo) { return 0; }
// scePlayGoGetChunkId(h, u16* list, u32 n, u32* outEntries): enumerate the installed chunk ids.
HLE(s_playgo_getchunkid) { if (!a3) return PLAYGO_ERR_BAD_POINTER;
                           if (a1 && !(uint32_t)a2) return PLAYGO_ERR_BAD_SIZE;
                           const uint32_t count = static_cast<uint32_t>(g_playgo_chunks.size());
                           if (a1) {
                               const uint32_t copied = std::min((uint32_t)a2, count);
                               std::memcpy(PW(a1), g_playgo_chunks.data(), copied * sizeof(uint16_t));
                               *(uint32_t*)PW(a3) = copied;
                           } else *(uint32_t*)PW(a3) = count;
                           return 0; }
// scePlayGoGetEta(h, chunkIds, n, s64* outEta): everything installed -> 0 seconds.
HLE(s_playgo_geteta) { if (!a1 || !a3) return PLAYGO_ERR_BAD_POINTER;
                       if (!(uint32_t)a2) return PLAYGO_ERR_BAD_SIZE;
                       *(int64_t*)PW(a3) = 0; return 0; }
// scePlayGoGetInstallSpeed(h, s32* out): FULL (2) — nothing throttled.
HLE(s_playgo_getspeed) { if (!a1) return PLAYGO_ERR_BAD_POINTER; *(int32_t*)PW(a1) = 2; return 0; }
// scePlayGoGetLanguageMask(h, u64* out): all languages present. CONFIDENCE: MED (mask semantics
// are per-language bits; all-ones = every language's chunks installed).
HLE(s_playgo_getlang) { if (!a1) return PLAYGO_ERR_BAD_POINTER;
                        *(uint64_t*)PW(a1) = ~0ull; return 0; }

// --- libSceSaveData (PS5 "native" surface): report a clean FRESH console — no existing save. ----
// DOLL calls Initialize3 -> CreateTransactionResource -> Mount3 -> Umount2 -> Prepare -> Commit
// (live-captured first-seen order). Initialize3 is PS4-inherited (Kyty returns OK). Mount3 /
// Prepare / Commit / CreateTransactionResource are PS5-only (present ONLY in the PS5 3.20
// libSceSaveData native stub; no Kyty/shadPS4 implementation exists), so their exact structs are
// unreferenced. Policy: a mount of a save that does not exist returns NOT_FOUND (0x809F0008 —
// shadPS4 savedata_error.h; same 0x809F error facility on PS5) and writes NOTHING — the truthful
// first-boot state on real hardware, which a shipped game must handle by proceeding to a fresh
// game. This is strictly better than the previous unimpl->0 "mount succeeded" with a garbage
// mount-result the game then reads paths from. Transaction bookkeeping calls allocate/tear down
// local resources only, and Create returns the new resource's ID (see #1905 below).
// CONFIDENCE: HIGH on Initialize3/NOT_FOUND semantics;
// MED on Mount3's arg order (mount-desc in, result out — matches every PS4 Mount variant);
// LOW on Prepare/Commit internals (no-op success; PROSPER_SVCLOG captures their real args).
static constexpr uint64_t SAVE_DATA_ERR_PARAMETER = 0x809F0000ull;
static constexpr uint64_t SAVE_DATA_ERR_EXISTS = 0x809F0007ull;
static constexpr uint64_t SAVE_DATA_ERR_NOT_FOUND = 0x809F0008ull;
// 0x809F0018 is the "the operation is STILL IN FLIGHT, keep waiting" code. That meaning is
// corroborated by four independent titles in the local dump set, each of which sleeps and re-polls
// on it -- better evidence than exists for most constants in this file:
//
//   PPSA15552 Dead Cells      +0x173c9b0  call GetEventResult; cmp eax,0x809f0018; jne <exit>;
//                                         mov edi,0x1f40; call sleep; jmp 0x173c9b0
//   PPSA15552 Dead Cells      +0x173cda0  same idiom, second site
//   PPSA28061 Earthion        +0x12f6d    cmp [rbp-0x7c],0x809f0018; jne; sleep(50 ms); jmp <repoll>
//   PPSA03831 Sonic Frontiers +0x18a2285  cmp eax,0x809f0018; jne; sleep(1); jmp <repoll>
//   PPSA05325 Sonic Origins   +0x940385   byte-identical (same SEGA save library)
//
// prosper has NOTHING in flight: every file operation here completes synchronously and the only
// event ever queued is Umount2's. So answering 0x809F0018 was not an unestablished value -- it was
// a well-established "still busy" returned in a state where nothing is busy, i.e. a permanent lie,
// and a guaranteed infinite hang for any title that reaches one of those loops.
//
// A DRAINED queue is therefore reported with NOT_FOUND. PPSA20447 (The First Berserker: Khazan)
// pins that value directly -- its game thread's drain loop leaves only on 0x809F0008:
//
//   eboot+0x796eb2c   jmp    0x796eb3d             ; loop ENTRY -- it polls first, sleeps after
//   eboot+0x796eb38   call   0x1565790             ; FPlatformProcess::Sleep(float)
//   eboot+0x796eb41.. vmovups/mov                  ; zero a 104-byte SceSaveDataEvent (96 + 8)
//   eboot+0x796eb67   xor    edi,edi               ; eventParam = NULL
//   eboot+0x796eb6f   call   0x8eaf3d0             ; sceSaveDataGetEventResult(NULL, &event)
//   eboot+0x796eb7b   mov    r12d,eax              ; r12d IS the return value
//   eboot+0x796eb88   cmp    r12d,0x809f0008       ; <-- the ONLY value that ends the wait
//   eboot+0x796eb8f   jne    0x796eb30             ; anything else: sleep and poll again
//   eboot+0x796eb91   movzx  eax,BYTE PTR [rip+..] ; a SECOND gate can still re-enter the loop
//   eboot+0x796eb9a   jne    0x796eb30
//
// Earthion const-compares BOTH, in consecutive instructions -- 0x809F0018 -> sleep and re-poll,
// then 0x809F0008 -> give up and return -- so the two codes are genuinely distinct in a shipping
// title's bytes and this file must not merge their meanings, only their current value.
//
// SPELLED AS A LITERAL, not aliased to SAVE_DATA_ERR_NOT_FOUND on purpose: the drained-queue answer
// HAPPENS to be NOT_FOUND today because prosper never has an operation in flight. An implementation
// that gives sceSaveDataMount3 a real asynchronous path (which Earthion's wait needs -- see below)
// must return SAVE_DATA_ERR_IN_FLIGHT while the operation runs, and should be able to do that
// without unpicking an alias.
// CONFIDENCE: HIGH on both values (five titles' own compare instructions).
static constexpr uint64_t SAVE_DATA_ERR_IN_FLIGHT = 0x809F0018ull;   // "still running, keep waiting"
static constexpr uint64_t SAVE_DATA_ERR_NO_EVENT  = 0x809F0008ull;   // drained: same value as NOT_FOUND
static_assert(SAVE_DATA_ERR_NO_EVENT == SAVE_DATA_ERR_NOT_FOUND,
              "a drained queue is reported with NOT_FOUND (PPSA20447 eboot+0x796eb88)");
static_assert(SAVE_DATA_ERR_IN_FLIGHT != SAVE_DATA_ERR_NO_EVENT,
              "the in-flight and drained codes are distinct in Earthion's bytes (eboot+0x12f6d/+0x12f82)");
namespace { std::atomic<unsigned> g_savedata_umount_events{0}; }
HLE(s_savedata_init3)   { svc_log("sceSaveDataInitialize3", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_savedata_term)    { return 0; }

// --- Transaction resources (#1905). sceSaveDataCreateTransactionResource returns the NEW RESOURCE'S
// ID, not 0-on-success. Returning 0 is the shape of the call that succeeds, so this looked correct
// and is not: Sonic Origins' (PPSA05325) save handler at eboot+0x93fdb0 is exactly
//     xor edi,edi; call sceSaveDataCreateTransactionResource; test eax,eax; jle <fail>
// — a zero return is read as failure, the handler records error 3, every later save operation
// returns that sticky error without running, and the title's boot coroutine polls the failed job
// forever. That single wrong return value is why PPSA05325 never leaves its first boot step.
// It really is a HANDLE and not merely "a positive number" — Sonic's full lifecycle proves it:
//   0x93fdc7  mov [r14+0xc0],eax        Create's result is retained in a member
//   0x9402cd  mov eax,[r14+0xc0]        ...loaded again to build the Mount3 descriptor,
//   0x9402d4  mov [rsp+0x68],eax        ...at descriptor base (rsp+0x40) + 0x28,
//   0x93f24e  mov edi,[rdi+0xc0]        ...and handed back as Delete's sole argument,
//   0x93f254  cmp edi,0xffffffff        guarded by the guest's own -1 "no resource" sentinel.
// That also independently reproduces the Mount3 +0x28 field this file documents from DQ7. The id is
// opaque to the guest, so any positive value satisfies it; we hand out a monotonic counter and keep
// the live set so Delete can reject one that was never created.
// CONFIDENCE: HIGH on the polarity, on "returns the resource id", and on Delete's argument being a
// scalar int32 in edi (the mov above precedes all four of Sonic's Delete sites and Oregon Trail's).
// Delete's error CODE for an unknown id is the residual unknown: PARAMETER is the facility's generic
// bad-argument code, and no title is observed passing an id Create did not return.
namespace {
std::mutex g_savedata_tx_mu;
int32_t g_savedata_tx_next = 1;
std::set<int32_t> g_savedata_tx_live;
}   // namespace

int32_t savedata_tx_resource_create() {
    std::lock_guard<std::mutex> lk(g_savedata_tx_mu);
    if (g_savedata_tx_next <= 0) return -1;          // counter exhausted (unreachable in practice)
    int32_t id = g_savedata_tx_next++;
    g_savedata_tx_live.insert(id);
    return id;
}

bool savedata_tx_resource_destroy(int32_t id) {
    std::lock_guard<std::mutex> lk(g_savedata_tx_mu);
    return g_savedata_tx_live.erase(id) != 0;
}

size_t savedata_tx_resource_live_count() {
    std::lock_guard<std::mutex> lk(g_savedata_tx_mu);
    return g_savedata_tx_live.size();
}

HLE(s_savedata_txres) {
    svc_log("sceSaveDataCreateTransactionResource", a0,a1,a2,a3,a4,a5);
    int32_t id = savedata_tx_resource_create();
    if (id <= 0) return SAVE_DATA_ERR_PARAMETER;
    return (uint64_t)(uint32_t)id;
}
HLE(s_savedata_txres_del) {
    svc_log("sceSaveDataDeleteTransactionResource", a0,a1,a2,a3,a4,a5);
    if (!savedata_tx_resource_destroy((int32_t)(uint32_t)a0)) return SAVE_DATA_ERR_PARAMETER;
    return 0;
}

// --- libSceSaveData "save-data memory" API (#191). A per-(user,slot) fixed-size memory block the
// managed SaveData layer reads/writes by offset and syncs to storage; on PS5 this is how a title
// keeps a small always-resident save (settings/progress). We back each slot with a host-memory
// block: Setup allocates it, Set copies guest->block, Get copies block->guest, Sync commits. Struct
// layouts + field offsets verified against shadPS4 save_data/savedata.cpp; error codes from
// savedata_error.h. CONFIDENCE: HIGH on the happy-path round-trip (the contract a title depends on).
//   OrbisSaveDataMemoryData:  buf@0(ptr) bufSize@8(u64) offset@16(s64)               size 64
//   OrbisSaveDataMemorySetup2: option@0 userId@4 memorySize@8 iconMemorySize@16
//                              initParam@24 initIcon@32 slotId@40                    size 64
//   OrbisSaveDataMemorySet2:  userId@0 [pad@4] data@8(ptr) param@16 icon@24 dataNum@32 slotId@36
//   OrbisSaveDataMemoryGet2:  userId@0 [pad@4] data@8(ptr) param@16 icon@24 slotId@32
namespace {
    std::mutex g_savemem_mx;
    // The in-process view of the SaveDataMemory slots. Keyed by TITLE as well as (userId, slotId),
    // because (userId, slotId) is not unique across titles — every Unity title uses user 1, slot 0.
    //
    // No shipping frontend boots two titles in one process today: prosper-app's start_guest()
    // latches g_boot_attempted on the ATTEMPT and routes the second title through
    // relaunch_with_dump(), i.e. a new process. So this is not fixing a reachable collision; it is
    // keeping the cache and the on-disk layout partitioned the SAME way, so the cache cannot become
    // a second source of truth that disagrees with the files. What does exercise it is
    // test_savedata_title_namespace, which drives two application roots through one process — and a
    // cache that outlived the title switch would hand title B title A's block while the files were
    // correctly separated, which is a harder bug to see than the one being fixed.
    std::unordered_map<std::string, std::vector<uint8_t>> g_savemem;
    std::string savemem_key(int32_t userId, uint32_t slotId) {
        char suffix[48];
        snprintf(suffix, sizeof suffix, "/%d:%u", (int)userId, (unsigned)slotId);
        return save_title_namespace() + suffix;
    }
    // Host file backing one SaveDataMemory slot, so a save survives a process restart (the API is the
    // ENTIRE save path for the Unity titles — no file Mount — so without this every relaunch looks like
    // a fresh console and the game restarts from scratch; likely root of #299).
    //
    // PER TITLE: <PROSPER_SAVEDATA_DIR or the per-user default>/<TITLE_ID>/savemem_<user>_<slot>.bin.
    // (userId, slotId) is not unique across titles — every Unity title writes user 1 slot 0 — so a
    // flat directory made two titles share one save file (#2734). Not cached in a static: the title
    // component comes from set_app0_root()'s param.json parse, and caching would freeze whichever
    // title resolved it first. `create` is passed only by the writer, so reading a slot for a title
    // that has never saved does not manufacture a directory for it.
    std::string savemem_path(int32_t userId, uint32_t slotId, bool create = false) {
        const std::string base = create ? savedata_mem_ensure_dir() : savedata_mem_dir();
        if (base.empty()) return {};
        char name[64];
        snprintf(name, sizeof name, "/savemem_%d_%u.bin", (int)userId, (unsigned)slotId);
        return base + name;
    }
    std::vector<uint8_t> savemem_load(int32_t userId, uint32_t slotId) {
        std::vector<uint8_t> v;
        const std::string path = savemem_path(userId, slotId);
        if (path.empty()) return v;
        if (FILE* f = fopen(path.c_str(), "rb")) {
            fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
            if (n > 0) { v.resize((size_t)n); if (fread(v.data(), 1, (size_t)n, f) != (size_t)n) v.clear(); }
            fclose(f);
        }
        return v;
    }
    void savemem_store(int32_t userId, uint32_t slotId, const std::vector<uint8_t>& buf) {
        const std::string path = savemem_path(userId, slotId, /*create=*/true);
        if (path.empty()) return;
        if (FILE* f = fopen(path.c_str(), "wb")) {
            if (!buf.empty()) fwrite(buf.data(), 1, buf.size(), f);
            fclose(f);
        }
    }
    template <class T> T ld(uint64_t base, size_t off) {   // read a guest struct field at byte offset
        T v; memcpy(&v, (const uint8_t*)PW(base) + off, sizeof(T)); return v;
    }
    uint64_t savemem_setup_block(int32_t userId, uint32_t slotId, uint64_t memSize) {
        uint64_t existed = 0;
        std::lock_guard<std::mutex> lk(g_savemem_mx);
        auto& buf = g_savemem[savemem_key(userId, slotId)];
        if (buf.empty()) buf = savemem_load(userId, slotId);
        existed = buf.size();
        if (memSize > buf.size()) buf.resize(memSize, 0);
        return existed;
    }
    void savemem_write_range(std::vector<uint8_t>& dst, uint64_t guestBuf, uint64_t guestSize,
                             int64_t offset) {
        if (!guestBuf || offset < 0 || (uint64_t)offset >= dst.size()) return;
        uint64_t n = std::min<uint64_t>(guestSize, dst.size() - (uint64_t)offset);
        memcpy(dst.data() + offset, PW(guestBuf), n);
    }
    void savemem_read_range(const std::vector<uint8_t>& src, uint64_t guestBuf, uint64_t guestSize,
                            int64_t offset) {
        if (!guestBuf || offset < 0 || (uint64_t)offset >= src.size()) return;
        uint64_t n = std::min<uint64_t>(guestSize, src.size() - (uint64_t)offset);
        memcpy(PW(guestBuf), src.data() + offset, n);
    }
    constexpr uint64_t SD_ERR_PARAMETER      = 0x809F0000ull;
    constexpr uint64_t SD_ERR_MEMORY_NOTREADY = 0x809F0012ull;   // Set/Get before a successful Setup
}
// sceSaveDataSetupSaveDataMemory2(setup*, result*): allocate/grow the slot's block to memorySize
// (zero-filling new bytes, preserving any bytes already there this session). result->existedMemorySize
// (@0) = the size the block had before — the game uses it to tell first-run from resume.
HLE(s_savemem_setup) {
    svc_log("sceSaveDataSetupSaveDataMemory2", a0,a1,a2,a3,a4,a5);
    if (!a0) return SD_ERR_PARAMETER;
    int32_t  userId  = ld<int32_t>(a0, 4);
    uint64_t memSize = ld<uint64_t>(a0, 8);
    uint32_t slotId  = ld<uint32_t>(a0, 40);
    uint64_t existed = savemem_setup_block(userId, slotId, memSize);
    if (a1) *(uint64_t*)PW(a1) = existed;                          // result->existedMemorySize
    return 0;
}
// sceSaveDataSetSaveDataMemory2(set*): copy each of dataNum {buf,bufSize,offset} descriptors from
// guest memory into the slot's block at its offset (bounds-clamped — never write past the block).
HLE(s_savemem_set) {
    svc_log("sceSaveDataSetSaveDataMemory2", a0,a1,a2,a3,a4,a5);
    if (!a0) return SD_ERR_PARAMETER;
    int32_t  userId  = ld<int32_t>(a0, 0);
    uint64_t dataPtr = ld<uint64_t>(a0, 8);
    uint32_t dataNum = ld<uint32_t>(a0, 32);
    uint32_t slotId  = ld<uint32_t>(a0, 36);
    std::lock_guard<std::mutex> lk(g_savemem_mx);
    auto it = g_savemem.find(savemem_key(userId, slotId));
    if (it == g_savemem.end()) return SD_ERR_MEMORY_NOTREADY;      // Setup not called for this slot
    auto& buf = it->second;
    for (uint32_t i = 0; i < dataNum && dataPtr; i++) {
        uint64_t d     = dataPtr + (uint64_t)i * 64;               // sizeof(OrbisSaveDataMemoryData)
        uint64_t gbuf  = ld<uint64_t>(d, 0);
        uint64_t gsize = ld<uint64_t>(d, 8);
        int64_t  off   = ld<int64_t>(d, 16);
        savemem_write_range(buf, gbuf, gsize, off);
    }
    return 0;
}
// sceSaveDataGetSaveDataMemory2(get*): copy from the slot's block into the single {buf,bufSize,offset}
// descriptor (Get2 has no dataNum — one descriptor), bounds-clamped.
HLE(s_savemem_get) {
    svc_log("sceSaveDataGetSaveDataMemory2", a0,a1,a2,a3,a4,a5);
    if (!a0) return SD_ERR_PARAMETER;
    int32_t  userId  = ld<int32_t>(a0, 0);
    uint64_t dataPtr = ld<uint64_t>(a0, 8);
    uint32_t slotId  = ld<uint32_t>(a0, 32);
    std::lock_guard<std::mutex> lk(g_savemem_mx);
    auto it = g_savemem.find(savemem_key(userId, slotId));
    if (it == g_savemem.end()) return SD_ERR_MEMORY_NOTREADY;
    auto& buf = it->second;
    if (dataPtr) {
        uint64_t gbuf  = ld<uint64_t>(dataPtr, 0);
        uint64_t gsize = ld<uint64_t>(dataPtr, 8);
        int64_t  off   = ld<int64_t>(dataPtr, 16);
        savemem_read_range(buf, gbuf, gsize, off);
    }
    return 0;
}
// The original SaveDataMemory API passes scalar arguments and always addresses slot 0. These PS4-
// inherited exports are still present in the PS5 3.20 table; route them through the same backing
// store as the struct-based *2 API so callers do not receive fake success with no data transfer.
HLE(s_savemem_setup_v1) {
    svc_log("sceSaveDataSetupSaveDataMemory", a0,a1,a2,a3,a4,a5);
    savemem_setup_block((int32_t)a0, 0, a1);
    return 0;
}
HLE(s_savemem_set_v1) {
    svc_log("sceSaveDataSetSaveDataMemory", a0,a1,a2,a3,a4,a5);
    std::lock_guard<std::mutex> lk(g_savemem_mx);
    auto it = g_savemem.find(savemem_key((int32_t)a0, 0));
    if (it == g_savemem.end()) return SD_ERR_MEMORY_NOTREADY;
    savemem_write_range(it->second, a1, a2, (int64_t)a3);
    return 0;
}
HLE(s_savemem_get_v1) {
    svc_log("sceSaveDataGetSaveDataMemory", a0,a1,a2,a3,a4,a5);
    std::lock_guard<std::mutex> lk(g_savemem_mx);
    auto it = g_savemem.find(savemem_key((int32_t)a0, 0));
    if (it == g_savemem.end()) return SD_ERR_MEMORY_NOTREADY;
    savemem_read_range(it->second, a1, a2, (int64_t)a3);
    return 0;
}
// sceSaveDataSyncSaveDataMemory(sync*): commit the slot's block. A session's Set is already visible to a
// later Get in-process; Sync additionally writes the block to a host file (savemem_store), so the save
// survives a process restart — Setup reloads it next launch, making existedMemorySize truthful.
HLE(s_savemem_sync) {
    svc_log("sceSaveDataSyncSaveDataMemory", a0,a1,a2,a3,a4,a5);
    if (!a0) return SD_ERR_PARAMETER;
    int32_t  userId = ld<int32_t>(a0, 0);
    uint32_t slotId = ld<uint32_t>(a0, 4);
    std::lock_guard<std::mutex> lk(g_savemem_mx);
    auto it = g_savemem.find(savemem_key(userId, slotId));
    if (it == g_savemem.end()) return SD_ERR_MEMORY_NOTREADY;
    savemem_store(userId, slotId, it->second);   // Sync commits the slot to disk (survives restart)
    return 0;
}
// All three pinned mount ABIs feed one backend/result writer. The PS4-inherited layouts come from
// the public libSceSaveData ABI and retain their NIDs in the PS5 3.20 table:
//   Mount:  dirName pointer @0x10, blocks @0x20, mode @0x28, size 0x50
//   Mount2: dirName pointer @0x08, blocks @0x10, mode @0x18, size 0x40
// Mount3's PS5-native layout is documented below. MountResult is the shared exact 0x40-byte shape.
static uint64_t savedata_mount_common(const char* api, const char* dirname,
                                      uint32_t mode, uint64_t result_va) {
    if (!dirname || !*dirname || !result_va) return SAVE_DATA_ERR_PARAMETER;
    const SaveDataMountPolicy policy = (mode & 0x04) ? SaveDataMountPolicy::Create
        : (mode & 0x20) ? SaveDataMountPolicy::OpenOrCreate
                        : SaveDataMountPolicy::Open;
    const SaveDataMountOutcome outcome = savedata0_mount(dirname, policy);
    if (outcome == SaveDataMountOutcome::Exists) {
        if (svclog()) fprintf(stderr, "[svc]   %s dir='%s' mode=%#x -> EXISTS\n",
                              api, dirname, mode);
        return SAVE_DATA_ERR_EXISTS;
    }
    if (outcome == SaveDataMountOutcome::NotFound) {
        if (svclog()) fprintf(stderr, "[svc]   %s dir='%s' mode=%#x -> NOT_FOUND\n",
                              api, dirname, mode);
        return SAVE_DATA_ERR_NOT_FOUND;
    }
    uint8_t* result = (uint8_t*)PW(result_va);
    memset(result, 0, 0x40);
    memcpy(result, "/savedata0", 11);
    const bool created = outcome == SaveDataMountOutcome::Created;
    *(uint32_t*)(result + 0x1c) = created ? 1u : 0u;
    if (svclog()) fprintf(stderr, "[svc]   %s dir='%s' mode=%#x -> OK (created=%d)\n",
                          api, dirname, mode, (int)created);
    return 0;
}

HLE(s_savedata_mount) {
    svc_log("sceSaveDataMount", a0,a1,a2,a3,a4,a5);
    if (!a0 || !a1) return SAVE_DATA_ERR_PARAMETER;
    const uint8_t* mount = (const uint8_t*)PW(a0);
    const char* dirname = *(const char* const*)(mount + 0x10);
    const uint32_t mode = *(const uint32_t*)(mount + 0x28);
    return savedata_mount_common("Mount", dirname, mode, a1);
}

HLE(s_savedata_mount2) {
    svc_log("sceSaveDataMount2", a0,a1,a2,a3,a4,a5);
    if (!a0 || !a1) return SAVE_DATA_ERR_PARAMETER;
    const uint8_t* mount = (const uint8_t*)PW(a0);
    const char* dirname = *(const char* const*)(mount + 0x08);
    const uint32_t mode = *(const uint32_t*)(mount + 0x18);
    return savedata_mount_common("Mount2", dirname, mode, a1);
}

// sceSaveDataMount3(const Mount3* mount, MountResult* result). The mount desc layout is pinned
// from DOLL's OWN wrapper (eboot+0x2251610 disassembly, matching the live capture):
//   +0x00 u32 userId; +0x08 const char* dirName; +0x10 u64 blocks; +0x20 u32 mountMode;
//   +0x28 u32 transactionResourceId  (the id returned by sceSaveDataCreateTransactionResource)
// Live: dirName="Book", blocks 0x60 then 0x105, mode 1 (open-RO) then 5 (CREATE|RO).
// mountMode bits are PS4-inherited: 1=RDONLY 2=RDWR 4=CREATE 8=DESTRUCT_OFF 16=COPY_ICON.
// Behavior (real console semantics): open of a nonexistent save -> NOT_FOUND (fresh console;
// the game handles it and retries with CREATE); CREATE makes the save dir and mounts it.
// The 0x40-byte result is zeroed by the caller and fed to sceSaveDataPrepare(&{txId}, result);
// we fill it with the PS4 MountResult shape (the only referenced layout): mountPoint char[16]
// "/savedata0" @+0x00, requiredBlocks u64 @+0x10 = 0, mountStatus u32 @+0x1c (0=opened,
// 1=created). hle_file translates /savedata0 to the mounted host dir.
// CONFIDENCE: HIGH on the mount-desc layout + mode semantics (guest disasm + live capture +
// PS4 references agree); MED on the result layout (PS4 shape; the PS5 field placement is
// unproven — under live test which offsets the game actually reads).
HLE(s_savedata_mount3)  {
    svc_log("sceSaveDataMount3", a0,a1,a2,a3,a4,a5);
    if (!a0 || !a1) return SAVE_DATA_ERR_PARAMETER;
    const uint8_t* m = (const uint8_t*)PW(a0);
    const char* dirname = *(const char* const*)(m + 0x08);
    uint32_t mode = *(const uint32_t*)(m + 0x20);
    return savedata_mount_common("Mount3", dirname, mode, a1);
}
HLE(s_savedata_umount) {
    svc_log("sceSaveDataUmount", a0,a1,a2,a3,a4,a5);
    if (!a0) return SAVE_DATA_ERR_PARAMETER;
    const char* mount_point = (const char*)PW(a0); // OrbisSaveDataMountPoint: char data[16]
    if (strncmp(mount_point, "/savedata0", 16) != 0) return SAVE_DATA_ERR_NOT_FOUND;
    return savedata0_umount() ? 0 : SAVE_DATA_ERR_NOT_FOUND;
}
HLE(s_savedata_umount2) {
    svc_log("sceSaveDataUmount2", a0,a1,a2,a3,a4,a5);
    savedata0_umount();
    g_savedata_umount_events.fetch_add(1, std::memory_order_release);
    return 0;
}
// sceSaveDataGetMountInfo(mp, OrbisSaveDataMountInfo* info = { u64 blocks; u64 freeBlocks; u8 rsv[32] }, 48
// bytes): was MISSING -> success with garbage free-space, so a title sizing its save against freeBlocks
// could abort ("disk full") or corrupt its math. Report a generous, consistent size (256K blocks free).
HLE(s_savedata_mountinfo) {
    svc_log("sceSaveDataGetMountInfo", a0,a1,a2,a3,a4,a5);
    if (!a1) return 0x809F0000ull;   // SAVE_DATA_ERROR_PARAMETER
    uint8_t* i = (uint8_t*)PW(a1); memset(i, 0, 48);
    *(uint64_t*)(i + 0) = 0x40000; *(uint64_t*)(i + 8) = 0x40000;   // blocks / freeBlocks
    return 0;
}
HLE(s_savedata_prepare) { svc_log("sceSaveDataPrepare", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_savedata_commit)  { svc_log("sceSaveDataCommit", a0,a1,a2,a3,a4,a5); return 0; }
// sceSaveDataGetEventResult(eventParam, event): PS4 and PS5 share this NID and two-argument shape.
// Our file operations complete synchronously, but Umount2 still queues the completion event a title
// uses to finish its save job. Return one UMOUNT_BACKUP event per unmount, then NO_EVENT. Returning
// generic success with an untouched event made Dead Cells consume a fabricated type-0 completion.
// Event: { u32 type; s32 errorCode; s32 userId; u32 pad; titleId[16]; dirName[32]; reserved[40] },
// 104 bytes. CONFIDENCE: HIGH on signature/error/type/size (Dead Cells zeroes exactly 104 bytes and
// the identical PS4 NID/API defines that layout). The dirName SLOT is now HIGH too, not MED:
// PPSA28061 Earthion zeroes 0x68 bytes and then does memcmp(event + 0x20, <dirName>, 0x20) at
// eboot+0x12f99, which pins dirName[32] at +0x20 in a 104-byte struct from a second title's bytes.
//
// KNOWN GAP -- prosper posts NO PER-OPERATION COMPLETION EVENT, and one title is already waiting for
// one. The only event this ever queues comes from Umount2, and it is zero-filled: no titleId, no
// dirName. Earthion's wait loop (eboot+0x12f30) polls until it gets an event whose dirName matches
// the directory it is waiting on, and takes its match branch at +0x12fb4 only when that memcmp
// succeeds -- which is STRUCTURALLY UNREACHABLE here. It survives only because it also has a
// give-up branch on the drained code. Implementing async Mount3/Prepare/Commit means (a) returning
// SAVE_DATA_ERR_IN_FLIGHT while the operation runs and (b) queueing a completion event that carries
// the operation's dirName -- not just bumping a counter. Tracked on #2909 / #1880.
HLE(s_savedata_get_event) {
    svc_log("sceSaveDataGetEventResult", a0,a1,a2,a3,a4,a5);
    if (!a1) return SAVE_DATA_ERR_PARAMETER;
    unsigned pending = g_savedata_umount_events.load(std::memory_order_acquire);
    while (pending) {
        if (g_savedata_umount_events.compare_exchange_weak(
                pending, pending - 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            uint8_t* event = (uint8_t*)PW(a1);
            memset(event, 0, 104);
            *(uint32_t*)(event + 0) = 1;  // SCE_SAVE_DATA_EVENT_TYPE_UMOUNT_BACKUP
            *(int32_t*)(event + 8) = 1;  // initial user
            // dirName (+0x20) is deliberately left zeroed: this event is not attributed to any one
            // save directory, and inventing a name here would make Earthion's memcmp match the
            // wrong operation. See the KNOWN GAP above.
            return 0;
        }
    }
    return SAVE_DATA_ERR_NO_EVENT;
}
// sceSaveDataDirNameSearch(const SearchCond* cond, SearchResult* result) — PS4-inherited contract,
// pinned by DOLL's own callsite (eboot+0x224e920): cond {u32 userId@0; titleId*@8=0; dirName*@0x10=0;
// key/order@0x18=0}, result {u32 hitNum@0; DirName* dirNames@8 (caller buffer, this+0x40);
// u32 dirNamesNum@0x10=0x400; u32 setNum@0x14(1.7+)} — the guest stores hitNum to this+0x8040 as its
// save count. Fresh console: 0 saves found, success (shadPS4 does exactly this when the save path
// doesn't exist). CONFIDENCE: HIGH (guest callsite disassembly + shadPS4 agree).
HLE(s_savedata_dirsearch) {
    svc_log("sceSaveDataDirNameSearch", a0,a1,a2,a3,a4,a5);
    if (!a1) return 0x809F0000ull;        // SAVE_DATA_ERROR_PARAMETER
    uint8_t* res = (uint8_t*)PW(a1);
    // Enumerate the save dirs on disk so a prior session's saves show up in the load/continue list (#299).
    // Was hard-coded to 0 hits, so persisted saves were invisible. Optional cond->dirName filter @cond+0x10.
    const char* filter = nullptr;
    if (a0) { uint64_t dnp = *(uint64_t*)((uint8_t*)PW(a0) + 0x10); if (dnp) filter = (const char*)PW(dnp); }
    std::vector<std::string> dirs = savedata0_list_dirs();
    uint32_t cap = *(uint32_t*)(res + 0x10);            // dirNamesNum (caller buffer capacity, entries)
    if (cap > 0x400) cap = 0x400;                        // clamp to the documented buffer size
    uint64_t buf = *(uint64_t*)(res + 0x08);            // DirName* dirNames (caller buffer)
    uint32_t total = 0;
    for (const std::string& name : dirs) {
        if (filter && name != filter) continue;
        if (buf && (!cap || total < cap)) {              // SceSaveDataDirName = char dirName[32]
            char* entry = (char*)PW(buf) + (size_t)total * 32;
            memset(entry, 0, 32);
            strncpy(entry, name.c_str(), 31);
        }
        total++;
    }
    uint32_t hit = (cap && total > cap) ? cap : total;
    *(uint32_t*)(res + 0x00) = hit;        // hitNum
    *(uint32_t*)(res + 0x14) = hit;        // setNum
    return 0;
}

// --- libSceNpTrophy2 lifecycle: succeed with valid ids (trophy CONTENT stays unavailable). ------
// PS4 NpTrophy ABI carried to Trophy2 (context/handle are small s32 ids written through arg0;
// Kyty LibNpTrophy + shadPS4 np_trophy agree on the PS4 shape). The game's trophy worker needs
// CreateContext/CreateHandle/RegisterContext to hand back usable ids so its bring-up completes;
// the info queries (GetGameInfo/GetTrophyInfoArray) keep returning "unavailable" (see
// s_nptrophy2_unavailable above) which the guest handles on a clean path. CONFIDENCE: MED.
HLE(s_nptrophy2_createctx)    { svc_log("sceNpTrophy2CreateContext", a0,a1,a2,a3,a4,a5);
                                if (a0) *(int32_t*)PW(a0) = 1; return 0; }
HLE(s_nptrophy2_createhandle) { svc_log("sceNpTrophy2CreateHandle", a0,a1,a2,a3,a4,a5);
                                if (a0) *(int32_t*)PW(a0) = 1; return 0; }
HLE(s_nptrophy2_regctx)       { svc_log("sceNpTrophy2RegisterContext", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_nptrophy2_ok)           { return 0; }

// --- libSceShare / libSceGameLiveStreaming: local lifecycle, features unavailable headless. -----
// The PS5 3.20 import table supplies the exact NIDs. Kyty's matching SDK surface gives the call
// shapes: ShareSetContentParam takes one required C string; GameLiveStreamingInitialize takes only
// a heap size. Neither successful call has an out-param. CONFIDENCE: HIGH for the live Sonic call
// shapes (PROSPER_SVCLOG), MED for the reference-derived invalid-param value.
HLE(s_share_ok) { return 0; }
HLE(s_share_content_param) {
    svc_log("sceShareSetContentParam", a0,a1,a2,a3,a4,a5);
    return a0 ? 0 : 0x81960002ull; // SCE_SHARE_ERROR_INVALID_PARAM (Kyty libShare.cpp)
}
HLE(s_live_streaming_init) {
    svc_log("sceGameLiveStreamingInitialize", a0,a1,a2,a3,a4,a5);
    return 0;
}

// --- libSceNpUniversalDataSystem (PS5 telemetry/activities): hand out ids, stay inert. ----------
// PS5-only, no reference implementation; by symmetry with every Np Create* API the first arg of
// CreateContext/CreateHandle is the out-id pointer (pointer-range-guarded so a wrong guess can't
// fault). Offline console: everything else no-ops. CONFIDENCE: LOW (guarded).
HLE(s_npuds_create) { svc_log("sceNpUniversalDataSystemCreate*", a0,a1,a2,a3,a4,a5);
                      if (svc_ptrish(a0)) *(int32_t*)PW(a0) = 1; return 0; }
HLE(s_npuds_ok)     { return 0; }
// Two observed CreateEvent call shapes (both write opaque ids the guest never dereferences):
//  * Dead Cells: CreateEvent(name?, reserved, Event** a2, PropertyObject** a3) — a2 AND a3 are
//    out-pointers.
//  * Alex Kidd DX (PPSA02664, svc dump): CreateEvent(name a0="activityTerminate", ctx a1,
//    Event** a2, 0, PropertyObject** a4, 0) — outs are a2 and a4, a3 is literal 0, a5 literal 0.
//    The old a3-must-be-a-pointer guard returned NP-invalid-argument here, and the title retried
//    every frame forever — holding its cutscene->gameplay transition on a black screen (#320).
// Accept both: a2 is always the event out; the property out is a3 when pointer-like. The a4 form is
// admitted ONLY for Alex Kidd's exact trailing shape (a3==0 AND a5==0) — svc_ptrish is a wide range
// check that cannot by itself tell a real out-pointer from leftover-register garbage, so requiring
// both trailing reserved words to be zero pins the write to the observed 6-arg layout. A different
// title that legitimately passes a3==0 with a non-zero/garbage a5 hits neither property write and
// still returns success (0), which is what actually stops the retry loop. CONFIDENCE: LOW.
HLE(s_npuds_create_event) {
    svc_log("sceNpUniversalDataSystemCreateEvent", a0,a1,a2,a3,a4,a5);
    if (!svc_ptrish(a2)) return 0x80550003ull; // NP invalid argument
    *(uint64_t*)PW(a2) = g_handle.fetch_add(1);
    if (svc_ptrish(a3))                                *(uint64_t*)PW(a3) = g_handle.fetch_add(1);
    else if (a3 == 0 && a5 == 0 && svc_ptrish(a4))     *(uint64_t*)PW(a4) = g_handle.fetch_add(1);
    return 0;
}
HLE(s_npuds_post_event) {
    svc_log("sceNpUniversalDataSystemPostEvent", a0,a1,a2,a3,a4,a5);
    return 0;
}
HLE(s_npuds_destroy_event) {
    svc_log("sceNpUniversalDataSystemDestroyEvent", a0,a1,a2,a3,a4,a5);
    return 0;
}
HLE(s_npuds_object_set_string) {
    svc_log("sceNpUniversalDataSystemEventPropertyObjectSetString", a0,a1,a2,a3,a4,a5);
    return 0;
}
HLE(s_gameintent_init) {
    svc_log("sceNpGameIntentInitialize", a0,a1,a2,a3,a4,a5);
    if (!g_gameintent_initialized.exchange(true, std::memory_order_acq_rel))
        g_gameintent_event_delivered.store(false, std::memory_order_release);
    g_gameintent_data_ptr.store(0, std::memory_order_release);
    return 0;
}
HLE(s_gameintent_term) {
    svc_log("sceNpGameIntentTerminate", a0,a1,a2,a3,a4,a5);
    g_gameintent_data_ptr.store(0, std::memory_order_release);
    g_gameintent_event_delivered.store(false, std::memory_order_release);
    g_gameintent_initialized.store(false, std::memory_order_release);
    return 0;
}
HLE(s_gameintent_receive) {
    svc_log("sceNpGameIntentReceiveIntent", a0,a1,a2,a3,a4,a5);
    if (!svc_ptrish(a0)) return NP_GAME_INTENT_ERROR_INVALID_ARGUMENT;

    const int32_t invalid_user = -1;
    const char empty_type[NP_GAME_INTENT_TYPE_SIZE]{};
    static const uint8_t empty_data[NP_GAME_INTENT_DATA_SIZE]{};
    if (!svc_write_bytes(a0 + 8, &invalid_user, sizeof(invalid_user)) ||
        !svc_write_bytes(a0 + NP_GAME_INTENT_TYPE_OFFSET, empty_type, sizeof(empty_type)) ||
        !svc_write_bytes(a0 + NP_GAME_INTENT_DATA_OFFSET, empty_data, sizeof(empty_data))) {
        return NP_GAME_INTENT_ERROR_INVALID_ARGUMENT;
    }
    g_gameintent_data_ptr.store(0, std::memory_order_release);

    if (!g_gameintent_initialized.load(std::memory_order_acquire))
        return NP_GAME_INTENT_ERROR_INTENT_NOT_FOUND;
    if (!gameintent_activity_id()) return NP_GAME_INTENT_ERROR_INTENT_NOT_FOUND;

    const int32_t initial_user = 1;
    constexpr char launch_activity[] = "launchActivity";
    if (!svc_write_bytes(a0 + 8, &initial_user, sizeof(initial_user)) ||
        !svc_write_bytes(a0 + NP_GAME_INTENT_TYPE_OFFSET, launch_activity,
                         sizeof(launch_activity))) {
        return NP_GAME_INTENT_ERROR_INVALID_ARGUMENT;
    }
    g_gameintent_data_ptr.store(a0 + NP_GAME_INTENT_DATA_OFFSET, std::memory_order_release);
    return 0;
}
HLE(s_gameintent_get_property_string) {
    svc_log("sceNpGameIntentGetPropertyValueString", a0,a1,a2,a3,a4,a5);
    if (!svc_ptrish(a0) || !svc_ptrish(a1) || !svc_ptrish(a2) || a3 == 0)
        return NP_GAME_INTENT_ERROR_INVALID_ARGUMENT;

    char key[sizeof("activityId")]{};
    if (!svc_copy_bytes(a1, key, sizeof(key))) return NP_GAME_INTENT_ERROR_INVALID_ARGUMENT;

    size_t activity_length = 0;
    const char* activity = gameintent_activity_id(&activity_length);
    if (a0 != g_gameintent_data_ptr.load(std::memory_order_acquire) || !activity ||
        memcmp(key, "activityId", sizeof(key)) != 0) {
        const char empty = '\0';
        if (!svc_write_bytes(a2, &empty, sizeof(empty)))
            return NP_GAME_INTENT_ERROR_INVALID_ARGUMENT;
        return NP_GAME_INTENT_ERROR_VALUE_NOT_FOUND;
    }
    if (a3 < activity_length + 1) return NP_GAME_INTENT_ERROR_INVALID_ARGUMENT;
    if (!svc_write_bytes(a2, activity, activity_length + 1))
        return NP_GAME_INTENT_ERROR_INVALID_ARGUMENT;
    return 0;
}
HLE(s_npent_addcont_info) {
    svc_log("sceNpEntitlementAccessGetAddcontEntitlementInfo", a0,a1,a2,a3,a4,a5);
    std::string label;
    if (a0 > UINT32_MAX || !appcontent_read_label(a1, label) || !svc_ptrish(a2))
        return NP_ENTITLEMENT_ERROR_PARAMETER;
    const AddcontentInventorySnapshot inventory = addcontent_inventory_snapshot();
    const InstalledAddcontent* entry = inventory.state == AddcontentInventoryState::Ready
        ? find_addcontent(inventory, static_cast<uint32_t>(a0), label) : nullptr;
    if (!entry) return NP_ENTITLEMENT_ERROR_NO_ENTITLEMENT;
    const NpAddcontEntitlementInfo info = npent_info(*entry);
    return svc_write_bytes(a2, &info, sizeof(info)) ? 0 : NP_ENTITLEMENT_ERROR_PARAMETER;
}

// ===== Issue #306: honest OFFLINE console for the online/update/entitlement boot chain. =========
// DOLL's UE4 front-end runs a patch/entitlement check before the title screen; every subsystem in
// that chain answered success-with-garbage, so the check could neither succeed nor FAIL — the flow
// waited forever (docs/DOLL_LOADING_PROGRESSION.md). The blocks below give the chain the answers a
// real, network-disconnected, signed-out console gives.

// --- Guest-callback delivery discipline (shared by NetCtl + Np state callbacks). ----------------
// A registered callback is guest code: under PROSPER_GUEST_FS the HLE runs on the HOST %fs (the
// import swap-stub switched), so the guest callback must run with the GUEST %fs restored or its
// TLS accesses (UE MallocBinned caches!) read host TLS garbage. The swap-stub saves the guest fs
// base in its frame (push r11), so an asm entry shim (the f_apr_read_submit_entry pattern) hands
// the handler its entry %rsp. The guest swap path re-pushes args7/8/9, an alignment pad, then the
// saved r11: [rsp]=ret-to-stub, args at +8/+0x10/+0x18, pad at +0x20, guest fs at +0x28, and guest
// RA at +0x30. A [rsp] outside the stub region [0x6_0000_0000,0x7_0000_0000) means the
// host-context tail-jmp path (no swap happened) — call the callback on the current fs.
// Mechanism proven live by the PROSPER_NETCTL_CB experiment (run 7/9: delivered + consumed
// cleanly, no crash). CONFIDENCE: HIGH.
#ifndef _WIN32
namespace {
inline uint64_t cb_rd_fsbase() { uint64_t v; __asm__ volatile("rdfsbase %0" : "=r"(v)); return v; }
inline void     cb_wr_fsbase(uint64_t v) { __asm__ volatile("wrfsbase %0" : : "r"(v)); }
// RAII: run the enclosed guest callback on the guest %fs (no-op when guest_fs==0).
struct CbGuestFsScope {
    uint64_t saved = 0, active = 0;
    explicit CbGuestFsScope(uint64_t guest_fs) {
        if (guest_fs) { saved = cb_rd_fsbase(); cb_wr_fsbase(guest_fs); active = guest_fs; }
    }
    ~CbGuestFsScope() { if (active) cb_wr_fsbase(saved); }
};
}
#endif

// --- libSceNetCtl: a network-DISCONNECTED console (default ON since #306). ----------------------
// DOLL registers a NetCtl state callback once at boot (sceNetCtlRegisterCallback) and then pumps
// sceNetCtlCheckCallback EXACTLY once per frame forever (14,191 calls in a 240 s run). On real
// hardware CheckCallback invokes the registered callback on the calling thread with the current
// state — an offline console still delivers an immediate DISCONNECTED. Register records {func,arg}
// and writes the callback id (Kyty Network.cpp NetCtlRegisterCallback); CheckCallback invokes the
// callback ONCE with SCE_NET_CTL_EVENT_TYPE_DISCONNECTED (PS4-inherited constant = 1; identical
// export names+NIDs on PS5 3.20 — CONFIDENCE MED on the PS5 value). Was the gated experiment
// PROSPER_NETCTL_CB=1; proven correct+consumed live (DOLL run 7/9), now default ON.
// PROSPER_NETCTL_CB=0 restores the old unimplemented behavior.
#ifndef _WIN32
namespace {
std::atomic<uint64_t> g_netctl_cb_fn{0};
std::atomic<uint64_t> g_netctl_cb_arg{0};
std::atomic<int>      g_netctl_cb_delivered{0};
}
HLE(s_netctl_register_cb) {   // (func, arg, int* cid)
    svc_log("sceNetCtlRegisterCallback", a0,a1,a2,a3,a4,a5);
    g_netctl_cb_fn.store(a0);
    g_netctl_cb_arg.store(a1);
    if (svc_ptrish(a2)) *(int32_t*)PW(a2) = 1;   // callback id (Kyty Network.cpp NetCtlRegisterCallback)
    return 0;
}
// sceNetCtlGetState(int* state): 0 = DISCONNECTED (Kyty Network.cpp:1398 writes exactly this).
// Run-7 live capture: the game calls this for the FIRST time immediately after the DISCONNECTED
// callback delivery — the unimplemented success+garbage-out answer is what re-wedged the flow.
HLE(s_netctl_getstate) {
    svc_log("sceNetCtlGetState", a0,a1,a2,a3,a4,a5);
    if (svc_ptrish(a0)) *(int32_t*)PW(a0) = 0;   // SCE_NET_CTL_STATE_DISCONNECTED
    return 0;
}
extern "C" uint64_t s_netctl_check_cb_c(uint64_t a0, uint64_t a1, uint64_t a2,
                                        uint64_t a3, uint64_t a4, uint64_t a5,
                                        uint64_t entry_rsp);
PROSPER_ASM_TRAMPOLINE(s_netctl_check_cb_entry, s_netctl_check_cb_c)
extern "C" void s_netctl_check_cb_entry();
extern "C" uint64_t s_netctl_check_cb_c(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                        uint64_t entry_rsp) {
    uint64_t fn = g_netctl_cb_fn.load();
    if (!fn || g_netctl_cb_delivered.exchange(1)) return 0;   // deliver the initial state exactly once
    uint64_t gfs = callback_guest_fs_from_entry_stack(entry_rsp);
    {
        CbGuestFsScope fs(gfs);
        ((void (*)(int, void*))(uintptr_t)fn)(1 /*SCE_NET_CTL_EVENT_TYPE_DISCONNECTED*/,
                                              (void*)(uintptr_t)g_netctl_cb_arg.load());
    }
    // NOTE: log only AFTER the scope restored the host %fs — host libc (fprintf) reads %fs-based
    // TLS and crashes on the guest %fs (learned the hard way: NULL+0x308 fault in libc).
    fprintf(stderr, "[svc] NetCtl state callback DELIVERED (eventType=DISCONNECTED, guest_fs=%d)\n",
            gfs ? 1 : 0);
    return 0;
}
#endif
// sceNetCtlGetInfo(int code, SceNetCtlInfo* info): a console with no network connection answers
// NOT_CONNECTED for the connection-dependent info codes and writes nothing (shadPS4 netctl.cpp:163
// returns ORBIS_NET_CTL_ERROR_NOT_CONNECTED = 0x80412108 for ALL codes when disconnected; Kyty
// only implements the connected path). PS4-inherited surface, identical export on PS5 3.20
// (obuxdTiwkF8). Works on all platforms (no callback machinery). CONFIDENCE: HIGH on semantics,
// MED on the PS5 errno value (same 0x8041 NetCtl facility).
HLE(s_netctl_getinfo) {
    svc_log("sceNetCtlGetInfo", a0,a1,a2,a3,a4,a5);
    return 0x80412108ull;   // SCE_NET_CTL_ERROR_NOT_CONNECTED
}

// --- libSceNpManager state callback: deliver SIGNED_OUT once (#306). ----------------------------
// DOLL registers its Np sign-in state callback via sceNpRegisterStateCallbackA and pumps
// sceNpCheckCallback. shadPS4 (offline mode) queues exactly one SIGNED_OUT event for the initial
// user and delivers it inside sceNpCheckCallback on the pumping thread; we mirror that. Callback-A
// prototype (shadPS4 np_manager.h): void cb(s32 userId, s32 state, void* userdata); state
// SIGNED_OUT = 1 (Unknown=0, SignedOut=1, SignedIn=2 — Kyty + shadPS4 agree). Register returns the
// positive callback id (shadPS4 RegisterStateCallbackA returns slot+1). CONFIDENCE: HIGH on the
// contract (two agreeing PS4 references, PS4-inherited surface; PS5 3.20 exports the same names).
#ifndef _WIN32
namespace {
struct NpStateCbSlot { std::atomic<uint64_t> fn{0}, arg{0}; std::atomic<int> delivered{0}; };
NpStateCbSlot     g_np_state_cbs[4];
std::atomic<int>  g_np_state_cb_n{0};
}
HLE(s_np_register_state_cbA) {   // (SceNpStateCallbackA func, void* userdata) -> callback id
    svc_log("sceNpRegisterStateCallbackA", a0,a1,a2,a3,a4,a5);
    if (!a0) return 0x80550003ull;   // SCE_NP_ERROR_INVALID_ARGUMENT
    int i = g_np_state_cb_n.fetch_add(1);
    if (i >= 4) { g_np_state_cb_n.store(4); return 0x8055001Dull; }  // SCE_NP_ERROR_CALLBACK_MAX
    g_np_state_cbs[i].arg.store(a1);
    g_np_state_cbs[i].fn.store(a0);
    return (uint64_t)(i + 1);
}
extern "C" uint64_t s_np_check_cb_c(uint64_t a0, uint64_t a1, uint64_t a2,
                                    uint64_t a3, uint64_t a4, uint64_t a5,
                                    uint64_t entry_rsp);
PROSPER_ASM_TRAMPOLINE(s_np_check_cb_entry, s_np_check_cb_c)
extern "C" void s_np_check_cb_entry();
extern "C" uint64_t s_np_check_cb_c(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                    uint64_t entry_rsp) {
    int n = g_np_state_cb_n.load(); if (n > 4) n = 4;
    uint64_t gfs = callback_guest_fs_from_entry_stack(entry_rsp);
    for (int i = 0; i < n; i++) {
        uint64_t fn = g_np_state_cbs[i].fn.load();
        if (!fn || g_np_state_cbs[i].delivered.exchange(1)) continue;
        {
            CbGuestFsScope fs(gfs);
            ((void (*)(int32_t, int32_t, void*))(uintptr_t)fn)(
                1 /*initial userId (sceUserServiceGetInitialUser)*/, 1 /*SCE_NP_STATE_SIGNED_OUT*/,
                (void*)(uintptr_t)g_np_state_cbs[i].arg.load());
        }
        // Log only on the restored host %fs (fprintf on the guest %fs faults in libc TLS).
        fprintf(stderr, "[svc] Np state callback DELIVERED (userId=1, state=SIGNED_OUT, guest_fs=%d)\n",
                gfs ? 1 : 0);
    }
    return 0;
}
#endif

// --- libSceErrorDialog: the real Initialize/Open/Close lifecycle (auto-dismiss, headless). ------
// Status enum shared with CommonDialog: NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3 (shadPS4
// commondialog.h + error_dialog.cpp; PS4-inherited, identical export NIDs on PS5 3.20).
// GetStatus/UpdateStatus RETURN the status value directly (not an out-param). DOLL pumps
// sceErrorDialogUpdateStatus once per frame from boot (housekeeping); if the honest offline chain
// makes it Open the "could not download patch data" dialog, auto-dismiss to FINISHED so the flow's
// "wait until dismissed" loop exits — the MsgDialog precedent (#144). Open's param is
// { s32 size; s32 errorCode; s32 userId; s32 reserved } (shadPS4 error_dialog.cpp Param); the
// errorCode is printed unconditionally — a one-shot, high-value diagnostic of WHAT the game
// thinks failed. A PlatformUi that accepts Open is pinned as the non-owning expected backend; every
// later callback obtains a registry lease for that exact pointer. Unregister/replacement abandons
// safely to headless FINISHED and can never reroute to a backend that did not accept the request.
// CONFIDENCE: HIGH (shadPS4 implements this exact lifecycle).
namespace {
std::atomic<int> g_errdialog_status{0 /*NONE*/};
std::atomic<PlatformUi*> g_errdialog_ui{nullptr};

void errdialog_close_owner() {
    PlatformUi* expected = g_errdialog_ui.exchange(nullptr);
    if (!expected) return;
    auto ui = platform_ui_lease(expected);
    if (ui) ui->errorDialogClose();
}

PlatformUiLease errdialog_owner_lease(PlatformUi*& expected) {
    expected = g_errdialog_ui.load(std::memory_order_acquire);
    if (!expected) return {};
    return platform_ui_lease(expected);
}

void errdialog_abandon_owner(PlatformUi* expected) {
    if (!expected || !g_errdialog_ui.compare_exchange_strong(expected, nullptr)) return;
    // Publish the bounded foreground-return phase before the fallback FINISHED status. A thread
    // that acquires FINISHED can therefore never observe the pre-dialog idle phase.
    finish_error_dialog_background();
    g_errdialog_status.store(3 /*FINISHED: safe headless fallback*/, std::memory_order_release);
}
}
HLE(s_errdialog_init)   {
    errdialog_close_owner();
    g_error_dialog_background_phase.store(ErrorDialogIdle);
    g_errdialog_status.store(1 /*INITIALIZED*/);
    return 0;
}
HLE(s_errdialog_open)   {
    errdialog_close_owner();
    g_error_dialog_background_phase.store(ErrorDialogIdle);
    // Offer the error dialog to a registered PlatformUi first (a real message box); else auto-dismiss.
    {
        auto ui = platform_ui_lease();
        if (ui) {
            begin_error_dialog_background(true);
            if (ui->errorDialogOpen(a0)) {
                g_errdialog_ui.store(ui.get(), std::memory_order_release);
                return 0;
            }
        }
    }
    uint32_t code = 0;
    if (svc_ptrish(a0)) code = *(const uint32_t*)((const char*)PW(a0) + 4);
    fprintf(stderr, "[svc] sceErrorDialogOpen(errorCode=%#x) -> auto-dismiss FINISHED\n", code);
    // Arm first: publishing FINISHED before this release lets another guest thread act on dialog
    // completion while SystemService still exposes the pre-dialog idle state.
    begin_error_dialog_background(false);
    g_errdialog_status.store(3 /*FINISHED (auto-dismiss)*/, std::memory_order_release);
    return 0;
}
HLE(s_errdialog_close)  {
    errdialog_close_owner();
    finish_error_dialog_background();
    g_errdialog_status.store(3 /*FINISHED*/);
    return 0;
}
HLE(s_errdialog_term)   {
    errdialog_close_owner();
    finish_error_dialog_background();
    g_errdialog_status.store(0 /*NONE*/);
    return 0;
}
HLE(s_errdialog_status) {
    PlatformUi* expected = nullptr;
    auto ui = errdialog_owner_lease(expected);
    if (expected && ui) {
        const int status = ui->errorDialogStatus();
        if (status == 0 /*NONE*/ || status == 3 /*FINISHED*/) finish_error_dialog_background();
        return (uint64_t)(unsigned)status;
    }
    if (expected) errdialog_abandon_owner(expected);
    return (uint64_t)(unsigned)g_errdialog_status.load(std::memory_order_acquire);
}

// --- libSceNpEntitlementAccess / libSceGameUpdate: observability first. -------------------------
// PS5-only surfaces with NO reference implementation (absent from Kyty and shadPS4); the PS5 3.20
// stub tables give names+NIDs only. DOLL calls sceNpEntitlementAccessInitialize (retried 3x) and
// sceGameUpdateInitialize once, then never the follow-ups (CreateRequest/Check) — so before
// inventing contracts, capture the REAL args live (PROSPER_SVCLOG=1) and keep the return the same
// success the unimplemented path produced (a real console's local library init succeeds offline
// too). The follow-up calls stay UNIMPLEMENTED deliberately: if the honest Np/NetCtl answers
// unblock the flow into them, they surface in the PROSPER_PROGRESS_UNIMPL dump and get pinned from
// their live args before being given behavior. CONFIDENCE: MED (init-succeeds is the real-console
// offline behavior; arg shapes intentionally not guessed).
HLE(s_npent_init)      { svc_log("sceNpEntitlementAccessInitialize", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_gameupdate_init) { svc_log("sceGameUpdateInitialize",          a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_gameupdate_term) { svc_log("sceGameUpdateTerminate",           a0,a1,a2,a3,a4,a5); return 0; }
// The entitlement follow-ups DOLL's main menu fires once the flow is unblocked (live-captured
// after the #306 gate fell). ABI pinned from the live capture (r8):
//   GetAddcontEntitlementInfoList(SceNpServiceLabel serviceLabel, Info* list, u32 listNum,
//                                 u32* hitNum)
// — the game first count-queries with list=NULL/num=0 and an out pointer in a3, then calls again
// with a 4-entry buffer (its four addcont slots). Matches the documented PS4 signature 1:1.
// Entries come only from the validated local installation inventory. A title with no dlc_emu.ini
// retains the retail no-DLC answer SUCCESS/hitNum=0; a present but invalid manifest fails visibly.
// CONFIDENCE: HIGH on the shape (two independent PS5 guests allocate 0x1c bytes per entry and Sonic
// directly consumes status at +0x18). Package enum values and errno behavior are producer-pinned.
HLE(s_npent_addcont_list) {
    svc_log("sceNpEntitlementAccessGetAddcontEntitlementInfoList", a0,a1,a2,a3,a4,a5);
    if (a0 > UINT32_MAX) return NP_ENTITLEMENT_ERROR_PARAMETER;
    return addcontent_info_list<NpAddcontEntitlementInfo>(
        a1, a2, a3, static_cast<uint32_t>(a0), NP_ENTITLEMENT_ERROR_PARAMETER,
        NP_ENTITLEMENT_ERROR_NO_ENTITLEMENT, NP_ENTITLEMENT_ADDCONT_LIST_MAX, npent_info);
}
// GetEntitlementKey(serviceLabel, const Label* label, Key* out) — live capture: retried 4x with
// identical args (the garbage-consuming retry signature). Crisis Core independently consumes
// exactly 16 bytes from the successful output. Validated installed records use either their explicit
// key or the producer's deterministic default; unknown labels retain honest failure.
HLE(s_npent_getkey) {
    svc_log("sceNpEntitlementAccessGetEntitlementKey", a0,a1,a2,a3,a4,a5);
    std::string label;
    if (a0 > UINT32_MAX || !appcontent_read_label(a1, label) || !svc_ptrish(a2))
        return NP_ENTITLEMENT_ERROR_PARAMETER;
    const AddcontentInventorySnapshot inventory = addcontent_inventory_snapshot();
    const InstalledAddcontent* entry = inventory.state == AddcontentInventoryState::Ready
        ? find_addcontent(inventory, static_cast<uint32_t>(a0), label) : nullptr;
    if (!entry) return NP_ENTITLEMENT_ERROR_NO_ENTITLEMENT;
    return svc_write_bytes(a2, entry->entitlement_key.data(), entry->entitlement_key.size())
        ? 0 : NP_ENTITLEMENT_ERROR_PARAMETER;
}

// sceNpEntitlementAccessGetSkuFlag(int32_t* skuFlag) — the SAME question
// sceAppContentAppParamGetInt answers for paramId 0, asked through libSceNpEntitlementAccess. It is a
// platform query about the SKU of the LOCALLY INSTALLED application, not a query about anything a
// user purchased from a network service, so it is answered from the same one local derivation and the
// two libraries cannot disagree.
//
// The NID was registered nowhere, so it fell to the dispatcher's unimplemented stub, which reports
// SUCCESS while leaving the out pointer untouched. That is worse than a wrong value: of the four
// titles in the project's local dumps that call this, three (PPSA07809, PPSA02154, PPSA05143) hand it
// an uninitialized stack slot and then act on whatever residue was there, and the fourth (PPSA04263)
// pre-seeds TRIAL and keeps it. Failing when the SKU is unknown is a path all four already handle —
// each falls back to its own conservative default — whereas reporting a SKU prosper cannot derive is
// a value the guest cannot tell from a real one.
//
// THE OUT POINTER LEADS (arg 0), and that needed proving rather than assuming: both neighbouring exports
// in this library (GetAddcontEntitlementInfoList, GetEntitlementKey) lead with a SceNpServiceLabel, so
// if this one did too, writing through a0 would be writing through a label and the whole fix would be
// inert with every test still green. A live GTA V (PPSA04263) boot under PROSPER_SVCLOG settles it — the
// second line is the load-bearing one, because it is an observation of the slot rather than a reading of
// guest code:
//   [svc] sceNpEntitlementAccessGetSkuFlag(0x7f4fc82d6e2c, 0, 0xb, 0, 0x1d, 0xb)
//   [svc]   a0 -> 0001a9bf00000001 0000003ab0ba4900 …
// a0 is a stack address, and the low dword at it is 0x00000001 — the guest's own pre-seeded TRIAL
// default, which is what the unimplemented stub used to leave standing. Only the SHAPE of the pointer
// carries the argument: 0x7f4f… is run-local under ASLR, so a later run will print different digits and
// that is not a disagreement.
// What this does NOT establish: whether a trailing SceNpServiceLabel parameter exists. An
// (out, label = 0) two-argument form would log byte-identically, and the sibling capture in the same run
// shows label 0 is exactly what these calls pass — GetAddcontEntitlementInfoList(0, 0, 0, 0x7f4f…, …).
// Unresolved and immaterial here: this handler ignores a1 either way.
//
// CONFIDENCE: HIGH on the contract and the enum (guest-pinned in three titles).
//
// LOW on the exact errno for the unknown-SKU case, and deliberately left admitted rather than dressed
// up. NO_ENTITLEMENT is a semantic stretch for "prosper cannot derive this application's SKU": it is a
// placeholder picked inside the producer-pinned 0x817D NpEntitlementAccess facility, NOT a known
// contract, and a more precise-looking value invented here would be worse than the stretch, because
// the next reader would treat a specific code as evidence that the contract is known.
// What makes the placeholder safe: no caller's behaviour depends on the value. All four titles in the
// project's local dumps test only for non-zero and fall back to their own conservative default, so the
// observable behaviour is identical for any error in this facility.
// What would settle it, and let this be replaced without re-deriving anything above: a title observed
// branching on a specific error code from this call, or the value observed from real firmware.
HLE(s_npent_skuflag) {
    svc_log("sceNpEntitlementAccessGetSkuFlag", a0,a1,a2,a3,a4,a5);
    if (!svc_ptrish(a0)) return NP_ENTITLEMENT_ERROR_PARAMETER;
    const AppParamDeclaration decl = app_param_declaration();
    if (!decl.declared || !decl.sku_flag) return NP_ENTITLEMENT_ERROR_NO_ENTITLEMENT;
    const int32_t value = (int32_t)*decl.sku_flag;
    return svc_write_bytes(a0, &value, sizeof(value)) ? 0 : NP_ENTITLEMENT_ERROR_PARAMETER;
}

// sceSaveDataTransferringMount (PS5-only, live-captured x4 in DOLL's save-slot menu): mount a
// PS4-era save for transfer/import. We have no PS4 save to transfer — the truthful fresh-console
// answer is NOT_FOUND with the result untouched. The previous success+garbage made the game read
// an EMPTY mount-point string out of the unwritten result and open '/GameSaveData245.dat' (no
// mount prefix) at filesystem root — observed live. CONFIDENCE: MED on semantics (same 0x809F
// facility + NOT_FOUND as Mount3 of a nonexistent save), LOW on the exact arg layout (nothing is
// written, so no layout is assumed; PROSPER_SVCLOG captures it).
HLE(s_savedata_transfermount) {
    svc_log("sceSaveDataTransferringMount", a0,a1,a2,a3,a4,a5);
    return SAVE_DATA_ERR_NOT_FOUND;
}

// sceSaveDataTransferringMountPs4 — the SIBLING of the call above, and the one Sonic Frontiers
// (PPSA03831) uses. It mounts the **PS4 edition's** save data so a PS5 title can import it on first
// run. A PS5 installation that was never upgraded from a PS4 copy has no such save area, and that
// is every installation prosper can present: prosper has no PS4 save-data store at all, and no
// local dump carries one, so "no PS4 save data exists here" is derived from local inventory rather
// than assumed — the same NOT_FOUND a Mount3 of a nonexistent save returns, in the same 0x809F
// facility.
//
// Unregistered, this reached `prosper_on_unimpl`'s `return 0` — the FALSE SUCCESS class (#2081) —
// and it produced the *identical* downstream signature its sibling's comment records for DOLL.
// Frontiers zeroes its 32-byte mount-point result, calls this, is told the mount succeeded, and
// then formats "<mountPoint>/gamedata" out of the still-empty result and opens **`/gamedata`** at
// filesystem root. That open fails ENOENT, the title retries it once per frame forever, and the
// boot state machine never leaves GameModeInitialize: measured live, ~1,450 failed `/gamedata`
// opens and 1,319 dispatcher hits on this NID in one 60 s CPU-only arm. All five guest call sites
// gate on the result (`test eax,eax`, `tools/re/nid_gate_scan.py --nid RjMlsR8EXrw`), and the
// error arm returns straight out of the transfer check — so an honest error is a path the title
// already has, not one this invents.
//
// Nothing is written to the result: a caller that reads a mount point after an error is reading
// its own buffer, which is exactly what must not be papered over. CONFIDENCE: HIGH that success
// is wrong here (the guest's own use of the unwritten result is observed); MED on NOT_FOUND being
// the precise firmware errno, LOW on the argument layout (nothing is assumed — nothing is read).
HLE(s_savedata_transfermount_ps4) {
    svc_log("sceSaveDataTransferringMountPs4", a0,a1,a2,a3,a4,a5);
    return SAVE_DATA_ERR_NOT_FOUND;
}

// sceSaveDataDirNameSearchPs4 (X4MYzukPc3g) -- the PS4 sibling of sceSaveDataDirNameSearch
// (dyIhnXq-0SM, registered below), and with the entry point above the ONLY two PS4-namespace exports
// libSceSaveData has across all 275 PS5 3.20 libraries. Fixing it closes the pair (#2210).
//
// Unregistered, this reached prosper_on_unimpl's `return 0` -- SCE_OK for this contract, the FALSE
// SUCCESS class (#2081). The caller is told a search over PS4 save data SUCCEEDED and the result
// struct it passed is never written. That reads as "zero hits" only because callers happen to zero
// the struct first: right by accident, not by construction. A caller reusing a result struct, or one
// whose hit count lands on non-zero stack residue, is handed a count over memory nothing wrote --
// the #213 shape, where a garbage count sized a 34 GB array.
//
// NOT_FOUND is derived from local inventory, not assumed: prosper has no PS4 save-data store at all
// (its two save areas are both PS5-side -- PROSPER_SAVEDATA_DIR -> save-data-memory, PROSPER_SAVE0 ->
// the mounted /savedata0) and no local dump carries a PS4 save area, so a search over PS4 save data
// honestly finds nothing. Same 0x809F facility, same answer a search of a nonexistent save returns.
//
// NOTHING is written to the result, deliberately. Returning SCE_OK with an explicitly-zeroed hit
// count would also be defensible -- but only if the argument layout were established from live
// evidence, and it is not. Inventing a written result is the exact mirror of the defect being fixed
// (#2208's failure was a guest reading an unwritten result). A caller that reads a result after an
// error is reading its own buffer, which is what must not be papered over.
//
// CONFIDENCE: HIGH that success is wrong here (it is SCE_OK over an unwritten out-struct by
// construction); MED on NOT_FOUND being the precise firmware errno; LOW on the argument layout --
// nothing is assumed about it because nothing is read or written.
HLE(s_savedata_dirname_search_ps4) {
    svc_log("sceSaveDataDirNameSearchPs4", a0,a1,a2,a3,a4,a5);
    // #3124: answer this exactly as the PS5 sibling above does, with an EXPLICIT zero-hit result.
    //
    // #2302 replaced the dispatcher's `return 0` with NOT_FOUND, and its diagnosis was right: SCE_OK
    // over an out-struct nothing wrote is the false-success class, correct only by accident when the
    // caller happens to pre-zero. But it changed the answer for a call it had not seen, and said so:
    // "Not observed being CALLED at any boot depth reached so far". Tactics Ogre: Reborn (PPSA03839)
    // calls it, once, right after sceSaveDataInitialize3 -- and on NOT_FOUND the title submits two
    // DCBs, draws once and then stops submitting while staying alive, rendering one black frame for
    // the rest of the run. Bisected over 700 commits.
    //
    // The commit named the better option and declined it only for want of evidence: "Returning
    // SCE_OK with an explicitly-zeroed hit count would also be defensible -- but only if the
    // argument layout were established from live evidence, and it is not."
    //
    // IT IS NOW, twice over. Captured with PROSPER_SVCLOG=1 on this title, the PS4 and PS5 calls
    // carry a byte-identical result struct -- [0x00]=0, [0x08]=caller buffer pointer, [0x10]=0x400
    // capacity -- and s_savedata_dirsearch above already writes that same layout (hitNum @0x00,
    // dirNames @0x08, dirNamesNum @0x10, setNum @0x14), derived from live evidence in #299.
    //
    // So this is not an invented result. Zero hits is the honest answer: prosper has no PS4 save
    // area at all -- both its save roots are PS5-side (PROSPER_SAVEDATA_DIR, PROSPER_SAVE0) -- so a
    // search over PS4 save data genuinely finds nothing. Writing the count makes that true BY
    // CONSTRUCTION rather than by the caller's habit of zeroing first, which is precisely what
    // #2302 objected to. On the observed caller the write is a no-op (the field is already 0); on
    // one that does not pre-zero it replaces stack residue, the #213 shape where a garbage count
    // sized a 34 GB array.
    //
    // The charter's rule also binds here: the same question must be answered the same way through
    // every library that exposes it. The PS5 spelling enumerates and reports its count; the PS4
    // spelling reporting a hard error for the same question was a divergence on top of the
    // regression.
    //
    // CONFIDENCE: HIGH that zero hits is the correct answer and that SCE_OK unblocks the title
    // (measured both ways). HIGH on the layout -- two independent captures plus the sibling's
    // established use of the same offsets.
    if (!a1) return 0x809F0000ull;             // SAVE_DATA_ERROR_PARAMETER, as the sibling does
    uint8_t* res = (uint8_t*)PW(a1);
    if (!res) return 0x809F0000ull;
    *(uint32_t*)(res + 0x00) = 0;              // hitNum
    *(uint32_t*)(res + 0x14) = 0;              // setNum
    return 0;
}

// sceSystemServiceGetNoticeScreenSkipFlag(bool* flag) — polled from DOLL's front-end menu.
// PS5-only (no reference). Live capture pinned the out-pointer to an ODD stack address
// (0x...ff307), so the flag is a single byte (bool), NOT an int32 — a 4-byte write would clobber
// 3 adjacent stack bytes. 0 = "no skip" is the inert default a retail console with no
// notice-screen state reports. CONFIDENCE: MED (byte-sized out pinned live; value semantics LOW).
HLE(s_syss_noticeskip) {
    svc_log("sceSystemServiceGetNoticeScreenSkipFlag", a0,a1,a2,a3,a4,a5);
    if (svc_ptrish(a0)) *(uint8_t*)PW(a0) = 0;
    return 0;
}

// ===== libSceRandom ============================================================================
//
// `sceRandomGetRandomNumber(void* buf, size_t size)` is the library's ONLY export (PS5 3.20
// `libSceRandom.c` lists exactly one `sprx_dlsym` line), and 27 of the 44 local dumps import it.
//
// Unregistered, it reached the dispatcher's `return 0` — and for this contract 0 is SCE_OK. The
// guest was told its buffer had been filled while nothing wrote to it, so it read back whatever
// its own stack or heap already held and used that as entropy. The worst property of that failure
// is that it is STABLE: a fresh stack allocation at the same call site tends to hold the same
// residue on every call, so a "random" session id or nonce can be identical all run. Nothing
// crashes and no diagnostic fires; only code that inspected the buffer could notice. (#2065, swept
// under #2081.)
//
// The fix is a real implementation rather than a fail-visible stub, because the honest answer here
// is cheap: the host has a CSPRNG. Returning an error instead would be strictly worse — this is a
// *value-producing* contract, and guests do gate on it.
//
// How the guests actually consume it, measured with `tools/re/nid_gate_scan.py --nid PI7jIZj4pcE`
// over the local dumps (call sites classified by what happens to eax):
//
//     PPSA08804  nonzero=2 ignored=1        PPSA04263  nonzero=2
//     PPSA24651  nonzero=1                  PPSA13579  nonzero=1
//     PPSA17942  ignored=1                  PPSA19244  forward=1 ignored=1
//
// So most call sites branch on "did this fail", and — the load-bearing part — NOT ONE compares the
// result against a specific constant (no `const` / `other-cmp` site anywhere). That is what makes
// the reject arms safe to add: the guest reads only the SIGN of the answer, never its value.
// CONFIDENCE: HIGH that success must fill the buffer and that failure must be non-zero;
// LOW on the exact error constant — the SCE_RANDOM error space is not in the 3.20 dump (which
// carries names and NIDs only) and no call site discriminates it, so the libkernel encoding is
// used per the project's default rather than an invented SCE_RANDOM_* value.
namespace {

// The published single-request cap. A guest asking for more than this gets an error on hardware, so
// prosper must not quietly serve it: a partial fill reported as success would recreate the exact
// bug this handler exists to remove, and an over-long fill would paper over a guest bug that real
// hardware rejects.
//
// CONFIDENCE: MED — the cap is from the API documentation, not from a live capture. This is also
// the ONLY arm of this handler that can newly FAIL a call the previous stub "succeeded", so it is
// the one place a wrong constant costs something rather than merely being imprecise.
//
// What is NOT established: the request sizes local titles actually pass. `nid_gate_scan` classifies
// what the guest does with `eax` and cannot recover an argument, so no instrument here has measured
// the `size` operand at any call site — deliberately stated rather than left as an implied "we
// checked". Settling it needs the argument registers read at the call, e.g. PROSPER_SVCLOG-style
// logging on this NID or a hardware breakpoint at the import. If a title is ever found requesting
// more, this constant is where to look first.
constexpr uint64_t kRandomMaxBytes = 64;

// Fill `bytes` from the host CSPRNG. Returns false if the host cannot supply entropy — which the
// caller MUST surface as an error, never as a zero-filled success. Deterministic zeros presented as
// random are the same lie in a different costume.
bool svc_host_entropy(void* dst, size_t bytes) {
#ifdef _WIN32
    return BCryptGenRandom(nullptr, (PUCHAR)dst, (ULONG)bytes,
                           BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;
#else
    auto* p = static_cast<unsigned char*>(dst);
    while (bytes) {
        const size_t chunk = bytes < 256 ? bytes : 256;   // getentropy's published maximum
        if (getentropy(p, chunk) != 0) return false;
        p += chunk;
        bytes -= chunk;
    }
    return true;
#endif
}

} // namespace

HLE(s_random_get_random_number) {
    (void)a2; (void)a3; (void)a4; (void)a5;
    const uint64_t buf = a0, size = a1;
    // A zero-length request asks for nothing and trivially gets it; there is no buffer to leave
    // stale, so this is a real success rather than the empty kind.
    if (size == 0) return 0;
    if (size > kRandomMaxBytes) return prosper::hle::kSceKernelErrorEINVAL;
    if (!svc_ptrish(buf)) return prosper::hle::kSceKernelErrorEFAULT;

    unsigned char tmp[kRandomMaxBytes];
    if (!svc_host_entropy(tmp, (size_t)size))
        return prosper::hle::sce_kernel_error(prosper::hle::FreeBsdErrno::EIo);
    // Fault-contained: an unmapped or unwritable guest buffer must return an error, not take down
    // the emulator, and must not report success for a write that did not land.
    if (!svc_write_bytes(buf, tmp, (size_t)size)) return prosper::hle::kSceKernelErrorEFAULT;
    return 0;
}

void register_service_hle() {
    register_json2_hle();
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
#ifndef _WIN32
    // NetCtl offline-console state delivery — default ON since #306 (see block comment above).
    // PROSPER_NETCTL_CB=0 restores the previous unimplemented behavior.
    {
        // DEFAULT ON since #306. `strtol` answered 0 for `=yes`/`=true`/`=on`, and the consequence
        // is not a lost diagnostic: the three callback NIDs guarded here (RegisterCallback,
        // CheckCallback, GetState) go UNREGISTERED, so the guest gets the pre-#306 unimplemented
        // behaviour from what the operator read as "enable it". GetInfo and GetResult below are
        // registered unconditionally and are unaffected (#3267).
        const char* e = getenv("PROSPER_NETCTL_CB");
        if (prosper::diag::env_u64_or_default_auto("PROSPER_NETCTL_CB", e, 1ull) != 0) {
            R("sceNetCtlRegisterCallback", s_netctl_register_cb);      // UJ+Z7Q+4ck0
            R("sceNetCtlCheckCallback",    s_netctl_check_cb_entry);   // iQw3iQPhvUQ
            R("sceNetCtlGetState",         s_netctl_getstate);         // uBPlr0lbuiI
        }
    }
#endif
    Hle::register_fn("obuxdTiwkF8", (HleFn)s_netctl_getinfo, "sceNetCtlGetInfo");  // NOT_CONNECTED
    Hle::register_fn("0cBgduPRR+M", (HleFn)s_netctl_getresult, "sceNetCtlGetResult");
    Hle::register_fn("dgJBaeJnGpo", (HleFn)s_net_pool_create, "sceNetPoolCreate");
    Hle::register_fn("hdpVEUDFW3s", (HleFn)s_ssl_init, "sceSslInit");
    Hle::register_fn("3JCe3lCbQ8A", (HleFn)s_http2_init, "sceHttp2Init");
    Hle::register_fn("+o9816YQhqQ", (HleFn)s_npweb_init, "sceNpWebApi2Initialize");
    Hle::register_fn("sk54bi6FtYM", (HleFn)s_npweb_create_user_context,
                     "sceNpWebApi2CreateUserContext");
    // NpTrophy2: the config/info queries whose success-with-garbage-out crashed DOLL (see above).
    // ALL FIVE of the library's info queries must answer here, not just the two that a title
    // happened to crash on. Each writes its result through a caller-supplied out-struct, so any one
    // of them left unregistered returns the dispatcher's 0 — SCE_OK — over memory nothing wrote,
    // which is the failure #213 diagnosed (a heap-garbage trophy count sized a 34 GB array). The
    // singular/plural pairs are the trap: registering `…TrophyInfoArray` and not `…TrophyInfo`
    // leaves the identical shape live behind a name that looks covered. #1956, swept under #2081.
    Hle::register_fn("4IzqhhUQ3nk", (HleFn)s_nptrophy2_unavailable, "sceNpTrophy2GetGameInfo");
    Hle::register_fn("y3zHpdZO6ME", (HleFn)s_nptrophy2_unavailable, "sceNpTrophy2GetTrophyInfoArray");
    Hle::register_fn("EwNylPdWUTM", (HleFn)s_nptrophy2_unavailable, "sceNpTrophy2GetTrophyInfo");
    Hle::register_fn("DoZWauG8mu0", (HleFn)s_nptrophy2_unavailable, "sceNpTrophy2GetGroupInfo");
    Hle::register_fn("+PDSI6WgPRc", (HleFn)s_nptrophy2_unavailable, "sceNpTrophy2GetGroupInfoArray");
    // libSceRandom's only export — see the block comment above s_random_get_random_number.
    Hle::register_fn("PI7jIZj4pcE", (HleFn)s_random_get_random_number, "sceRandomGetRandomNumber");
    // user service
    R("sceUserServiceGetInitialUser", s_user_initial);
    Hle::register_fn("eNb53LQJmIM", (HleFn)s_user_initial, "sceUserServiceGetForegroundUser");  // was MISSING -> garbage userId
    R("sceUserServiceGetEvent", s_user_getevent);   // deliver the initial-user LOGIN event once
    R("sceUserServiceGetLoginUserIdList", s_user_idlist);
    R("sceUserServiceGetUserName", s_user_name);
    R("sceUserServiceGetAccessibilityVibration", s_user_int_out);
    R("sceUserServiceGetAccessibilityPressAndHoldDelay", s_user_int_out);
    R("sceUserServiceGetAccessibilityZoomEnabled", s_user_int_out);
    // More (userId, int* out) getters the game queries at startup — same family: write a sane default
    // (accessibility off = 0; age level = adult) so the caller reads a deterministic value instead of
    // uninitialized stack. Registered by raw NID (guaranteed match; these names aren't in our NidDb).
    Hle::register_fn("woNpu+45RLk", (HleFn)s_user_age,     "sceUserServiceGetAgeLevel");
    Hle::register_fn("rnEhHqG-4xo", (HleFn)s_user_int_out, "sceUserServiceGetAccessibilityChatTranscription");
    Hle::register_fn("O6IW1-Dwm-w", (HleFn)s_user_int_out, "sceUserServiceGetAccessibilityZoomFollowFocus");
    Hle::register_fn("-3Y5GO+-i78", (HleFn)s_user_int_out, "sceUserServiceGetAccessibilityTriggerEffect");
    // sceUserServiceGetGamePresets (-sD02mFDBh4): returns 0 with a zeroed payload (see s_gamepresets).
    // History: it once returned 0x80960006 to avoid the game reading an untouched garbage struct — but
    // that non-zero errno made the Unity engine's per-controller check disconnect the pad every frame
    // and killed ALL gamepad input (#234). s_gamepresets now zeroes the payload after the caller-set
    // size field and returns success, satisfying both concerns.
    Hle::register_fn("-sD02mFDBh4", (HleFn)s_gamepresets, "sceUserServiceGetGamePresets");
    Hle::register_fn("qbwy0Ub8b3M", (HleFn)s_user_number, "sceUserServiceGetUserNumber");
    // Sonic imports LoginDialog only to initialize the service at startup.  There is no UI to show
    // until Open is requested, so initialization is a truthful successful no-op in the headless HLE.
    Hle::register_fn("qP-EvQRl2Hc", (HleFn)s_ok, "sceLoginDialogInitialize");
    Hle::register_fn("RnDibcGCPKw", (HleFn)s_videodec2_query_compute_memory,
                     "sceVideodec2QueryComputeMemoryInfo");
    Hle::register_fn("eD+X2SmxUt4", (HleFn)s_videodec2_allocate_compute_queue,
                     "sceVideodec2AllocateComputeQueue");
    Hle::register_fn("UvtA3FAiF4Y", (HleFn)s_videodec2_release_compute_queue,
                     "sceVideodec2ReleaseComputeQueue");
    Hle::register_fn("qqMCwlULR+E", (HleFn)s_videodec2_query_decoder_memory,
                     "sceVideodec2QueryDecoderMemoryInfo");
    Hle::register_fn("CNNRoRYd8XI", (HleFn)s_videodec2_create_decoder, "sceVideodec2CreateDecoder");
    Hle::register_fn("jwImxXRGSKA", (HleFn)s_videodec2_delete_decoder, "sceVideodec2DeleteDecoder");
    Hle::register_fn("852F5+q6+iM", (HleFn)s_videodec2_decode, "sceVideodec2Decode");
    Hle::register_fn("l1hXwscLuCY", (HleFn)s_videodec2_flush, "sceVideodec2Flush");
    Hle::register_fn("wJXikG6QFN8", (HleFn)s_videodec2_reset, "sceVideodec2Reset");
    // #1658: kjrLbcyhEiw is sceVideodec2GetAvcPictureInfo, not a second NID for the generic form —
    // the 3.20 firmware database names them separately. They still share a handler because no title
    // evidence yet shows the AVC output layout differs; the handler says so and captures the first
    // mismatch rather than assuming they are identical.
    Hle::register_fn("NtXRa3dRzU0", (HleFn)s_videodec2_picture_info, "sceVideodec2GetPictureInfo");
    Hle::register_fn("kjrLbcyhEiw", (HleFn)s_videodec2_avc_picture_info,
                     "sceVideodec2GetAvcPictureInfo");
    R("sceUserServiceInitialize", s_ok);
    R("sceUserServiceTerminate", s_ok);
    // NP — an honest signed-out console (#306). NIDs verified against the PS5 3.20
    // libSceNpManager stub table AND shadPS4's PS4 registrations (identical).
    R("sceNpGetState", s_np_state);
    R("sceNpGetNpReachabilityState", s_np_reach);
    R("sceNpGetAccountIdA", s_np_accountid);
    R("sceNpGetAccountCountryA", s_np_country);
    Hle::register_fn("XDncXQIJUSk", (HleFn)s_np_getonlineid, "sceNpGetOnlineId");
    Hle::register_fn("p-o74CnoNzY", (HleFn)s_np_getnpid,   "sceNpGetNpId");       // was MISSING -> faked identity
    Hle::register_fn("2rsFmlGWleQ", (HleFn)s_np_check_avail, "sceNpCheckNpAvailability");   // was MISSING -> faked "available"
    Hle::register_fn("8Z2Jc5GvGDI", (HleFn)s_np_check_avail, "sceNpCheckNpAvailabilityA");
    Hle::register_fn("KfGZg2y73oM", (HleFn)s_np_check_avail, "sceNpCheckNpReachability");
    Hle::register_fn("Oad3rvY-NJQ", (HleFn)s_np_has_signed_up, "sceNpHasSignedUp");
    Hle::register_fn("a8R9-75u4iM", (HleFn)s_np_accountid, "sceNpGetAccountId");  // non-A variant: zero id + SIGNED_OUT
    R("sceNpRegisterStateCallback", s_ok);
#ifndef _WIN32
    // sceNpCheckCallback pumps the registered A-callbacks (SIGNED_OUT delivered once, guest %fs).
    Hle::register_fn("3Zl8BePTh9Y", (HleFn)s_np_check_cb_entry,      "sceNpCheckCallback");
    Hle::register_fn("qQJfO8HAiaY", (HleFn)s_np_register_state_cbA,  "sceNpRegisterStateCallbackA");
    Hle::register_fn("M3wFXbYQtAA", (HleFn)s_ok,                     "sceNpUnregisterStateCallbackA");
#else
    R("sceNpCheckCallback", s_ok);
#endif
    // pad -> hle_pad.cpp (register_pad_hle). mouse:
    R("sceMouseInit", s_ok);
    R("sceMouseOpen", s_open);
    R("sceMouseRead", s_mouse_read);
    // app content / dialogs
    R("sceAppContentInitialize", s_ok);
    R("sceAppContentAppParamGetInt", s_appcontent_int);
    // temp-data mount: raw NIDs (names not in our NidDb). Unmount = OK (nothing to tear down).
    Hle::register_fn("buYbeLOGWmA", (HleFn)s_appcontent_tmpmount2, "sceAppContentTemporaryDataMount2");
    Hle::register_fn("SaKib2Ug0yI", (HleFn)s_appcontent_tmpspace, "sceAppContentTemporaryDataGetAvailableSpaceKb");
    Hle::register_fn("Gl6w5i0JokY", (HleFn)s_appcontent_tmpspace, "sceAppContentDownloadDataGetAvailableSpaceKb");  // was MISSING -> garbage KB
    Hle::register_fn("bcolXMmp6qQ", (HleFn)s_ok,                  "sceAppContentTemporaryDataUnmount");
    // Add-content (DLC) enumeration/mount comes only from the validated installed inventory.
    // No manifest retains no-DLC truth (hitNum=0 / no entitlement), not garbage-count OOM (#213).
    Hle::register_fn("xnd8BJzAxmk", (HleFn)s_appcontent_addcont_list,   "sceAppContentGetAddcontInfoList");
    Hle::register_fn("m47juOmH0VE", (HleFn)s_appcontent_addcont_info,   "sceAppContentGetAddcontInfo");
    Hle::register_fn("XTWR0UXvcgs", (HleFn)s_appcontent_entitlement_key, "sceAppContentGetEntitlementKey");
    Hle::register_fn("VANhIWcqYak", (HleFn)s_appcontent_addcont_mount, "sceAppContentAddcontMount");
    // The inverse. Unregistered it answered SCE_OK without releasing the claim, so a
    // mount/unmount/re-mount cycle was permanently BUSY (#2004, swept under #2081).
    Hle::register_fn("3rHWaV-1KC4", (HleFn)s_appcontent_addcont_unmount, "sceAppContentAddcontUnmount");
    R("sceCommonDialogInitialize", s_ok);
    R("sceCommonDialogIsUsed", s_ok);   // 0 = not in use (our dialogs auto-dismiss) - intentional, not an unimpl log
    R("sceSystemServiceParamGetInt", s_syss_param_int);   // language-aware (US English default), not blanket 0
    // sceSystemServiceParamGetString (SsC-m-S9JTA): write a valid empty string (not an unfilled buffer).
    Hle::register_fn("SsC-m-S9JTA", (HleFn)s_param_string, "sceSystemServiceParamGetString");
    // message dialog: track the Initialize/Open/Close lifecycle (#144) — NONE/INITIALIZED before an
    // Open, then auto-dismiss to FINISHED so the startup dialog flow still completes headless.
    R("sceMsgDialogInitialize", s_dialog_initialize);  R("sceMsgDialogTerminate", s_dialog_terminate);
    R("sceMsgDialogOpen", s_dialog_open);              R("sceMsgDialogClose", s_dialog_close);
    R("sceMsgDialogUpdateStatus", s_dialog_status);
    R("sceMsgDialogGetStatus", s_dialog_status);
    R("sceMsgDialogGetResult", s_dialog_result);
    // libSceSaveDataDialog[.native] uses raw NIDs in current dumps. Register the complete lifecycle
    // so future modes do not fall back to success-without-state even though Dead Cells currently uses
    // only Initialize/Open/UpdateStatus (#768).
    Hle::register_fn("fH46Lag88XY", (HleFn)s_savedlg_close,      "sceSaveDataDialogClose");
    Hle::register_fn("yEiJ-qqr6Cg", (HleFn)s_savedlg_result,     "sceSaveDataDialogGetResult");
    Hle::register_fn("ERKzksauAJA", (HleFn)s_savedlg_status,     "sceSaveDataDialogGetStatus");
    Hle::register_fn("s9e3+YpRnzw", (HleFn)s_savedlg_initialize, "sceSaveDataDialogInitialize");
    Hle::register_fn("en7gNVnh878", (HleFn)s_savedlg_ready,      "sceSaveDataDialogIsReadyToDisplay");
    Hle::register_fn("4tPhsP6FpDI", (HleFn)s_savedlg_open,       "sceSaveDataDialogOpen");
    Hle::register_fn("V-uEeFKARJU", (HleFn)s_savedlg_progress_inc, "sceSaveDataDialogProgressBarInc");
    Hle::register_fn("hay1CfTmLyA", (HleFn)s_savedlg_progress_set, "sceSaveDataDialogProgressBarSetValue");
    Hle::register_fn("YuH2FA7azqQ", (HleFn)s_savedlg_terminate,  "sceSaveDataDialogTerminate");
    Hle::register_fn("KK3Bdg1RWK0", (HleFn)s_savedlg_status,     "sceSaveDataDialogUpdateStatus");
    // libSceImeDialog (#191): auto-completing text-entry dialog (no keyboard UI). Raw NIDs.
    Hle::register_fn("NUeBrN7hzf0", (HleFn)s_imedlg_init,   "sceImeDialogInit");
    Hle::register_fn("IADmD4tScBY", (HleFn)s_imedlg_status, "sceImeDialogGetStatus");
    Hle::register_fn("x01jxu+vxlc", (HleFn)s_imedlg_result, "sceImeDialogGetResult");
    Hle::register_fn("gyTyVn+bXMw", (HleFn)s_imedlg_term,   "sceImeDialogTerm");
    Hle::register_fn("oBmw4xrmfKs", (HleFn)s_imedlg_abort,  "sceImeDialogAbort");
    R("sceSystemServiceHideSplashScreen", s_ok);
    // libSceIme keyboard API (#186): no physical keyboard -> consistent "none connected" state. Raw NIDs.
    Hle::register_fn("eaFXjfJv3xs", (HleFn)s_ime_kbd_open,  "sceImeKeyboardOpen");
    Hle::register_fn("PMVehSlfZ94", (HleFn)s_ime_kbd_close, "sceImeKeyboardClose");
#ifdef _WIN32
    Hle::register_fn("-4GCfYdNF1s", (HleFn)s_ime_update, "sceImeUpdate");         // plain handler (guest_fs=0)
#else
    Hle::register_fn("-4GCfYdNF1s", (HleFn)s_ime_update_entry, "sceImeUpdate");   // entry-rsp trampoline (#1286)
#endif
    Hle::register_fn("VkqLPArfFdc", (HleFn)s_ime_kbd_info,  "sceImeKeyboardGetInfo");
    Hle::register_fn("dKadqZFgKKQ", (HleFn)s_ime_kbd_resid, "sceImeKeyboardGetResourceId");
    // libSceAvPlayer (#324): let a post-credits / intro video complete so the game reaches its scene.
    // Init/InitEx must return a non-NULL handle; IsActive must report finished; the rest succeed as no-ops.
    R("sceAvPlayerInit",           s_avplayer_init);
    R("sceAvPlayerInitEx",         s_avplayer_initex);
    R("sceAvPlayerPostInit",       s_avp_postinit);
    R("sceAvPlayerSetLogCallback", s_avp_setlogcb);
#ifdef _WIN32
    R("sceAvPlayerAddSource",      s_avp_addsource);
    R("sceAvPlayerAddSourceEx",    s_avp_addsourceex);
    R("sceAvPlayerStart",          s_avp_start);
    R("sceAvPlayerIsActive",       s_avplayer_isactive);
    R("sceAvPlayerGetVideoData",   s_avp_getvideodata);
    R("sceAvPlayerGetVideoDataEx", s_avp_getvideodataex);
#else
    R("sceAvPlayerAddSource",      s_avp_addsource_entry);
    R("sceAvPlayerAddSourceEx",    s_avp_addsourceex_entry);
    R("sceAvPlayerStart",          s_avp_start_entry);
    R("sceAvPlayerIsActive",       s_avplayer_isactive_entry);
    R("sceAvPlayerGetVideoData",   s_avp_getvideodata_entry);
    R("sceAvPlayerGetVideoDataEx", s_avp_getvideodataex_entry);
#endif
    R("sceAvPlayerGetAudioData",   s_avp_getaudiodata);
#ifdef _WIN32
    R("sceAvPlayerStop",           s_avp_stop);
    R("sceAvPlayerClose",          s_avp_close);
#else
    R("sceAvPlayerStop",           s_avp_stop_entry);
    R("sceAvPlayerClose",          s_avp_close_entry);
#endif
    // PS5 raw NIDs verified against the 3.20 import stubs. Astro Bot enumerates streams before it
    // consumes GetVideoDataEx; returning the generic unresolved-import zero meant "no streams" and
    // left its logo movie state machine permanently resident even after synthetic EOF.
    Hle::register_fn("hdTyRzCXQeQ", (HleFn)s_avp_streamcount,   "sceAvPlayerStreamCount");
    Hle::register_fn("wwM99gjFf1Y", (HleFn)s_avp_current_time,  "sceAvPlayerCurrentTime");
    Hle::register_fn("d8FcbzfAdQw", (HleFn)s_avp_getstreaminfo, "sceAvPlayerGetStreamInfo");
    Hle::register_fn("ctTAcF5DiKQ", (HleFn)s_avp_getstreaminfoex, "sceAvPlayerGetStreamInfoEx");
    Hle::register_fn("ODJK2sn9w4A", (HleFn)s_avp_stream_ok, "sceAvPlayerEnableStream");
    Hle::register_fn("BOVKAzRmuTQ", (HleFn)s_avp_stream_ok, "sceAvPlayerDisableStream");
    Hle::register_fn("buMCiJftcfw", (HleFn)s_avp_stream_ok, "sceAvPlayerChangeStream");
#ifdef _WIN32
    Hle::register_fn("9y5v+fGN4Wk", (HleFn)s_avp_pause, "sceAvPlayerPause");
    Hle::register_fn("w5moABNwnRY", (HleFn)s_avp_resume, "sceAvPlayerResume");
#else
    Hle::register_fn("9y5v+fGN4Wk", (HleFn)s_avp_pause_entry, "sceAvPlayerPause");
    Hle::register_fn("w5moABNwnRY", (HleFn)s_avp_resume_entry, "sceAvPlayerResume");
#endif
    // sceAvPlayerJumpToTime (#1949). Unregistered, its dispatcher default of 0 was read as a
    // completed seek and deadlocked PPSA30490's Unity video splash; the handler now performs a real
    // backend seek and returns a real error when it cannot. NID verified in the PS5 3.20
    // libSceAvPlayer stub table. Calls no guest code, so it needs no entry-rsp trampoline.
    Hle::register_fn("XC9wM+xULz8", (HleFn)s_avp_jumptotime, "sceAvPlayerJumpToTime");
    Hle::register_fn("k-q+xOxdc3E", (HleFn)s_avp_stream_ok, "sceAvPlayerSetAvSyncMode");
    Hle::register_fn("OVths0xGfho", (HleFn)s_avp_stream_ok, "sceAvPlayerSetLooping");
    R("sceSystemServiceGetStatus", s_syss_getstatus);
    R("sceSystemServiceReceiveEvent", s_sysservice_receiveevent);   // NO_EVENT, don't leave the struct garbage
    // sceSystemServiceGetDisplaySafeAreaInfo (1n37q1Bvc5Y) — fill ratio=1.0 (see s_syss_safearea).
    Hle::register_fn("1n37q1Bvc5Y", (HleFn)s_syss_safearea, "sceSystemServiceGetDisplaySafeAreaInfo");
    Hle::register_fn("mPpPxv5CZt4", (HleFn)s_syss_hdr_luminance,
                     "sceSystemServiceGetHdrToneMapLuminance");
    // #3119: the guest reporting its OWN crash must not look like a hang. See the handler.
    Hle::register_fn("3s8cHiCBKBE", (HleFn)s_syss_report_abnormal_termination,
                     "sceSystemServiceReportAbnormalTermination");

    // ---- Issue #232 services (raw NIDs; every pair verified against the PS5 3.20 stub tables) ----
    // libScePlayGo — everything installed & locus-local.
    Hle::register_fn("ts6GlZOKRrE", (HleFn)s_playgo_init,        "scePlayGoInitialize");
    Hle::register_fn("MPe0EeBGM-E", (HleFn)s_playgo_term,        "scePlayGoTerminate");
    Hle::register_fn("M1Gma1ocrGE", (HleFn)s_playgo_open,        "scePlayGoOpen");
    Hle::register_fn("Uco1I0dlDi8", (HleFn)s_playgo_close,       "scePlayGoClose");
    Hle::register_fn("uWIYLFkkwqk", (HleFn)s_playgo_getlocus,    "scePlayGoGetLocus");
    Hle::register_fn("-RJWNMK3fC8", (HleFn)s_playgo_getprogress, "scePlayGoGetProgress");
    Hle::register_fn("Nn7zKwnA5q0", (HleFn)s_playgo_gettodo,     "scePlayGoGetToDoList");
    Hle::register_fn("gUPGiOQ1tmQ", (HleFn)s_playgo_settodo,     "scePlayGoSetToDoList");
    Hle::register_fn("73fF1MFU8hA", (HleFn)s_playgo_getchunkid,  "scePlayGoGetChunkId");
    Hle::register_fn("v6EZ-YWRdMs", (HleFn)s_playgo_geteta,      "scePlayGoGetEta");
    Hle::register_fn("rvBSfTimejE", (HleFn)s_playgo_getspeed,    "scePlayGoGetInstallSpeed");
    Hle::register_fn("4AAcTU9R3XM", (HleFn)s_ok,                 "scePlayGoSetInstallSpeed");
    Hle::register_fn("3OMbYZBaa50", (HleFn)s_playgo_getlang,     "scePlayGoGetLanguageMask");
    Hle::register_fn("LosLlHOpNqQ", (HleFn)s_ok,                 "scePlayGoSetLanguageMask");
    Hle::register_fn("-Q1-u1a7p0g", (HleFn)s_ok,                 "scePlayGoPrefetch");
    // libSceSaveData (PS5 native surface) — fresh console: mount of a nonexistent save NOT_FOUND.
    Hle::register_fn("TywrFKCoLGY", (HleFn)s_savedata_init3,     "sceSaveDataInitialize3");
    // libSceSaveData "save-data memory" API (#191): a real per-(user,slot) memory block round-trip.
    Hle::register_fn("oQySEUfgXRA", (HleFn)s_savemem_setup, "sceSaveDataSetupSaveDataMemory2");
    Hle::register_fn("cduy9v4YmT4", (HleFn)s_savemem_set,   "sceSaveDataSetSaveDataMemory2");
    Hle::register_fn("QwOO7vegnV8", (HleFn)s_savemem_get,   "sceSaveDataGetSaveDataMemory2");
    Hle::register_fn("v7AAAMo0Lz4", (HleFn)s_savemem_setup_v1, "sceSaveDataSetupSaveDataMemory");
    Hle::register_fn("h3YURzXGSVQ", (HleFn)s_savemem_set_v1,   "sceSaveDataSetSaveDataMemory");
    Hle::register_fn("7Bt5pBC-Aco", (HleFn)s_savemem_get_v1,   "sceSaveDataGetSaveDataMemory");
    Hle::register_fn("wiT9jeC7xPw", (HleFn)s_savemem_sync,  "sceSaveDataSyncSaveDataMemory");
    Hle::register_fn("yKDy8S5yLA0", (HleFn)s_savedata_term,      "sceSaveDataTerminate");
    Hle::register_fn("gjRZNnw0JPE", (HleFn)s_savedata_txres,     "sceSaveDataCreateTransactionResource");
    Hle::register_fn("lJUQuaKqoKY", (HleFn)s_savedata_txres_del, "sceSaveDataDeleteTransactionResource");
    Hle::register_fn("32HQAQdwM2o", (HleFn)s_savedata_mount,     "sceSaveDataMount");
    Hle::register_fn("0z45PIH+SNI", (HleFn)s_savedata_mount2,    "sceSaveDataMount2");
    Hle::register_fn("ZP4e7rlzOUk", (HleFn)s_savedata_mount3,    "sceSaveDataMount3");
    Hle::register_fn("BMR4F-Uek3E", (HleFn)s_savedata_umount,    "sceSaveDataUmount");
    Hle::register_fn("uW4vfTwMQVo", (HleFn)s_savedata_umount2,   "sceSaveDataUmount2");
    Hle::register_fn("sDCBrmc61XU", (HleFn)s_savedata_prepare,   "sceSaveDataPrepare");
    Hle::register_fn("ie7qhZ4X0Cc", (HleFn)s_savedata_commit,    "sceSaveDataCommit");
    Hle::register_fn("dyIhnXq-0SM", (HleFn)s_savedata_dirsearch, "sceSaveDataDirNameSearch");
    Hle::register_fn("65VH0Qaaz6s", (HleFn)s_savedata_mountinfo, "sceSaveDataGetMountInfo");  // was MISSING -> garbage free-space
    Hle::register_fn("j8xKtiFj0SY", (HleFn)s_savedata_get_event, "sceSaveDataGetEventResult");
    // libSceNpTrophy2 lifecycle — valid ids; content queries stay "unavailable" (above).
    Hle::register_fn("Bagshr7OQ6Q", (HleFn)s_nptrophy2_createctx,    "sceNpTrophy2CreateContext");
    Hle::register_fn("Gz1rmUZpROM", (HleFn)s_nptrophy2_createhandle, "sceNpTrophy2CreateHandle");
    Hle::register_fn("bIDov3wBu5Q", (HleFn)s_nptrophy2_regctx,       "sceNpTrophy2RegisterContext");
    Hle::register_fn("sUXGfNMalIo", (HleFn)s_nptrophy2_ok,           "sceNpTrophy2RegisterUnlockCallback");
    Hle::register_fn("sysY2FHYff4", (HleFn)s_nptrophy2_ok,           "sceNpTrophy2DestroyContext");
    Hle::register_fn("d8P11CI40KE", (HleFn)s_nptrophy2_ok,           "sceNpTrophy2DestroyHandle");
    Hle::register_fn("fYapWA9xVmA", (HleFn)s_nptrophy2_ok,           "sceNpTrophy2AbortHandle");
    // libSceShare — succeed; sharing simply unavailable headless.
    Hle::register_fn("nBDD66kiFW8", (HleFn)s_share_ok, "sceShareInitialize");
    Hle::register_fn("0IL1keINExQ", (HleFn)s_share_ok, "sceShareTerminate");
    Hle::register_fn("7QZtURYnXG4", (HleFn)s_share_content_param, "sceShareSetContentParam");
    Hle::register_fn("ORspsWDXPps", (HleFn)s_share_ok, "sceShareSetContentParamForApplicationTitle");
    Hle::register_fn("T64o-315wbg", (HleFn)s_share_ok, "sceShareSetScreenshotOverlayImage");
    Hle::register_fn("kvYEw2lBndk", (HleFn)s_live_streaming_init, "sceGameLiveStreamingInitialize");
    Hle::register_fn("9yK6Fk8mKOQ", (HleFn)s_share_ok, "sceGameLiveStreamingTerminate");
    // libSceErrorDialog — real lifecycle, auto-dismiss (#306). NIDs from the PS5 3.20 stub table
    // (identical to shadPS4's PS4 registrations).
    Hle::register_fn("I88KChlynSs", (HleFn)s_errdialog_init,   "sceErrorDialogInitialize");
    Hle::register_fn("M2ZF-ClLhgY", (HleFn)s_errdialog_open,   "sceErrorDialogOpen");
    Hle::register_fn("jrpnVQfJYgQ", (HleFn)s_errdialog_open,   "sceErrorDialogOpenDetail");
    Hle::register_fn("wktCiyWoDTI", (HleFn)s_errdialog_open,   "sceErrorDialogOpenWithReport");
    Hle::register_fn("ekXHb1kDBl0", (HleFn)s_errdialog_close,  "sceErrorDialogClose");
    Hle::register_fn("9XAxK2PMwk8", (HleFn)s_errdialog_term,   "sceErrorDialogTerminate");
    Hle::register_fn("t2FvHRXzgqk", (HleFn)s_errdialog_status, "sceErrorDialogGetStatus");
    Hle::register_fn("WWiGuh9XfgQ", (HleFn)s_errdialog_status, "sceErrorDialogUpdateStatus");
    // libSceNpEntitlementAccess / libSceGameUpdate — observability (svc_log) with the real-console
    // "local init succeeds offline" return; follow-ups deliberately left unimplemented (see above).
    Hle::register_fn("jO8DM8oyego", (HleFn)s_npent_init,      "sceNpEntitlementAccessInitialize");
    Hle::register_fn("YJtKLttI9fM", (HleFn)s_gameupdate_init, "sceGameUpdateInitialize");
    Hle::register_fn("NSH-C-OmoNI", (HleFn)s_gameupdate_term, "sceGameUpdateTerminate");
    // Post-gate follow-ups (fire from DOLL's now-reachable main menu; NIDs from PS5 3.20 tables).
    Hle::register_fn("TFyU+KFBv54", (HleFn)s_npent_addcont_list,
                     "sceNpEntitlementAccessGetAddcontEntitlementInfoList");
    Hle::register_fn("5LiMEPuW0DQ", (HleFn)s_npent_getkey, "sceNpEntitlementAccessGetEntitlementKey");
    Hle::register_fn("lPDO62PpJIA", (HleFn)s_npent_skuflag, "sceNpEntitlementAccessGetSkuFlag");
    Hle::register_fn("WAzWTZm1H+I", (HleFn)s_savedata_transfermount, "sceSaveDataTransferringMount");
    Hle::register_fn("RjMlsR8EXrw", (HleFn)s_savedata_transfermount_ps4, "sceSaveDataTransferringMountPs4");
    Hle::register_fn("X4MYzukPc3g", (HleFn)s_savedata_dirname_search_ps4, "sceSaveDataDirNameSearchPs4");
    Hle::register_fn("3RQ5aQfnstU", (HleFn)s_syss_noticeskip, "sceSystemServiceGetNoticeScreenSkipFlag");
    // libSceNpUniversalDataSystem — inert ids (guarded LOW-confidence out-writes).
    Hle::register_fn("sjaobBgqeB4", (HleFn)s_npuds_ok,     "sceNpUniversalDataSystemInitialize");
    Hle::register_fn("5zBnau1uIEo", (HleFn)s_npuds_create, "sceNpUniversalDataSystemCreateContext");
    Hle::register_fn("hT0IAEvN+M0", (HleFn)s_npuds_create, "sceNpUniversalDataSystemCreateHandle");
    Hle::register_fn("tpFJ8LIKvPw", (HleFn)s_npuds_ok,     "sceNpUniversalDataSystemRegisterContext");
    Hle::register_fn("p+GcLqwpL9M", (HleFn)s_npuds_create_event,
                     "sceNpUniversalDataSystemCreateEvent");
    Hle::register_fn("CzkKf7ahIyU", (HleFn)s_npuds_post_event,
                     "sceNpUniversalDataSystemPostEvent");
    Hle::register_fn("wG+84pnNIuo", (HleFn)s_npuds_destroy_event,
                     "sceNpUniversalDataSystemDestroyEvent");
    Hle::register_fn("MfDb+4Nln64", (HleFn)s_npuds_object_set_string,
                     "sceNpUniversalDataSystemEventPropertyObjectSetString");
    Hle::register_fn("m87BHxt-H60", (HleFn)s_gameintent_init,
                     "sceNpGameIntentInitialize");
    Hle::register_fn("0HBYxYAjmf0", (HleFn)s_gameintent_term,
                     "sceNpGameIntentTerminate");
    Hle::register_fn("jEIXUAr9XE8", (HleFn)s_gameintent_receive,
                     "sceNpGameIntentReceiveIntent");
    Hle::register_fn("rPl0INNc-M8", (HleFn)s_gameintent_get_property_string,
                     "sceNpGameIntentGetPropertyValueString");
    Hle::register_fn("xddD23+8TfQ", (HleFn)s_npent_addcont_info,
                     "sceNpEntitlementAccessGetAddcontEntitlementInfo");
    // libScePsml (PPSA20447): unregistered, these answered SCE_OK and left the caller's out-struct
    // untouched -- see s_psml_unimplemented. Names unknown: libScePsml is not in the 3.20 reference
    // set, so the raw NIDs are the only identity available and the label says exactly that.
    Hle::register_fn("3WVD91e12ZQ", (HleFn)s_psml_unimplemented, "libScePsml::3WVD91e12ZQ");
    Hle::register_fn("+2KpvixvL6E", (HleFn)s_psml_unimplemented, "libScePsml::+2KpvixvL6E");
    #undef R
}

} // namespace prosper
