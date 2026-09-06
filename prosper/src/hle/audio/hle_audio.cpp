// hle_audio.cpp — libSceAudioOut/AudioIn HLE, with pluggable output (see audio.hpp).
//
// Decodes the PS5 sceAudioOut* calls into port lifecycle + interleaved PCM grains and forwards
// them to the installed backend. The inherited AudioIn core provides deterministic paced silence.
// prosper_core stays dependency-free; a concrete output frontend (SDL3, ...) installs itself via
// audio_set_sink() from outside the core.
#include "hle/dispatch/dispatch.hpp"
#include "host/abi/sysv_ms_bridge.hpp"   // #3246: kLegacyForwardedArgs, the fixed prologue's capacity
#include "host/image/boot_program.hpp"   // #1659: shared guest-module labelling
#include "hle/dispatch/nid.hpp"
#include "hle/audio/audio.hpp"
#include "hle/audio/ajm_decoder.hpp"     // optional host codecs (MP3); core retains AJM ABI + guest copies
#include "hle/audio/atrac9_decode.hpp"    // vendored LibAtrac9 glue — AJM ATRAC9 batch decode (Blasphemous 2)
#include "hle/dispatch/callback_fs.hpp"      // recover the caller's guest %fs for firing guest callbacks
#include <memory>
#include "host/platform/posix_shim.hpp" // PROSPER_ASM_TRAMPOLINE (pass entry %rsp as 7th arg)
#include <algorithm>
#include <atomic>
#include <cmath>
#include <deque>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <sys/uio.h>
#include <unistd.h>
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include "host/platform/posix_shim.hpp"   // Darwin: process_vm_readv/writev

namespace prosper {

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define HLE8(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5, uint64_t a6, \
                                       uint64_t a7)
#define HLE10(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5, uint64_t a6, \
                                       uint64_t a7, uint64_t a8, uint64_t a9)
// HLE10 is the WIDEST handler shape in the tree, and that is a fact the Windows import stub depends
// on: its fixed integer prologue forwards exactly `kLegacyForwardedArgs` guest arguments and would
// silently drop anything beyond them (#3246, where the census was taken). Widening this macro — or
// adding an HLE11 anywhere — needs the prologue widened first, so the coupling is asserted here
// rather than left to be rediscovered on a platform nobody develops on.
static_assert(10 <= prosper::abi::kLegacyForwardedArgs,
              "HLE10 declares more arguments than the Windows import stub forwards");
#define P(x) ((void*)(uintptr_t)(x))

namespace {

constexpr int kMaxPorts  = 16;
// Host-only slots above the 16 public sceAudioOut handles keep each of the four AudioOut2
// contexts on an independent device stream. The host device mixes those streams concurrently;
// serializing multiple contexts through one paced stream would insert one context's grains into
// another context's timeline.
constexpr int kMaxSinkPorts = kMaxPorts + 4;
constexpr int kVolume0dB = 32768;   // SCE_AUDIO_VOLUME_0DB

struct Port {
    bool          in_use = false;
    int           type = 0;   // SceAudioOutPortType (0=MAIN, 1=BGM, 2=VOICE, 3=PERSONAL, 4=PADSPK, 127=AUX)
    AudioPortInfo info;
    int           vol[8] = { kVolume0dB, kVolume0dB, kVolume0dB, kVolume0dB,
                             kVolume0dB, kVolume0dB, kVolume0dB, kVolume0dB };
};

// SCE audio error codes the guest actually tests for (Kyty Errno.h:342/344). A generic -1 is
// unrecognizable to retry-vs-abort logic (a single wrong errno already caused a full render
// stall once — see hle_service.cpp's GetEvent note). Sign-extended: the guest ABI returns
// int32 in eax (negative), and host-side callers/tests compare the full u64 as int64.
constexpr uint64_t kAudioErrInvalidPort = (uint64_t)(int64_t)(int32_t)0x80260003;
constexpr uint64_t kAudioErrPortFull    = (uint64_t)(int64_t)(int32_t)0x80260005;

std::mutex g_mx;                 // guards the port table
Port       g_ports[kMaxPorts];

// --- default backend: silent, real-time paced (headless) --------------------------------
// sceAudioOutOutput on real hardware blocks until the audio ring has room, which paces the
// game's audio thread at real time. With no device attached we reproduce that pacing by
// sleeping each grain's wall-clock duration, so the guest advances at the correct speed.
struct RealtimeSilentSink : AudioSink {
    struct Pace { std::chrono::steady_clock::time_point next{}; long long ns_per_grain = 0; };
    Pace p_[kMaxSinkPorts];
    bool open(int port, const AudioPortInfo& info) override {
        if (port < 1 || port > kMaxSinkPorts) return false;
        int freq = info.freq > 0 ? info.freq : 48000;
        int grain = info.grain > 0 ? info.grain : 256;
        p_[port - 1].ns_per_grain = (long long)grain * 1000000000LL / freq;
        p_[port - 1].next = {};
        return true;
    }
    void output(int port, const void*, int frames) override {
        if (port < 1 || port > kMaxSinkPorts) return;
        auto& s = p_[port - 1];
        long long ns = s.ns_per_grain > 0 ? s.ns_per_grain : ((long long)frames * 1000000000LL / 48000);
        auto now = std::chrono::steady_clock::now();
        auto dur = std::chrono::nanoseconds(ns);
        // (Re)sync if unset or we fell far behind (e.g. after a stall) to avoid burst catch-up.
        if (s.next.time_since_epoch().count() == 0 || s.next < now - dur * 4) s.next = now;
        s.next += dur;
        if (s.next > now) std::this_thread::sleep_until(s.next);
    }
    void close(int port) override { if (port >= 1 && port <= kMaxSinkPorts) p_[port - 1] = {}; }
};

RealtimeSilentSink        g_default_sink;
std::atomic<AudioSink*>   g_sink{ &g_default_sink };

// Caller must hold g_mx.
Port* port_of(int handle) {
    if (handle < 1 || handle > kMaxPorts) return nullptr;
    Port& p = g_ports[handle - 1];
    return p.in_use ? &p : nullptr;
}

} // namespace

// --- public backend hooks (audio.hpp) ---------------------------------------------------
void audio_set_sink(AudioSink* sink) { g_sink.store(sink ? sink : &g_default_sink); }
AudioSink* audio_sink() { return g_sink.load(); }
static void audio_in_reset_ports();
static void audio2_reset();

void audio_decode_format(uint32_t param, int& channels, AudioFmt& fmt) {
    switch (param & 0xff) {                                   // SceAudioOutParamFormat (low byte)
        case 0: channels = 1; fmt = AudioFmt::S16; break;     // S16_MONO
        case 1: channels = 2; fmt = AudioFmt::S16; break;     // S16_STEREO
        case 2: channels = 8; fmt = AudioFmt::S16; break;     // S16_8CH
        case 3: channels = 1; fmt = AudioFmt::F32; break;     // FLOAT_MONO
        case 4: channels = 2; fmt = AudioFmt::F32; break;     // FLOAT_STEREO
        case 5: channels = 8; fmt = AudioFmt::F32; break;     // FLOAT_8CH
        case 6: channels = 8; fmt = AudioFmt::S16; break;     // S16_8CH_STD
        case 7: channels = 8; fmt = AudioFmt::F32; break;     // FLOAT_8CH_STD
        default: channels = 2; fmt = AudioFmt::S16; break;    // unknown -> S16 stereo
    }
}

void audio_apply_channel_volumes(int dst[8], uint32_t mask, const int* vols) {
    if (!dst || !vols) return;
    for (int c = 0; c < 8; c++)
        if (mask & (1u << c)) dst[c] = vols[c];
}

int audio_peak_channel_volume(uint32_t mask, const int* vols) {
    if (!vols) return 0;
    int peak = 0;
    for (int c = 0; c < 8; c++)
        if ((mask & (1u << c)) && vols[c] > peak) peak = vols[c];
    return peak;
}

bool audio2_reserve_queue_slot(uint32_t& queued, uint32_t queue_depth) {
    if (queued >= queue_depth) return false;
    ++queued;
    return true;
}

uint64_t audio_count_nonzero_samples(const void* pcm, std::size_t bytes, bool s16) {
    if (!pcm) return 0;
    uint64_t n = 0;
    if (s16) {
        const auto* s = static_cast<const int16_t*>(pcm);
        for (size_t i = 0, m = bytes / sizeof(int16_t); i < m; ++i)
            if (s[i] != 0) ++n;
    } else {
        const auto* s = static_cast<const float*>(pcm);
        // `!= 0.0f`, never a byte test: -0.0f compares equal to zero and has a non-zero bit
        // pattern, and NaN compares unequal to zero, which is the same treatment
        // AudioSignalStats gives both.
        for (size_t i = 0, m = bytes / sizeof(float); i < m; ++i)
            if (s[i] != 0.0f) ++n;
    }
    return n;
}

AudioGrainVerdict audio_classify_stamped_grain(bool same_address, bool matches_stamp,
                                               uint64_t nonzero_samples) {
    // Two properties are load-bearing, and a third that looks like it is, is not. Recorded because
    // the difference was established by mutation rather than by reading, and reading had it wrong:
    //   1. The address check must precede both content tests. A stamp read back from a DIFFERENT
    //      buffer says nothing about either one, and the double-buffered case is otherwise reported
    //      as a clean verdict about a buffer nobody looked at.
    //   2. The identity test must EXIST. The stamp is deliberately non-zero in every sample, so a
    //      surviving stamp carries a large non-zero count; drop this line and "prosper is reading
    //      memory the guest never writes" is reported as "the port is healthy" — the exact inversion
    //      the probe exists to prevent.
    //   3. Whether identity is tested BEFORE or AFTER emptiness is immaterial, and the first version
    //      of this comment claimed otherwise. Swapping them is behaviour-preserving for as long as
    //      property 2's invariant holds, because the two conditions are then mutually exclusive: a
    //      surviving stamp is never empty. The swap arm PASSED, which is what corrected the comment.
    if (!same_address) return AudioGrainVerdict::PointerMoved;
    if (matches_stamp) return AudioGrainVerdict::Intact;
    if (nonzero_samples == 0) return AudioGrainVerdict::Cleared;
    return AudioGrainVerdict::Overwritten;
}

// Stereo fold-down for a MAIN speaker bed. See audio.hpp for the contract.
//
// The bed order for 1..8 channels is FL, FR, FC, LFE, then surround pairs, with the small layouts
// kept useful (3ch adds FC, 4ch is quad, 5ch is FL/FR/FC/SL/SR, 6ch is 5.1).
//
// A stereo fold only needs each source channel's SIDE (left, right, or both) — the front/back and
// height identity within a side does not change the result — and that is what prosper can measure
// with PROSPER_AUDIO_LAYOUT (see the probe below and docs/AUDIO.md). Measured on Dragon Quest VII
// Reimagined's 12-channel MAIN port over 19,962,368 frames (#1700):
//   ch0/ch1  carry all of the content (rms 3.1e-2 / 4.3e-2, peak 0.407) and correlate +0.46 — a
//            real decorrelated stereo pair, not a duplicated mono feed.
//   ch2      correlates near-equally with both side groups below (0.49/0.46/0.41/0.40): centre-like.
//   ch4,ch6  correlate +0.96 with each other and ch5,ch7 correlate +0.95, while ch4-ch5 is -0.04.
//            The surround tier therefore pairs as even=left, odd=right, which is what the 6..8
//            mapping below already assumed. CONFIDENCE: MED — the grouping is unambiguous but the
//            channels sit at a ~1e-9 residue level, so it corroborates rather than proves.
//   ch3 and ch8..ch11 are EXACTLY zero for the whole run — the title's panner writes neither the
//            LFE nor the height position under the order assumed here. Stated as indices on
//            purpose: which POSITIONS those are is the mapping this evidence supports, not a
//            premise it may borrow.
// Channels 8..15 consequently have no measured side and are deliberately left UNPLACED (see the
// return value): a fold-down that sounds plausible but images content to the wrong side is worse
// than one that reports the gap, and no title in evidence puts signal there. The leading hypothesis
// is that the same even=left/odd=right pairing continues, but that is inference from the tiers
// below it, not a measurement, so it is tracked as an issue instead of shipped.
// What the measurement CANNOT settle, stated so nobody reads more into it than it carries. The
// probe GROUPS channels by side; it does not ORIENT the groups, and those are different claims:
//   - Correlation is symmetric, so it proves ch0/ch1 are a genuine front pair but cannot tell a
//     left/right SWAP from the correct assignment. The same is true of the {ch4,ch6} / {ch5,ch7}
//     grouping: which group is the left one is not measured anywhere here.
//     **CONFIDENCE: MED for the left/right ORIENTATION of every pair in this table, and the basis is
//     convention, not evidence** — index 0 = FrontLeft is universal across published multichannel
//     bed layouts, and prosper's own v1 sceAudioOut path assumes it. TWO conventions, not three: a
//     third was checked and REJECTED. CRI Atom's `EAtomSpeakerID` names are present in this title's
//     eboot and order FrontLeft first, which looked like a property of the code producing these
//     samples rather than an inherited assumption — but `tools/re/xref.py` finds ZERO code
//     references to that string and one data-pointer relocation into a UE reflection table. The
//     names are Blueprint metadata that survived into the image; nothing calls them, so their
//     presence says nothing about the bed's order. Conventions can share one ancestor, and a symbol
//     in a binary is not a code path. prosper's own test suite does pin index -> side for widths
//     6..8, so a self-inconsistent fold fails in CI; what is unconfirmed is the match to hardware.
//   - A listening test cannot settle it either, and specifically cannot settle this title's layout
//     at all: ten of its twelve channels are measured empty, so every mapping that routes ch0 and
//     ch1 to the two sides produces a host bed differing only by the ~1e-9 residue on ch2 and
//     ch4..ch7, some 150 dB below the content — not "bit-identical", because that wording would
//     re-merge "exactly zero" with "1e-9 residue", the very distinction this table draws.
//     Broadly stereo music also folds down
//     plausibly under a swap. A human confirming "it sounds right" is real rung-4 evidence that
//     audio reaches the device through the guest's own path, and is NOT evidence for this mapping.
//   - THE DISCRIMINATING EXPERIMENT, for whoever gets a title that supports it: content with
//     distinct per-channel placement — a hard-panned effect whose true side is known from the game,
//     or centre-channel dialogue — measured with PROSPER_AUDIO_LAYOUT. It settles orientation in
//     one run, where any amount of music cannot. Tracked with the height tier on #1720, since one
//     capture answers both.
// CONFIDENCE: HIGH that ch0/ch1 are the front pair and that 1..2-channel beds are right (a mono or
// stereo bed has no orientation to get wrong); MED for which channel of a pair is the left one, and
// for the 3..8 placements; 9..16 places only its first eight and reports the rest.
unsigned audio_stereo_downmix(unsigned channels, AudioStereoGain* out, unsigned out_capacity) {
    if (!out || !channels || channels > kAudioMaxBedChannels || out_capacity < channels)
        return channels;
    for (unsigned c = 0; c < channels; ++c) out[c] = AudioStereoGain{};

    constexpr float kUnity    = 1.0f;
    constexpr float kCenter   = 0.70710678f;   // -3 dB, centre split equally across the pair
    constexpr float kLfe      = 0.5f;          // -6 dB
    constexpr float kSurround = 0.70710678f;   // -3 dB

    if (channels == 1) { out[0] = {kUnity, kUnity}; return 0; }   // mono feeds both sides
    out[0] = {kUnity, 0.0f};                                      // FL
    out[1] = {0.0f, kUnity};                                      // FR
    if (channels == 2) return 0;
    if (channels == 3) { out[2] = {kCenter, kCenter}; return 0; }
    if (channels == 4) {                                          // quad: FL FR SL SR
        out[2] = {kSurround, 0.0f};
        out[3] = {0.0f, kSurround};
        return 0;
    }
    if (channels == 5) {                                          // FL FR FC SL SR
        out[2] = {kCenter, kCenter};
        out[3] = {kSurround, 0.0f};
        out[4] = {0.0f, kSurround};
        return 0;
    }
    // 6..16: FL FR FC LFE then surround pairs.
    out[2] = {kCenter, kCenter};
    out[3] = {kLfe, kLfe};
    out[4] = {kSurround, 0.0f};
    out[5] = {0.0f, kSurround};
    if (channels == 6) return 0;
    if (channels == 7) {                                          // 6.1: one rear-centre channel
        out[6] = {kSurround * 0.5f, kSurround * 0.5f};
        return 0;
    }
    out[6] = {kSurround, 0.0f};                                   // 7.1 second surround pair
    out[7] = {0.0f, kSurround};
    unsigned unplaced = 0;
    for (unsigned c = 8; c < channels; ++c) ++unplaced;            // height tier: not yet placed
    return unplaced;
}

void audio_reset() {
    AudioSink* s = audio_sink();
    {
        std::lock_guard<std::mutex> lk(g_mx);
        for (int i = 0; i < kMaxPorts; i++) {
            if (g_ports[i].in_use && s) s->close(i + 1);
            g_ports[i] = Port{};
        }
    }
    // Close AudioOut2's host-only context streams while the currently installed sink is still
    // reachable, then restore the headless sink for the next test/application lifecycle.
    audio2_reset();
    g_sink.store(&g_default_sink);
    audio_in_reset_ports();
}

// --- legacy-path diagnostics --------------------------------------------------------------
// PROSPER_AUDIOLOG=1: log port opens (raw args + decoded format), SetVolume args, and a
// once-per-second PCM peak per port, so a silent/quiet/garbled title shows WHERE the signal
// degrades (no output calls vs silent PCM vs wrong format vs volume mapping).
// PROSPER_AUDIO_DUMP=PATH: append each port's raw output() grains to PATH.portN.raw for
// offline analysis, mirroring PROSPER_SHADER_DUMP's capture-first workflow.
namespace {

int audiolog_level() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("PROSPER_AUDIOLOG"); v = (e && *e) ? atoi(e) : 0; if (v < 0) v = 0; }
    return v;
}
bool audiolog() { return audiolog_level() >= 1; }

const char* audio_dump_path() {
    static const char* p = getenv("PROSPER_AUDIO_DUMP");
    return (p && *p) ? p : nullptr;
}

// Peak |sample| of one grain, normalized to [0,1] for either sample format. `sumsq` accumulates
// the squared samples so the caller can also report RMS: peak alone cannot distinguish a mix that
// is audibly present from one stray non-zero sample in an otherwise silent buffer.
double audio_pcm_peak(const void* pcm, int frames, const AudioPortInfo& info,
                      double* sumsq = nullptr) {
    if (!pcm || frames <= 0) return 0.0;
    const int n = frames * info.channels;
    double peak = 0.0, ss = 0.0;
    if (info.fmt == AudioFmt::F32) {
        const float* s = (const float*)pcm;
        for (int i = 0; i < n; i++) { double v = s[i] < 0 ? -(double)s[i] : (double)s[i]; if (v > peak) peak = v; ss += (double)s[i] * (double)s[i]; }
    } else {
        const int16_t* s = (const int16_t*)pcm;
        for (int i = 0; i < n; i++) { double f = (double)s[i] / 32768.0; double v = f < 0 ? -f : f; if (v > peak) peak = v; ss += f * f; }
    }
    if (sumsq) *sumsq += ss;
    return peak;
}

// Once-per-second peak report + optional raw dump. Called from the output paths (guest audio
// thread) with the port's decoded info; per-port state, no cross-port locking needed beyond
// the distinct slots.
void audio_observe_output(int handle, const void* pcm, int frames, const AudioPortInfo& info) {
    if (!audiolog() && !audio_dump_path()) return;
    if (handle < 1 || handle > kMaxSinkPorts) return;
    static struct Obs { uint64_t calls = 0; uint64_t bytes = 0; uint64_t samples = 0;
                        double peak = 0.0; double sumsq = 0.0; FILE* dump = nullptr;
                        std::chrono::steady_clock::time_point last{}; } st[kMaxSinkPorts];
    Obs& s = st[handle - 1];
    s.calls++;
    if (pcm && frames > 0) {
        s.bytes += (uint64_t)frames * audio_frame_bytes(info);
        s.samples += (uint64_t)frames * info.channels;
    }
    double p = audio_pcm_peak(pcm, frames, info, &s.sumsq);
    if (p > s.peak) s.peak = p;
    if (audiolog()) {
        auto now = std::chrono::steady_clock::now();
        if (s.last.time_since_epoch().count() == 0) s.last = now;
        if (now - s.last >= std::chrono::seconds(1)) {
            fprintf(stderr, "[audio] port %d: %llu output calls (total), 1s-bytes=%llu 1s-peak=%.4f 1s-rms=%.4f"
                            " (fmt=%s ch=%d freq=%d grain=%d)\n",
                    handle, (unsigned long long)s.calls, (unsigned long long)s.bytes, s.peak,
                    s.samples ? std::sqrt(s.sumsq / (double)s.samples) : 0.0,
                    info.fmt == AudioFmt::F32 ? "f32" : "s16", info.channels, info.freq, info.grain);
            s.last = now; s.peak = 0.0; s.sumsq = 0.0; s.samples = 0; s.bytes = 0;
        }
    }
    if (const char* base = audio_dump_path()) {
        if (!s.dump) {
            char path[1024];
            snprintf(path, sizeof path, "%s.port%d.raw", base, handle);
            s.dump = fopen(path, "ab");
            fprintf(stderr, "[audio] port %d: dumping raw PCM to %s (fmt=%s ch=%d freq=%d grain=%d)\n",
                    handle, path, info.fmt == AudioFmt::F32 ? "f32" : "s16",
                    info.channels, info.freq, info.grain);
        }
        if (s.dump && pcm && frames > 0) fwrite(pcm, 1, (size_t)frames * audio_frame_bytes(info), s.dump);
    }
}

} // namespace

// --- sceAudioOut HLE --------------------------------------------------------------------
HLE(audio_init) { (void)a0; return 0; }   // sceAudioOutInit: idempotent success

// sceAudioOutOpen(userId, type, index, len, freq, param) -> handle (>=1) or negative error.
HLE(audio_open) {
    (void)a0; (void)a2;
    AudioPortInfo info;
    info.grain = (int)(a3 ? a3 : 256);
    info.freq  = (int)(a4 ? a4 : 48000);
    audio_decode_format((uint32_t)a5, info.channels, info.fmt);
    int type = (int)a1;   // kept for GetPortState's type-dependent output/channel report

    int handle = 0;
    { std::lock_guard<std::mutex> lk(g_mx);
      for (int i = 0; i < kMaxPorts; i++) {
          if (g_ports[i].in_use) continue;
          g_ports[i].in_use = true;
          g_ports[i].type = type;
          g_ports[i].info = info;
          for (int c = 0; c < 8; c++) g_ports[i].vol[c] = kVolume0dB;
          handle = i + 1;
          break;
      } }
    if (!handle) return kAudioErrPortFull;
    if (audiolog())
        fprintf(stderr, "[audio] open: handle=%d userId=%d type=%d index=%d len=%llu freq=%llu param=0x%llx"
                        " -> fmt=%s ch=%d grain=%d\n",
                handle, (int)a0, type, (int)a2, (unsigned long long)a3, (unsigned long long)a4,
                (unsigned long long)a5, info.fmt == AudioFmt::F32 ? "f32" : "s16",
                info.channels, info.grain);
    if (auto* s = audio_sink()) s->open(handle, info);
    return (uint64_t)handle;
}

// sceAudioOutOutput(handle, ptr) -> frames written (>=0) or negative error. ptr==0 => drain.
HLE(audio_output) {
    AudioPortInfo info;
    { std::lock_guard<std::mutex> lk(g_mx); Port* p = port_of((int)a0); if (!p) return kAudioErrInvalidPort; info = p->info; }
    if (a1 == 0) return 0;   // drain/flush: nothing buffered in the headless model
    audio_observe_output((int)a0, P(a1), info.grain, info);
    if (auto* s = audio_sink()) s->output((int)a0, P(a1), info.grain);
    return (uint64_t)info.grain;
}

// sceAudioOutOutputs(SceAudioOutOutputParam param[], int num) -> total frames or negative error.
// SceAudioOutOutputParam = { int32 handle; int32 reserved; void* ptr } (16 bytes).
HLE(audio_outputs) {
    struct OutParam { int32_t handle; int32_t reserved; uint64_t ptr; };
    const auto* arr = (const OutParam*)P(a0);
    int num = (int)a1;
    if (!arr || num <= 0) return 0;
    // sceAudioOutOutputs writes the SAME time-slice to N ports in parallel; the return is samples-per-
    // channel of that slice (one grain), NOT the additive sum over ports. Returning the sum made a guest
    // using the count as a sample-clock over-count by N x (Kyty/shadPS4 both return a single port's grain).
    uint64_t grain = 0; bool have = false;
    for (int i = 0; i < num; i++) {
        AudioPortInfo info; bool ok;
        { std::lock_guard<std::mutex> lk(g_mx); Port* p = port_of(arr[i].handle); ok = (p != nullptr); if (ok) info = p->info; }
        if (!ok) continue;
        if (arr[i].ptr) { audio_observe_output(arr[i].handle, P(arr[i].ptr), info.grain, info);
                          if (auto* s = audio_sink()) s->output(arr[i].handle, P(arr[i].ptr), info.grain); }
        if (!have) { grain = info.grain; have = true; }
    }
    return grain;
}

// sceAudioOutSetVolume(handle, flag(channel mask), int vol[]) -> 0 or negative error.
HLE(audio_set_volume) {
    uint32_t mask = (uint32_t)a1;
    const int* vols = (const int*)P(a2);
    if (audiolog() && vols)
        fprintf(stderr, "[audio] set_volume: handle=%d mask=0x%x vols=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
                (int)a0, mask, vols[0], vols[1], vols[2], vols[3], vols[4], vols[5], vols[6], vols[7]);
    { std::lock_guard<std::mutex> lk(g_mx);
      Port* p = port_of((int)a0); if (!p) return kAudioErrInvalidPort;
      audio_apply_channel_volumes(p->vol, mask, vols); }
    if (auto* s = audio_sink()) s->set_volume((int)a0, mask, vols);
    return 0;
}

// sceAudioOutClose(handle) -> 0 or negative error.
HLE(audio_close) {
    int handle = (int)a0;
    { std::lock_guard<std::mutex> lk(g_mx);
      Port* p = port_of(handle); if (!p) return kAudioErrInvalidPort; p->in_use = false; }
    if (auto* s = audio_sink()) s->close(handle);
    return 0;
}

// sceAudioOutGetPortState(handle, SceAudioOutPortState* state) -> 0 or negative error.
// Layout per Kyty Audio.cpp:340 (the previous fill invented its own: channel as u16 @2
// clobbering reserved1, volume as u32 @8 — which is the FLAG field — and left the real
// volume @4 zero, i.e. "muted"): uint16 output @0; uint8 channel @2; uint8 reserved @3;
// int16 volume @4; uint16 reroute_counter @6; uint64 flag @8; uint64 reserved2[2] @0x10.
HLE(audio_get_port_state) {
    int channels, type;
    { std::lock_guard<std::mutex> lk(g_mx); Port* p = port_of((int)a0); if (!p) return kAudioErrInvalidPort;
      channels = p->info.channels; type = p->type; }
    if (a1) {
        auto* st = (uint8_t*)P(a1);
        memset(st, 0, 0x20);
        *(int16_t*)(st + 4) = 127;   // volume (Kyty AudioOutGetPortState reports 127)
        switch (type) {              // output/channel are port-type dependent (Kyty :432-448)
            case 2: case 3: *(uint16_t*)(st + 0) = 0x40; st[2] = 1; break;                // voice/personal -> headphone
            case 4:         *(uint16_t*)(st + 0) = 0x04; st[2] = 1; break;                // pad speaker
            case 127:       *(uint16_t*)(st + 0) = 0x80; break;                           // aux -> external
            default:        *(uint16_t*)(st + 0) = 0x01;
                            st[2] = (uint8_t)(channels > 2 ? 2 : channels); break;          // main/bgm -> primary
        }
    }
    return 0;
}

// ---- libSceAudioOut2 (PS5-only; no Kyty/shadPS4 reference exists) -----------------------
// DOLL's CRI Atom (ADX) middleware drives audio through AudioOut2. The generic unimplemented
// stub (return 0, out-params untouched) made CRI read an UNINITIALIZED context-memory size and
// malloc/memset it: when the stack garbage happened to be unallocatable the main thread died in
// libc memset(NULL) (RUN ENDED at libc.prx+0x10556, backtrace through the CRI region
// eboot+0x5ff..0x605M) — the intermittent "1 flip then crash" of issue #213. NID identities
// recovered by nid_hash brute force over the sce_stubs corpus: g2tViFIohHE=sceAudioOut2Initialize,
// t5YrizufpQc=sceAudioOut2ContextResetParam, pDmme7Bgm6E=sceAudioOut2ContextQueryMemory.
//
// Contracts recovered by LIVE CAPTURE (PROSPER_AUDIO2LOG probe run, /tmp/draws_a2.log,
// 2026-07-09) — this is a null-device backend in the sense of Wine's null audio driver: real
// handle lifecycle + real-time pacing, no host audio device.
//   sceAudioOut2ContextResetParam(param*)              param is 0x40 bytes (guest zero-fills
//     0x00..0x3f then sets {+0:queue=8, +4:0x40, +8:0, +0xc:2, +0x10:grain=0x100, +0x14:1}).
//   sceAudioOut2ContextQueryMemory(param*, size_t* out) out is the work-memory byte size the
//     guest allocates and hands to ContextCreate (a1 = a0-8 on the create path, live).
//   sceAudioOut2ContextCreate(param*, mem, memSize, Handle* out)
//   sceAudioOut2UserCreate(userId, Handle* out)         (userId=0xff live)
//   sceAudioOut2PortCreate(ctx, portParam*, Handle* out, ...)
//   pump loop (dedicated CRI server thread, live): PortGetState(port, state*) ->
//     PortSetAttributes(port, attr*, n) -> ContextAdvance(ctx) -> ContextPush(ctx, flag).
// ContextPush paces one grain of wall-clock time (blocking-when-full HW semantics, same model
// as RealtimeSilentSink) so the pump thread advances at real time instead of spinning.
// CONFIDENCE: MED (arg positions + struct sizes live-verified; field meanings partly inferred;
// PortGetState layout unknown -> zero-filled 0x20, marked LOW below).
namespace {

bool audio2log() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("PROSPER_AUDIO2LOG"); v = (e && *e && *e != '0') ? 1 : 0; }
    return v == 1;
}

// Fault-safe guest-memory hexdump for live ABI capture (unmapped args must not crash the HLE).
void a2_dump(const char* tag, uint64_t p, size_t n) {
#ifndef _WIN32
    if (!p) return;
    std::vector<uint8_t> buf(n);
    struct iovec l { buf.data(), n }, r { (void*)(uintptr_t)p, n };
    ssize_t got = process_vm_readv(getpid(), &l, 1, &r, 1, 0);
    if (got <= 0) { fprintf(stderr, "[audio2]   %s @0x%llx: <unreadable>\n", tag, (unsigned long long)p); return; }
    fprintf(stderr, "[audio2]   %s @0x%llx:", tag, (unsigned long long)p);
    for (ssize_t i = 0; i < got; i++) {
        if ((i & 15) == 0) fprintf(stderr, "\n[audio2]     +%02zx ", (size_t)i);
        fprintf(stderr, "%02x ", buf[i]);
    }
    fprintf(stderr, "\n");
#else
    (void)tag; (void)p; (void)n;
#endif
}

void a2_log(const char* name, uint64_t a0, uint64_t a1, uint64_t a2,
            uint64_t a3, uint64_t a4, uint64_t a5, void* ra) {
    if (!audio2log()) return;
    fprintf(stderr, "[audio2] %s(0x%llx, 0x%llx, 0x%llx, 0x%llx, 0x%llx, 0x%llx) ra=eboot+0x%llx\n",
            name, (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
            (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5,
            prosper::guest_module_name((uint64_t)ra),
            (unsigned long long)prosper::guest_module_offset((uint64_t)ra));
}

} // namespace

#define A2LOG(name) a2_log(name, a0, a1, a2, a3, a4, a5, __builtin_return_address(0))

// AudioOut2 handle space. Handles are opaque u64s the guest stores and passes back; tag them so
// stray guest values are distinguishable in logs. One context + a few ports is all CRI uses.
constexpr uint64_t kA2CtxTag  = 0xA2C0000000000000ull;
constexpr uint64_t kA2UserTag = 0xA2D0000000000000ull;
constexpr uint64_t kA2PortTag = 0xA2E0000000000000ull;
constexpr uint64_t kA2SpeakerArrayTag = 0xA2F0000000000000ull;
constexpr uint64_t kA2HandleTagMask = 0xffff000000000000ull;
constexpr uint64_t kA2HandleIndexMask = 0xffffull;
constexpr uint32_t kA2MaxPorts = 256;
// Main-bed port flag bit 1 carries 20 dB of intentional digital headroom for AudioOut2's platform
// mastering stage.  Restore that reference level at the host boundary; flag-zero ports are already
// consumer-level PCM and must remain unchanged (notably video playback and ordinary stereo titles).
constexpr uint32_t kA2PortFlag20DbHeadroom = 1u << 1;
constexpr float kA2HeadroomGain = 10.0f;

struct A2Context {
    bool     used = false;
    uint32_t generation = 0;
    bool     sink_opened = false;
    bool     sink_open_ok = false;
    uint32_t queue_depth = 4;
    uint32_t grain = 256;         // samples per Advance/Push cycle (param +0x10, live: 0x100)
    uint32_t queued = 0;          // submitted grains not yet consumed by the 48 kHz device clock
    std::chrono::steady_clock::time_point last_queue_update{};
    // Real-time pacing state for ContextPush (blocking-when-full HW semantics).
    std::chrono::steady_clock::time_point next{};
};
std::mutex g_a2_mx;
// Serializes each context slot's host stream across open/output/close. Always acquire a slot mutex
// before g_a2_mx when both are needed, so Destroy cannot close a stream reopened by a recycled
// context and an in-flight Push cannot output after that slot has been torn down.
std::mutex g_a2_sink_mx[4];
A2Context  g_a2_ctx[4];
uint32_t   g_a2_users = 0;

// AudioOut2 data path (derived live from Evergate/GTA V and cross-checked against Kyty's public
// reverse-engineered structs):
// each tick the guest calls PortSetAttributes(port, attrs, count) where an attribute triple
// {u32 id=0, u32 reserved, value_ptr, value_size=8} carries a guest pointer to the port's current grain of
// interleaved PCM (grain frames from the context param, 256 live). ContextAdvance
// advances engine state and ContextPush submits the grain to the device. SetAttributes must copy
// each grain: guests reuse one scratch buffer for multiple ports before Advance/Push (#3411).
struct A2PortState {
    bool     used = false;
    uint32_t generation = 0;
    uint64_t context = 0;      // owning context; Push must not submit another context's ports
    uint64_t pcm_ptr = 0;      // source address, retained for diagnostics (attr id 0)
    // Normalize the guest's interleaved grain into independently owned channel buffers. Neither
    // another output's scratch reuse nor the final stereo mix can change these source channels.
    std::array<std::vector<float>, kAudioMaxBedChannels> pcm_channels;
    uint32_t pcm_frames = 0;
    bool pcm_pending = false; // a submission is consumed once, never looped to fill a late grain
    uint16_t type = 0;         // portParam +0x00: 0 = MAIN (speaker output); others aux (personal/...)
    uint32_t data_format = 0;  // portParam +0x04: bits 8..15=channels, bits 0..6=0 F32 / 1 S16,
                               // bit 7 selects the standard 8-channel order
    uint32_t flags = 0;        // portParam +0x0c: device/mastering mode bits
};
A2PortState g_a2_port_state[kA2MaxPorts];
bool g_a2_speaker_arrays[32]{};
constexpr int kA2SinkPortBase = kMaxPorts + 1; // host-only ports 17..20, one per AudioOut2 context

// --- PROSPER_AUDIO_FLOW: end-to-end audio submission probe --------------------------------
// A silent title has three mutually exclusive causes that look IDENTICAL from outside, and each
// needs a different fix:
//   (1) the guest never submits PCM        -> the defect is upstream (mixer/voice/HLE return),
//   (2) the guest submits all-zero PCM     -> the defect is in the title's own mixing/decode,
//   (3) the guest submits real PCM and we  -> the defect is ours (routing/sink/volume).
//       drop, discard or mute it
// The pre-existing diagnostics cannot separate these, because BOTH of them fall silent in cases
// (1) and (2): the once-per-second `[audio] port N` line is only reached after `have_pcm &&
// open_ok` (so it never prints if no MAIN port yielded frames), and `PROSPER_AUDIO2_PROBE` only
// prints a port whose `|max| > 0` (so an all-zero submission prints nothing). "No output" was
// therefore an uninformative negative.
//
// PROSPER_AUDIO_FLOW=1 reports UNCONDITIONALLY — including all-zero signal and zero-length reads —
// once per second per AudioOut2 context, from BOTH Advance and Push, so every branch is visible:
//   no lines at all              -> the guest never created a context / never ran its pump loop
//   `advance=N push=0`           -> the pump runs but never submits            (case 1)
//   `push=N ... bed-peak=0.0000` -> real submissions carrying silence          (case 2)
//   `push=N ... bed-peak>0`      -> real signal reaches the host sink          (case 3)
// The per-port breakdown additionally attributes losses that the mixed bed alone hides: a port
// carrying signal that is discarded for not being MAIN (`skip-not-main`), an unsupported format
// (`skip-fmt`), a never-published PCM pointer (`no-pcm`), or a short/faulting guest read
// (`short`). Reported per interval: submission count, byte count, and both peak and RMS.
namespace {

bool audio_flow() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("PROSPER_AUDIO_FLOW"); v = (e && *e && *e != '0') ? 1 : 0; }
    return v == 1;
}

// The non-silence measure itself lives in audio.hpp as AudioSignalStats so the rule this whole
// diagnostic rests on (silent vs quiet vs absent) is unit-tested without a device or a live boot.
using FlowSignal = AudioSignalStats;

struct FlowPort {
    uint64_t reads = 0, mixed = 0, frames = 0, bytes = 0;
    uint64_t no_pcm = 0, no_grain = 0, skip_fmt = 0, skip_not_main = 0, short_read = 0;
    uint16_t type = 0;
    uint32_t data_format = 0;
    FlowSignal sig;
    bool     seen = false;
    uint32_t ctx_slot = 0;      // owning context, so each context reports only ITS ports
    // Run-total signal, never reset. A per-interval line alone is a trap: a port can be silent for
    // the interval that happens to be read (a gap between cues, or the moments before playback
    // starts) while carrying strong signal over the run. Reading one such line cost this
    // investigation a completely inverted conclusion, so every line carries the lifetime total too.
    // Deliberately the SAME accumulator type as the interval measure: the verdict is drawn from
    // this number, so it must not be a second, untested hand-rolled copy of the rule.
    AudioSignalStats life;
    // NaN is neither silence nor signal, and mapping it to either hides a real defect: the sink
    // clamps NaN to zero, so a NaN-producing guest is audibly silent while carrying "data".
    uint64_t nan_samples = 0, life_nan = 0;
};

struct FlowCtx {
    uint64_t advances = 0, pushes = 0, pushes_with_pcm = 0, silent_paced = 0, not_ready = 0;
    uint64_t sink_grains = 0, sink_bytes = 0;
    FlowSignal bed;
    // The mixed bed needs a never-reset total for exactly the reason each port does — and more
    // urgently, because `bed-nonzero` is what the documented decision table keys on. A per-interval
    // bed value is a RATE; only the run total is a verdict.
    FlowSignal bed_life;
    uint64_t bed_nan = 0, bed_life_nan = 0;
    bool sink_opened = false, sink_open_ok = false;
    std::chrono::steady_clock::time_point last{};
};

// Reached-ness counters, so a null reading is a FINDING rather than an ambiguity. "Zero
// submissions" means something completely different depending on how far the guest got, and
// those cases are indistinguishable from the submission counters alone:
//   ports_created=0                 -> the guest never built an audio graph at all
//   ports_created>0, pcm_published=0 -> it built one but never handed us a PCM buffer
//   pcm_published>0, pushes=0        -> buffers are wired up but the pump never submits
// Counted globally (not per context) because they are lifecycle facts about the whole run.
std::atomic<uint64_t> g_flow_ctx_created{0};
std::atomic<uint64_t> g_flow_ports_created{0};
std::atomic<uint64_t> g_flow_pcm_published{0};   // PortSetAttributes calls that set a PCM pointer

// g_flow_mx is a LEAF: no other lock may be acquired while it is held, and the report must never
// read g_a2_* state from inside it. That is the invariant, not "acquired last" — it is taken both
// with g_a2_mx/g_a2_sink_mx held and with nothing held, and only leaf-ness makes both safe.
// Concretely: do NOT look a port's owning context up from g_a2_port_state inside the report (the
// obvious way to filter by context) — that would be g_flow_mx -> g_a2_mx against the
// g_a2_mx -> g_flow_mx order used by the push path, i.e. a genuine deadlock. FlowPort::ctx_slot is
// cached precisely so the report needs no other lock.
std::mutex g_flow_mx;
FlowCtx    g_flow_ctx[4];
FlowPort   g_flow_port[kA2MaxPorts];

// Tag a port slot with the context that owns it, so each context reports only its own ports.
void flow_tag_port(FlowPort& fp, uint32_t ctx_slot, uint16_t type, uint32_t data_format) {
    fp.seen = true;
    fp.ctx_slot = ctx_slot;
    fp.type = type;
    fp.data_format = data_format;
}

// Accumulate one raw sample into both the per-interval and the never-reset run totals.
// NaN is counted and then treated as silence, matching what the sink actually outputs.
void flow_add_sample(FlowPort& fp, float v) {
    if (v != v) { ++fp.nan_samples; ++fp.life_nan; v = 0.0f; }
    fp.sig.add(v);
    fp.life.add(v);
}

// Same for the mixed stereo bed. The bed must classify NaN exactly as the ports do: the output
// path clamps NaN to zero, so counting it as signal would report "case 3, signal reaches the sink"
// for a guest whose audio is inaudible.
void flow_add_bed_sample(FlowCtx& f, float v) {
    if (v != v) { ++f.bed_nan; ++f.bed_life_nan; v = 0.0f; }
    f.bed.add(v);
    f.bed_life.add(v);
}

// Emit one interval line per context plus a line per port that this context actually touched.
// Caller must NOT hold g_flow_mx.
void audio_flow_report(uint32_t slot) {
    if (!audio_flow() || slot >= 4) return;
    std::lock_guard<std::mutex> lk(g_flow_mx);
    FlowCtx& f = g_flow_ctx[slot];
    const auto now = std::chrono::steady_clock::now();
    if (f.last.time_since_epoch().count() == 0) { f.last = now; return; }
    if (now - f.last < std::chrono::seconds(1)) return;
    f.last = now;
    fprintf(stderr,
            "[audio-flow] ctx%u advance=%llu push=%llu (with-pcm=%llu silent-paced=%llu not-ready=%llu)"
            " sink=%s port=%d grains=%llu bytes=%llu bed-peak=%.5f bed-rms=%.5f bed-nonzero=%llu/%llu"
            " bed-nan=%llu | BED LIFE: nonzero=%llu/%llu peak=%.5f rms=%.5f nan=%llu"
            " | lifetime: ctx=%llu ports=%llu pcm-published=%llu\n",
            slot, (unsigned long long)f.advances, (unsigned long long)f.pushes,
            (unsigned long long)f.pushes_with_pcm, (unsigned long long)f.silent_paced,
            (unsigned long long)f.not_ready,
            !f.sink_opened ? "never-opened" : (f.sink_open_ok ? "open" : "OPEN-FAILED"),
            kA2SinkPortBase + (int)slot,
            (unsigned long long)f.sink_grains, (unsigned long long)f.sink_bytes,
            f.bed.peak, f.bed.rms(),
            (unsigned long long)f.bed.nonzero, (unsigned long long)f.bed.samples,
            (unsigned long long)f.bed_nan,
            (unsigned long long)f.bed_life.nonzero, (unsigned long long)f.bed_life.samples,
            f.bed_life.peak, f.bed_life.rms(), (unsigned long long)f.bed_life_nan,
            (unsigned long long)g_flow_ctx_created.load(),
            (unsigned long long)g_flow_ports_created.load(),
            (unsigned long long)g_flow_pcm_published.load());
    for (uint32_t i = 0; i < kA2MaxPorts; ++i) {
        FlowPort& p = g_flow_port[i];
        // Only this context's own ports. g_flow_port[] is a single global table, so without this
        // filter every context prints (and RESETS) every other context's counters, splitting each
        // port's per-second totals arbitrarily across unrelated context lines.
        if (!p.seen || p.ctx_slot != slot) continue;
        fprintf(stderr,
                "[audio-flow]   port%u type=0x%x fmt=0x%x reads=%llu mixed=%llu frames=%llu bytes=%llu"
                " peak=%.5f rms=%.5f nonzero=%llu/%llu nan=%llu no-pcm=%llu skip-fmt=%llu"
                " skip-not-main=%llu short=%llu no-grain=%llu"
                " | LIFE: nonzero=%llu/%llu peak=%.5f rms=%.5f nan=%llu\n",
                i + 1, p.type, p.data_format, (unsigned long long)p.reads,
                (unsigned long long)p.mixed, (unsigned long long)p.frames,
                (unsigned long long)p.bytes, p.sig.peak, p.sig.rms(),
                (unsigned long long)p.sig.nonzero, (unsigned long long)p.sig.samples,
                (unsigned long long)p.nan_samples,
                (unsigned long long)p.no_pcm, (unsigned long long)p.skip_fmt,
                (unsigned long long)p.skip_not_main, (unsigned long long)p.short_read,
                (unsigned long long)p.no_grain,
                (unsigned long long)p.life.nonzero, (unsigned long long)p.life.samples,
                p.life.peak, p.life.rms(), (unsigned long long)p.life_nan);
        // Per-interval counters: reset so each line describes the LAST second, not the whole run.
        // The `life:` totals above are deliberately NOT reset.
        p.reads = p.mixed = p.frames = p.bytes = 0;
        p.no_pcm = p.no_grain = p.skip_fmt = p.skip_not_main = p.short_read = p.nan_samples = 0;
        p.seen = false;
        p.sig.reset();
    }
    f.advances = f.pushes = f.pushes_with_pcm = f.silent_paced = f.not_ready = 0;
    f.sink_grains = f.sink_bytes = 0;
    f.bed_nan = 0;
    f.bed.reset();          // f.bed_life / f.bed_life_nan are deliberately NOT reset
}

// One-shot lifecycle markers. These make "no [audio-flow] lines at all" a POSITIVE result rather
// than an ambiguous one: with a ctx-create marker present and no interval lines following it, the
// guest demonstrably built an audio context and then never ran its pump loop.
void audio_flow_note_create(uint32_t slot, uint32_t grain, uint32_t queue_depth) {
    if (!audio_flow() || slot >= 4) return;
    ++g_flow_ctx_created;
    std::lock_guard<std::mutex> lk(g_flow_mx);
    g_flow_ctx[slot] = FlowCtx{};   // a recycled context slot starts a fresh history, lifetime included
    fprintf(stderr, "[audio-flow] ctx%u created: grain=%u queue_depth=%u (host sink port %d)\n",
            slot, grain, queue_depth, kA2SinkPortBase + (int)slot);
}

void audio_flow_note_port_create(uint32_t port_index, uint16_t type, uint32_t data_format,
                                 uint32_t flags) {
    if (!audio_flow()) return;
    ++g_flow_ports_created;
    // Port slots are recycled first-free, and object-heavy titles churn far more ports than there
    // are slots. Without this clear, slot i's never-reset `LIFE:` total would merge a music port's
    // history with an unrelated SFX bus that later reuses the slot — the same mis-attribution the
    // per-context filter fixes, one level down, and now more damaging because the docs direct the
    // reader to judge a port by exactly that total. Caller holds g_a2_mx; g_flow_mx is a leaf.
    if (port_index < kA2MaxPorts) {
        std::lock_guard<std::mutex> lk(g_flow_mx);
        g_flow_port[port_index] = FlowPort{};
    }
    fprintf(stderr, "[audio-flow] port%u created: type=0x%x data_format=0x%x flags=0x%x%s\n",
            port_index + 1, type, data_format, flags,
            type == 0 ? " (MAIN -> mixed to host)" : " (non-MAIN -> NOT mixed to host)");
}

// --- PROSPER_AUDIO_LAYOUT: per-channel bed-layout probe -----------------------------------
// A multichannel MAIN port has to be folded into the host's stereo bed, and a fold-down is only
// correct if we know which source channel is left, right, centre or LFE. `data_format` carries a
// channel COUNT and nothing else, so the ORDER is not derivable from it — and a wrong order sounds
// plausible, which is worse than silence because it reads as working.
//
// This probe measures the guest's own PCM and reports the properties that identify each channel's
// role without assuming any ordering:
//   rms/peak/nonzero — which channels carry the bed at all. A height channel a title never uses is
//                      then measurably ABSENT rather than merely assumed quiet.
//   hf  = RMS(x[n]-x[n-1]) / RMS(x[n]) — a spectral-tilt measure. An LFE feed is low-passed (about
//                      120 Hz), so at 48 kHz its value is ~0.016, while full-band content sits
//                      around 0.3..1.5. This identifies LFE from content alone, at any index.
//   corr(c,d) — normalized cross-correlation. A front stereo pair is strongly but not perfectly
//                      correlated, a duplicated channel reads 1.00, an independent height channel
//                      reads near 0. This is what assigns each channel a LEFT/RIGHT side, which is
//                      all a stereo fold-down actually needs to be correct.
//   stride    — the smallest non-zero distance between two distinct PCM grain pointers the guest
//                      publishes for the port. Titles double-buffer the grain, so this is one
//                      grain's byte size and therefore an INDEPENDENT measure of the channel count
//                      (channels = stride / (grain * bytes-per-sample)) — a cross-check on the
//                      whole data_format decode rather than a restatement of it.
// One name for the clamp: audio2_format_channels(), the fold matrix and this probe must agree.
constexpr uint32_t kA2MaxChannels = kAudioMaxBedChannels;

bool audio_layout() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("PROSPER_AUDIO_LAYOUT"); v = (e && *e && *e != '0') ? 1 : 0; }
    return v == 1;
}

// PROSPER_AUDIO_LAYOUT_DUMP=PATH additionally appends each measured port's RAW interleaved grain
// to PATH.portN.<channels>ch.f32 as float32 (both sample types converted, so one reader handles either). The
// in-process statistics answer the layout question, but the raw capture is what lets a later reader
// re-derive them, run a different analysis, or render a channel to something audible — the same
// capture-first workflow as PROSPER_SHADER_DUMP.
const char* audio_layout_dump_path() {
    static const char* p = getenv("PROSPER_AUDIO_LAYOUT_DUMP");
    return (p && *p) ? p : nullptr;
}

struct LayoutPort {
    bool     seen = false;
    uint32_t ctx_slot = 0;
    uint16_t type = 0;
    uint32_t data_format = 0;
    uint32_t channels = 0;
    uint32_t grain = 0;
    uint64_t frames = 0;
    double   sumsq[kA2MaxChannels]{};
    double   diff_sumsq[kA2MaxChannels]{};
    double   cross[kA2MaxChannels][kA2MaxChannels]{};
    double   peak[kA2MaxChannels]{};
    uint64_t nonzero[kA2MaxChannels]{};
    float    prev[kA2MaxChannels]{};
    bool     have_prev = false;
    // Grain-pointer stride (published by PortSetAttributes, not by the mix loop).
    uint64_t last_ptr = 0;
    uint64_t min_stride = 0;
    uint64_t ptr_updates = 0;
    FILE*    dump = nullptr;      // PROSPER_AUDIO_LAYOUT_DUMP, opened once per port
};

// Allocated only when the probe is enabled: 256 ports x a 16x16 correlation matrix is ~700 KiB
// that a default run must not carry.
std::mutex g_layout_mx;
std::unique_ptr<LayoutPort[]> g_layout_port;

LayoutPort* layout_port_locked(uint32_t index) {
    if (!audio_layout() || index >= kA2MaxPorts) return nullptr;
    if (!g_layout_port) g_layout_port.reset(new LayoutPort[kA2MaxPorts]());
    return &g_layout_port[index];
}

// One grain of interleaved PCM from one port. `frame(f, c)` reads the already-converted float, so
// S16 and F32 ports are measured identically.
template <typename Sample>
void layout_add_grain(uint32_t port_index, uint32_t ctx_slot, uint16_t type, uint32_t data_format,
                      uint32_t channels, uint32_t grain, uint64_t frames_got, Sample&& frame) {
    std::lock_guard<std::mutex> lk(g_layout_mx);
    LayoutPort* lp = layout_port_locked(port_index);
    if (!lp || !channels || channels > kA2MaxChannels) return;
    if (lp->channels && lp->channels != channels) lp->have_prev = false;  // reconfigured mid-run
    lp->seen = true;
    lp->ctx_slot = ctx_slot;
    lp->type = type;
    lp->data_format = data_format;
    lp->channels = channels;
    lp->grain = grain;
    lp->frames += frames_got;
    if (const char* base = audio_layout_dump_path(); base && !lp->dump) {
        // The channel count is in the NAME, not just the log line. The file is a bare interleaved
        // stream with no header, and port slots are recycled first-free, so a differently-shaped
        // port reusing this slot would otherwise append a stream of another width to the same file
        // and every offline reader would silently de-interleave the tail wrong.
        // "wb", not "ab", for the SPAN half of the same problem: the statistics reset per port
        // lifetime while an appending file does not, so a recycled slot of the SAME width would
        // leave the printed `frames=N` and the file describing different spans — and an offline
        // reader has no way to see the seam. One file per port lifetime keeps them the same span.
        char path[512];
        snprintf(path, sizeof path, "%s.port%u.%uch.f32", base, port_index + 1, channels);
        lp->dump = fopen(path, "wb");
        fprintf(stderr, "[audio-layout] port%u: dumping raw guest PCM to %s "
                        "(float32 interleaved, %u channels, 48000 Hz)\n",
                port_index + 1, path, channels);
    }
    FILE* const dump = lp->dump;
    for (uint64_t f = 0; f < frames_got; ++f) {
        float v[kA2MaxChannels];
        for (uint32_t c = 0; c < channels; ++c) {
            float s = frame(f, c);
            if (s != s) s = 0.0f;                        // NaN measured as the silence the sink emits
            v[c] = s;
        }
        if (dump) fwrite(v, sizeof(float), channels, dump);
        for (uint32_t c = 0; c < channels; ++c) {
            const double d = (double)v[c];
            const double a = d < 0 ? -d : d;
            if (a > lp->peak[c]) lp->peak[c] = a;
            lp->sumsq[c] += d * d;
            if (v[c] != 0.0f) ++lp->nonzero[c];
            if (lp->have_prev) {
                const double delta = d - (double)lp->prev[c];
                lp->diff_sumsq[c] += delta * delta;
            }
            for (uint32_t e = c; e < channels; ++e) lp->cross[c][e] += d * (double)v[e];
        }
        for (uint32_t c = 0; c < channels; ++c) lp->prev[c] = v[c];
        lp->have_prev = true;
    }
    if (dump) fflush(dump);   // a run killed by its route timeout must still leave a readable capture
}

// A recycled port slot starts a fresh layout history, exactly as the flow probe's LIFE totals do:
// merging a destroyed port's channel statistics into the unrelated port that reuses the slot would
// corrupt the very correlations the layout verdict is drawn from.
void layout_note_port_create(uint32_t port_index) {
    if (!audio_layout()) return;
    std::lock_guard<std::mutex> lk(g_layout_mx);
    if (LayoutPort* lp = layout_port_locked(port_index)) {
        if (lp->dump) fclose(lp->dump);
        *lp = LayoutPort{};
    }
}

void layout_note_pcm_ptr(uint32_t port_index, uint64_t pcm) {
    if (!audio_layout() || !pcm) return;
    std::lock_guard<std::mutex> lk(g_layout_mx);
    LayoutPort* lp = layout_port_locked(port_index);
    if (!lp) return;
    ++lp->ptr_updates;
    if (lp->last_ptr && pcm != lp->last_ptr) {
        const uint64_t stride = pcm > lp->last_ptr ? pcm - lp->last_ptr : lp->last_ptr - pcm;
        if (!lp->min_stride || stride < lp->min_stride) lp->min_stride = stride;
    }
    lp->last_ptr = pcm;
}

void audio_layout_report() {
    if (!audio_layout()) return;
    static std::chrono::steady_clock::time_point last{};
    const auto now = std::chrono::steady_clock::now();
    {
        static std::mutex clock_mx;
        std::lock_guard<std::mutex> lk(clock_mx);
        if (last.time_since_epoch().count() == 0) { last = now; return; }
        if (now - last < std::chrono::seconds(5)) return;
        last = now;
    }
    std::lock_guard<std::mutex> lk(g_layout_mx);
    if (!g_layout_port) return;
    for (uint32_t i = 0; i < kA2MaxPorts; ++i) {
        LayoutPort& lp = g_layout_port[i];
        if (!lp.seen || !lp.frames || !lp.channels) continue;
        const double n = (double)lp.frames;
        double rms[kA2MaxChannels]{};
        for (uint32_t c = 0; c < lp.channels; ++c) rms[c] = std::sqrt(lp.sumsq[c] / n);
        // Stride is only meaningful against a known sample width; report the implied channel count
        // so the reader does not have to redo the arithmetic (and can see a disagreement at once).
        const uint32_t bytes_per_sample = (lp.data_format & 0x7fu) == 1 ? 2u : 4u;
        const uint64_t frame_bytes = lp.grain ? (uint64_t)lp.grain * bytes_per_sample : 0;
        char stride_note[128];
        if (lp.min_stride && frame_bytes)
            snprintf(stride_note, sizeof stride_note, "stride=%lluB (implies %.2f channels)",
                     (unsigned long long)lp.min_stride, (double)lp.min_stride / (double)frame_bytes);
        else if (lp.ptr_updates)
            // A positive finding, not a missing measurement: the guest republished the SAME grain
            // address every time, so there is no second buffer to measure a stride against.
            snprintf(stride_note, sizeof stride_note,
                     "stride=n/a (single grain buffer 0x%llx, %llu publications)",
                     (unsigned long long)lp.last_ptr, (unsigned long long)lp.ptr_updates);
        else
            snprintf(stride_note, sizeof stride_note, "stride=n/a (no PCM pointer observed)");
        // What prosper DID with each channel, printed beside what the channel measured as. The two
        // halves of the layout question are "what role does this channel play in the guest's mix"
        // (rms/hf/corr, above) and "where did prosper send it" — and until now the second half was
        // only ever readable by opening hle_audio.cpp and applying the table by hand, which is
        // exactly the step nobody takes while reading a log. A channel that measures as a surround
        // feed and folds to `UNPLACED`, or one that measures centre-like and folds hard to one
        // side, is then visible in the same two lines instead of in an offline derivation.
        // Only a MAIN port is folded at all; anything else is not mixed to the host bed, so
        // claiming a placement for it would be a straight falsehood.
        AudioStereoGain fold[kA2MaxChannels]{};
        const bool      folded = lp.type == 0;
        const unsigned  unplaced =
            folded ? audio_stereo_downmix(lp.channels, fold, kA2MaxChannels) : lp.channels;
        char fold_note[96];
        if (!folded)
            snprintf(fold_note, sizeof fold_note, "fold=n/a (non-MAIN: not mixed to the host bed)");
        else if (unplaced)
            snprintf(fold_note, sizeof fold_note, "fold=7.1 + %u UNPLACED", unplaced);
        else
            snprintf(fold_note, sizeof fold_note, "fold=all %u placed", lp.channels);
        fprintf(stderr,
                "[audio-layout] port%u ctx%u type=0x%x fmt=0x%x channels=%u grain=%u frames=%llu %s %s\n",
                i + 1, lp.ctx_slot, lp.type, lp.data_format, lp.channels, lp.grain,
                (unsigned long long)lp.frames, stride_note, fold_note);
        for (uint32_t c = 0; c < lp.channels; ++c) {
            const double hf = rms[c] > 0.0 ? std::sqrt(lp.diff_sumsq[c] / n) / rms[c] : 0.0;
            fprintf(stderr, "[audio-layout]   ch%-2u rms=%.6f peak=%.6f nonzero=%5.1f%% hf=%.4f",
                    c, rms[c], lp.peak[c], 100.0 * (double)lp.nonzero[c] / n, hf);
            if (!folded)
                fprintf(stderr, " fold=n/a  ");
            else if (fold[c].left == 0.0f && fold[c].right == 0.0f)
                fprintf(stderr, " fold=UNPLACED");
            else
                fprintf(stderr, " fold=L%.3f/R%.3f", (double)fold[c].left, (double)fold[c].right);
            fprintf(stderr, " corr=");
            for (uint32_t e = 0; e < lp.channels; ++e) {
                const uint32_t lo = c < e ? c : e, hi = c < e ? e : c;
                const double denom = rms[lo] * rms[hi] * n;
                const double corr = denom > 0.0 ? lp.cross[lo][hi] / denom : 0.0;
                fprintf(stderr, " %+.2f", corr);
            }
            fprintf(stderr, "\n");
        }
    }
}

} // namespace


uint32_t audio2_next_generation(uint32_t generation) {
    ++generation;
    return generation ? generation : 1;
}

uint64_t audio2_make_handle(uint64_t tag, uint32_t generation, uint32_t one_based_index) {
    return tag | ((uint64_t)generation << 16) | one_based_index;
}

template <typename Slot>
void audio2_clear_slot(Slot& slot) {
    const uint32_t generation = slot.generation;
    slot = Slot{};
    slot.generation = generation;
}

// Callers hold g_a2_mx. Centralizing the full tag/index/generation/live checks prevents a stale
// handle from aliasing a recycled slot and keeps every implemented AudioOut2 operation consistent.
A2Context* audio2_context_locked(uint64_t handle, uint32_t* slot_out = nullptr) {
    const uint32_t one_based_index = (uint32_t)(handle & kA2HandleIndexMask);
    const uint32_t generation = (uint32_t)(handle >> 16);
    if ((handle & kA2HandleTagMask) != kA2CtxTag || one_based_index < 1 ||
        one_based_index > std::size(g_a2_ctx) || !generation) return nullptr;
    A2Context& context = g_a2_ctx[one_based_index - 1];
    if (!context.used || context.generation != generation) return nullptr;
    if (slot_out) *slot_out = one_based_index - 1;
    return &context;
}

A2PortState* audio2_port_locked(uint64_t handle, uint32_t* slot_out = nullptr) {
    const uint32_t one_based_index = (uint32_t)(handle & kA2HandleIndexMask);
    const uint32_t generation = (uint32_t)(handle >> 16);
    if ((handle & kA2HandleTagMask) != kA2PortTag || one_based_index < 1 ||
        one_based_index > kA2MaxPorts || !generation) return nullptr;
    A2PortState& port = g_a2_port_state[one_based_index - 1];
    if (!port.used || port.generation != generation) return nullptr;
    if (slot_out) *slot_out = one_based_index - 1;
    return &port;
}

// The DECLARED channel count, unclamped (an unset field is AudioOut2's stereo default). It used to
// be clamped to the fold's maximum, which is exactly the disagreement #1700 was about — and a clamp
// is the more dangerous half of it: silently reading a 20-channel grain as 16 channels would walk
// the guest's buffer at the wrong stride and mix garbage, where the old over-8 reject at least
// produced silence. Range is the consumer's decision, and it is made loudly at the one place that
// sizes a read.
uint32_t audio2_format_channels(uint32_t data_format) {
    const uint32_t encoded = (data_format >> 8u) & 0xffu;
    return encoded ? encoded : 2u;
}

static void audio2_reset() {
    std::scoped_lock sink_locks(g_a2_sink_mx[0], g_a2_sink_mx[1],
                                g_a2_sink_mx[2], g_a2_sink_mx[3]);
    bool close_sink[4]{};
    {
        std::lock_guard<std::mutex> lk(g_a2_mx);
        for (size_t i = 0; i < 4; ++i) {
            close_sink[i] = g_a2_ctx[i].sink_opened;
            audio2_clear_slot(g_a2_ctx[i]);
        }
        for (auto& port : g_a2_port_state) audio2_clear_slot(port);
        std::fill(std::begin(g_a2_speaker_arrays), std::end(g_a2_speaker_arrays), false);
        g_a2_users = 0;
    }
    // The probe's own state is per-lifecycle too: leaving stale contexts/ports/lifetime totals here
    // would carry one application's audio history into the next (and across tests).
    if (audio_flow()) {
        std::lock_guard<std::mutex> lk(g_flow_mx);
        for (auto& f : g_flow_ctx) f = FlowCtx{};
        for (auto& fp : g_flow_port) fp = FlowPort{};
        g_flow_ctx_created = 0;
        g_flow_ports_created = 0;
        g_flow_pcm_published = 0;
    }
    if (audio_layout()) {
        std::lock_guard<std::mutex> lk(g_layout_mx);
        if (g_layout_port)
            for (uint32_t i = 0; i < kA2MaxPorts; ++i)
                if (g_layout_port[i].dump) fclose(g_layout_port[i].dump);
        g_layout_port.reset();
    }
    if (AudioSink* sink = audio_sink())
        for (size_t i = 0; i < 4; ++i)
            if (close_sink[i]) sink->close(kA2SinkPortBase + (int)i);
}

// Fault-safe store to a guest out-pointer (same rationale as apr_write_guest_dst: a bad pointer
// must fail the call, not SIGSEGV inside the HLE). WriteProcessMemory validates the complete
// destination range before copying, matching the all-or-fail process_vm_writev contract here.
bool audio_store_bytes(uint64_t dst, const void* src, size_t n) {
    if (!dst || (!src && n)) return false;
    if (!n) return true;
#ifndef _WIN32
    struct iovec l { const_cast<void*>(src), n }, r { (void*)(uintptr_t)dst, n };
    return process_vm_writev(getpid(), &l, 1, &r, 1, 0) == (ssize_t)n;
#else
    SIZE_T written = 0;
    return WriteProcessMemory(GetCurrentProcess(), (void*)(uintptr_t)dst, src, n, &written) &&
           written == n;
#endif
}

// Fault-safe read from guest memory. Both platform paths require the complete range so callers
// never consume a partially copied guest structure after an inaccessible-range failure.
bool audio_read_bytes(uint64_t src, void* dst, size_t n) {
    if (!src || (!dst && n)) return false;
    if (!n) return true;
#ifndef _WIN32
    struct iovec l { dst, n }, r { (void*)(uintptr_t)src, n };
    return process_vm_readv(getpid(), &l, 1, &r, 1, 0) == (ssize_t)n;
#else
    SIZE_T read = 0;
    return ReadProcessMemory(GetCurrentProcess(), (const void*)(uintptr_t)src, dst, n, &read) &&
           read == n;
#endif
}

// Best-effort read: copy as many leading bytes of [src, src+n) as are accessible, returning the
// count actually read (0 on total failure). Unlike audio_read_bytes this tolerates a short/partial
// guest range — the NGS2 mixer needs to consume whatever valid PCM a voice still has this render.
size_t audio_read_bytes_partial(uint64_t src, void* dst, size_t n) {
    if (!src || !dst || !n) return 0;
#ifndef _WIN32
    struct iovec l { dst, n }, r { (void*)(uintptr_t)src, n };
    ssize_t got = process_vm_readv(getpid(), &l, 1, &r, 1, 0);
    if (got > 0) return (size_t)got;
    // process_vm_readv is all-or-nothing per iovec; on failure, binary-search the readable prefix so a
    // block that ends partway through the requested span still yields its valid head.
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo + 1) / 2;
        struct iovec l2 { dst, mid }, r2 { (void*)(uintptr_t)src, mid };
        if (process_vm_readv(getpid(), &l2, 1, &r2, 1, 0) == (ssize_t)mid) lo = mid; else hi = mid - 1;
    }
    return lo;
#else
    SIZE_T read = 0;
    ReadProcessMemory(GetCurrentProcess(), (const void*)(uintptr_t)src, dst, n, &read);
    return (size_t)read;
#endif
}

bool a2_store_u32(uint64_t dst, uint32_t v) { return audio_store_bytes(dst, &v, sizeof v); }
bool a2_store_u64(uint64_t dst, uint64_t v) { return audio_store_bytes(dst, &v, sizeof v); }
bool a2_store_zeros(uint64_t dst, size_t n) {
    std::vector<uint8_t> z(n, 0);
    return audio_store_bytes(dst, z.data(), n);
}

// The AudioOut2 grain write-back probe. It lives here, after the fault-safe guest memory
// helpers it is built on, and is used by the mix loop further down.
namespace {

// --- PROSPER_AUDIO_STAMP: is the grain prosper READS the grain the guest WRITES? ---------------
//
// A port whose PCM reads as exactly zero for a whole run has two causes that need opposite fixes,
// and NOTHING in PROSPER_AUDIO_FLOW or PROSPER_AUDIO_LAYOUT can separate them, because every
// statistic those probes publish is computed from the same read:
//   (a) the guest holds the bus open and mixes silence into it — a non-defect, and
//   (b) prosper is reading a buffer the guest never fills — a pointer published once at setup and
//       thereafter stale, a second buffer of a pair we never see, a read taken at the wrong point
//       in the guest's fill cycle. Real audio is being lost, silently and invisibly.
// A wrong address is exactly as zero as a silent mix, so "nonzero=0/55175168" is a statement about
// an instrument until something outside that instrument says otherwise. This is that something.
//
// It settles the question by WRITING. Once the mix has consumed a grain, the probe stamps the
// guest's own grain buffer with a per-channel-distinct pattern, and on the NEXT push classifies
// what came back. The single fact it establishes is whether the guest WRITES this buffer between
// two pushes, which is exactly the fact no amount of reading can supply:
//   OVERWRITTEN  the stamp is gone and non-zero content is in its place -> the guest actively fills
//                this buffer with signal. A live bus carrying audio: the HEALTHY signature, and the
//                one to calibrate the probe against on a port already measured to carry content.
//   CLEARED      the stamp is gone and the grain is exactly zero -> the guest actively writes this
//                buffer and writes SILENCE into it. Case (a), now measured rather than assumed:
//                the port is a live bus with nothing on it, and not prosper's defect.
//   INTACT       the stamp survived byte-for-byte -> the guest did not touch this buffer at all
//                between two pushes. Case (b): prosper is reading memory the guest does not fill,
//                and the port's zero says nothing whatever about the guest's mix.
//   POINTER-MOVED the guest republished a different address before we could read the stamp back —
//                itself a finding (this port is double-buffered), and the reason the probe keys its
//                classification on the address it actually stamped rather than on the port slot.
// Note which way OVERWRITTEN cuts, because the name invites the opposite reading: prosper's own
// push-time read is of the SAME buffer a few microseconds earlier, so content found here is content
// the mix loop got. OVERWRITTEN is a report that the path works, not that something was missed.
//
// The probe is SELF-VALIDATING, which is the whole reason it is trustworthy: the stamp is read back
// IMMEDIATELY after being written, through the same audio_read_bytes_partial() the mix loop uses,
// and a probe whose own read-back does not return the stamp prints PROBE-INVALID and NO verdict.
// That read-back is a positive instance of "this exact port can report a non-zero sample",
// constructed by hand and outside whatever produced the null — a control drawn from the same
// silent guest would only ever have re-confirmed the silence.
//
// Guest memory is left byte-identical: an INTACT stamp is rolled back to the bytes that were there
// before it, and in every other case the guest has already replaced them itself. The probe is
// opt-in, names ONE port, and stops after a bounded number of pushes — the pushes it stamps do
// carry the stamp into the following push's flow/layout statistics, so an unbounded version would
// corrupt the very totals the investigation is reading.
//
//   PROSPER_AUDIO_STAMP=<port>[:<pushes>[:<after>]]
//       port    1-based, as printed by [audio-flow] / [audio-layout].
//       pushes  how many pushes to stamp. Default 8, maximum 64.
//       after   how many of the port's pushes to let past FIRST. Default 0.
// `after` is not a convenience. The probe has to be calibrated against a port whose answer is
// already known, and the only such port is one measured to carry content — but a title's first
// pushes are routinely silent while it boots, so a probe armed at push 0 would classify the
// KNOWN-GOOD port as CLEARED and "confirm" an instrument that was reading the wrong thing. At
// 188 pushes/s, `after` is the port's own clock: 9400 lands the stamp about fifty seconds in.
// A malformed value DISABLES the probe loudly rather than stamping some other port: a typo must
// cost a measurement, never produce a wrong one.
struct StampState {
    uint64_t             addr = 0;        // the address the outstanding stamp was written to
    std::vector<uint8_t> pattern;         // exactly what was written there
    std::vector<uint8_t> original;        // what was there before, for the INTACT rollback
    bool                 pending = false;
    uint32_t             done = 0;        // pushes stamped so far, against the budget
    uint64_t             seen = 0;        // pushes of this port observed, against `after`
    bool                 disabled = false;
};

bool audio_stamp_config(uint32_t& port, uint32_t& pushes, uint64_t& after) {
    static int      parsed = -1;
    static uint32_t cfg_port = 0, cfg_pushes = 8;
    static uint64_t cfg_after = 0;
    if (parsed < 0) {
        parsed = 0;
        if (const char* e = getenv("PROSPER_AUDIO_STAMP"); e && *e) {
            char*               end = nullptr;
            const unsigned long v = strtoul(e, &end, 0);
            bool                ok = end != e && v >= 1 && v <= kA2MaxPorts;
            if (ok && *end == ':') {
                char*               end2 = nullptr;
                const unsigned long n = strtoul(end + 1, &end2, 0);
                if (end2 == end + 1 || n < 1 || n > 64) ok = false;
                else {
                    cfg_pushes = (uint32_t)n;
                    if (*end2 == ':') {
                        char*               end3 = nullptr;
                        const unsigned long a = strtoull(end2 + 1, &end3, 0);
                        if (end3 == end2 + 1 || *end3 != '\0') ok = false;
                        else cfg_after = (uint64_t)a;
                    } else if (*end2 != '\0') {
                        ok = false;
                    }
                }
            } else if (ok && *end != '\0') {
                ok = false;
            }
            if (ok) { cfg_port = (uint32_t)v; parsed = 1; }
            else
                fprintf(stderr, "[audio-stamp] PROSPER_AUDIO_STAMP=\"%s\" is malformed (expected "
                                "<port 1..%u>[:<pushes 1..64>[:<after>]]); probe DISABLED\n",
                        e, (unsigned)kA2MaxPorts);
        }
    }
    port = cfg_port;
    pushes = cfg_pushes;
    after = cfg_after;
    return parsed == 1;
}

// Non-zero SAMPLES per channel, and the run peak, for one interleaved grain. Reported rather than a
// raw non-zero byte count: bytes are not comparable with anything else in this subsystem, whereas
// per-channel sample counts line up directly with the `nonzero%` column of PROSPER_AUDIO_LAYOUT and
// say WHICH channels the guest fills — which is the interesting half of an OVERWRITTEN verdict.
// Returns the TOTAL, which must equal audio_count_nonzero_samples() over the same buffer: the
// verdict is drawn from that shared rule, and this per-channel breakdown only says where the
// samples sit.
uint64_t audio_stamp_channel_stats(const std::vector<uint8_t>& buf, uint32_t channels,
                                   uint32_t data_type, uint64_t* nonzero_out, double& peak_out) {
    for (uint32_t c = 0; c < channels && c < kA2MaxChannels; ++c) nonzero_out[c] = 0;
    peak_out = 0.0;
    uint64_t     total = 0;
    const size_t sample_bytes = data_type == 0 ? sizeof(float) : sizeof(int16_t);
    for (size_t i = 0, n = buf.size() / sample_bytes; i < n; ++i) {
        const float v = data_type == 0 ? reinterpret_cast<const float*>(buf.data())[i]
                                       : (float)reinterpret_cast<const int16_t*>(buf.data())[i] *
                                             (1.0f / 32768.0f);
        if (v != 0.0f) {
            ++total;
            if ((i % channels) < kA2MaxChannels) ++nonzero_out[i % channels];
        }
        const double a = v < 0 ? -(double)v : (double)v;
        if (v == v && a > peak_out) peak_out = a;   // NaN never raises the peak, as elsewhere here
    }
    return total;
}

void audio_stamp_print_channels(const char* label, const uint64_t* nonzero, uint32_t channels,
                                double peak) {
    fprintf(stderr, "               %s peak=%.6f non-zero samples:", label, peak);
    for (uint32_t c = 0; c < channels && c < kA2MaxChannels; ++c)
        fprintf(stderr, " ch%u=%llu", c, (unsigned long long)nonzero[c]);
    fprintf(stderr, "\n");
}

// Per-channel-distinct and every sample non-zero, so the read-back doubles as a channel-identity
// check: whichever channel index a value comes back on is the index prosper's stride arithmetic
// put it at. Magnitudes stay under 0.5 at the widest bed so a stamp that does reach a device is
// not a full-scale bang.
void audio_stamp_fill(std::vector<uint8_t>& out, size_t n, uint32_t channels, uint32_t data_type) {
    out.assign(n, 0);
    if (data_type == 0) {
        auto* s = reinterpret_cast<float*>(out.data());
        for (size_t i = 0, m = n / sizeof(float); i < m; ++i)
            s[i] = (float)((i % channels) + 1) / 32.0f;
    } else {
        auto* s = reinterpret_cast<int16_t*>(out.data());
        for (size_t i = 0, m = n / sizeof(int16_t); i < m; ++i)
            s[i] = (int16_t)(((i % channels) + 1) * 1024);
    }
}

// Called from the mix loop with g_a2_mx held, once per push, for the one selected port, AFTER that
// push's grain has been read and measured — so the probe never perturbs the reading it explains.
void audio_stamp_step(uint32_t port_index, uint64_t pcm, uint32_t channels, uint32_t data_type,
                      uint32_t grain) {
    uint32_t want_port = 0, budget = 0;
    uint64_t after = 0;
    if (!audio_stamp_config(want_port, budget, after)) return;
    if (port_index + 1 != want_port) return;

    static StampState st;
    if (st.disabled) return;
    // The budget gates writing a NEW stamp, never resolving an outstanding one. Returning here on
    // `done >= budget` alone would leave the last stamp unclassified — costing a verdict, and worse,
    // leaving the pattern in the guest's buffer in exactly the INTACT case, which is the one where
    // nothing else ever overwrites it. That would quietly break the byte-identical guarantee at the
    // one address the probe was pointed at.
    const bool budget_spent = st.done >= budget;
    if (budget_spent && !st.pending) return;
    if (!budget_spent && st.seen++ < after) {
        if (st.seen == 1)
            fprintf(stderr, "[audio-stamp] port%u: armed, waiting %llu pushes before the first "
                            "stamp\n", port_index + 1, (unsigned long long)after);
        return;
    }
    const size_t sample_bytes = data_type == 0 ? sizeof(float) : sizeof(int16_t);
    const size_t want = sample_bytes * (size_t)channels * grain;
    if (!pcm || !want || !channels) return;

    // An INDEPENDENT read of the same range: the classification must not inherit whatever the mix
    // loop's read did, or a broken read would validate itself.
    std::vector<uint8_t> now(want);
    const size_t got = audio_read_bytes_partial(pcm, now.data(), want);
    if (got != want) {
        fprintf(stderr, "[audio-stamp] port%u: read 0x%llx +%zu returned %zu bytes; "
                        "probe DISABLED (a partial grain cannot be classified)\n",
                port_index + 1, (unsigned long long)pcm, want, got);
        st.disabled = true;
        return;
    }

    if (st.pending) {
        uint64_t ch_now[kA2MaxChannels]{};
        double   peak_now = 0.0;
        const uint64_t total_now =
            audio_stamp_channel_stats(now, channels, data_type, ch_now, peak_now);
        const bool matches_stamp = st.pattern.size() == now.size() &&
                                   std::memcmp(st.pattern.data(), now.data(), now.size()) == 0;
        switch (audio_classify_stamped_grain(st.addr == pcm, matches_stamp, total_now)) {
        case AudioGrainVerdict::PointerMoved:
            fprintf(stderr, "[audio-stamp] port%u POINTER-MOVED: stamped 0x%llx, this push "
                            "published 0x%llx -- the port is double-buffered, so a zero read of "
                            "either buffer alone proves nothing\n",
                    port_index + 1, (unsigned long long)st.addr, (unsigned long long)pcm);
            break;
        case AudioGrainVerdict::Intact:
            fprintf(stderr, "[audio-stamp] port%u INTACT: the stamp written to 0x%llx last push "
                            "survived byte-for-byte -- the guest did NOT write this buffer between "
                            "pushes, so prosper is reading memory the guest does not fill and this "
                            "port's zero says nothing about the guest's mix\n",
                    port_index + 1, (unsigned long long)st.addr);
            // Leave guest memory exactly as it was found.
            if (!audio_store_bytes(pcm, st.original.data(), st.original.size()))
                fprintf(stderr, "[audio-stamp] port%u: rollback of the intact stamp FAILED; the "
                                "buffer still holds the probe pattern\n", port_index + 1);
            break;
        case AudioGrainVerdict::Cleared:
            fprintf(stderr, "[audio-stamp] port%u CLEARED: the stamp written to 0x%llx last push "
                            "is gone and the grain is exactly zero -- the guest ACTIVELY writes "
                            "this buffer and writes silence into it. The port is a live bus "
                            "carrying no signal, not a buffer prosper is failing to read\n",
                    port_index + 1, (unsigned long long)st.addr);
            break;
        case AudioGrainVerdict::Overwritten:
            // The HEALTHY signature. prosper's own push-time read is of this same buffer a few
            // microseconds earlier, so content found here is content the mix loop received — this
            // says the path works, not that something was missed.
            fprintf(stderr, "[audio-stamp] port%u OVERWRITTEN: the stamp written to 0x%llx last "
                            "push is gone and the guest has filled the buffer with %llu non-zero "
                            "samples -- a live bus carrying signal, which is what a working port "
                            "looks like\n",
                    port_index + 1, (unsigned long long)st.addr, (unsigned long long)total_now);
            audio_stamp_print_channels("guest content:", ch_now, channels, peak_now);
            break;
        }
        st.pending = false;
    }
    // The last stamp has now been classified and, if it survived, rolled back. Stop before writing
    // another one.
    if (budget_spent) return;

    audio_stamp_fill(st.pattern, want, channels, data_type);
    st.original = now;
    if (!audio_store_bytes(pcm, st.pattern.data(), st.pattern.size())) {
        fprintf(stderr, "[audio-stamp] port%u: writing the stamp to 0x%llx FAILED; probe DISABLED "
                        "(guest memory untouched)\n", port_index + 1, (unsigned long long)pcm);
        st.disabled = true;
        return;
    }
    // THE POSITIVE CONTROL, and the reason a verdict from this probe is worth anything: read the
    // stamp back through the very call the mix loop uses. If a non-zero grain written by hand to
    // this port's own buffer does not come back non-zero, the probe cannot distinguish silence
    // from blindness and must not pretend to.
    std::vector<uint8_t> back(want);
    const size_t back_got = audio_read_bytes_partial(pcm, back.data(), want);
    if (back_got != want || std::memcmp(back.data(), st.pattern.data(), want) != 0) {
        fprintf(stderr, "[audio-stamp] port%u PROBE-INVALID: read back %zu of %zu stamped bytes "
                        "and they do not match what was written -- this port's read path cannot be "
                        "shown to report a non-zero sample, so NO verdict is drawn from it\n",
                port_index + 1, back_got, want);
        st.disabled = true;
        st.pending = false;
        return;
    }
    // Report the control quantitatively, per channel, so "the reader could have seen it" is a
    // number in the log rather than an assertion in a PR.
    uint64_t ch_back[kA2MaxChannels]{};
    double   peak_back = 0.0;
    audio_stamp_channel_stats(back, channels, data_type, ch_back, peak_back);
    fprintf(stderr, "[audio-stamp] port%u stamp %u/%u written to 0x%llx (%zu bytes, %u ch); "
                    "read-back OK -- this port CAN report a non-zero sample\n",
            port_index + 1, st.done + 1, budget, (unsigned long long)pcm, want, channels);
    audio_stamp_print_channels("positive control:", ch_back, channels, peak_back);
    st.addr = pcm;
    st.pending = true;
    ++st.done;
    if (st.done >= budget)
        fprintf(stderr, "[audio-stamp] port%u: stamp budget (%u) spent; one more verdict follows on "
                        "the next push, then the probe is done\n", port_index + 1, budget);
}

} // namespace

// ---- libSceAudioIn: headless, silent, real-time paced ----------------------------------
//
// The inherited PS4/PS5 core ABI has seven ports. Handles carry the public 0x30000000 tag,
// the port type in bits 16..23, and a zero-based port id in bits 0..7. We deliberately do not
// open a host microphone: returning exact-size silence gives privacy-preserving deterministic
// behavior, while grain/frequency pacing preserves the blocking capture-thread contract.
namespace {

constexpr int      kAudioInMaxPorts  = 7;
constexpr uint32_t kAudioInHandleTag = 0x30000000u;
constexpr uint64_t kAudioInErrInvalidHandle = (uint64_t)(int64_t)(int32_t)0x80260101;
constexpr uint64_t kAudioInErrInvalidSize   = (uint64_t)(int64_t)(int32_t)0x80260102;
constexpr uint64_t kAudioInErrInvalidFreq   = (uint64_t)(int64_t)(int32_t)0x80260103;
constexpr uint64_t kAudioInErrInvalidPtr    = (uint64_t)(int64_t)(int32_t)0x80260105;
constexpr uint64_t kAudioInErrInvalidParam  = (uint64_t)(int64_t)(int32_t)0x80260106;
constexpr uint64_t kAudioInErrPortFull      = (uint64_t)(int64_t)(int32_t)0x80260107;
constexpr uint64_t kAudioInErrNotOpened     = (uint64_t)(int64_t)(int32_t)0x80260109;

struct AudioInPort {
    bool in_use = false;
    uint32_t grain = 0;
    uint32_t freq = 0;
    uint32_t channels = 0;
    std::chrono::steady_clock::time_point next{};
};

std::mutex  g_audio_in_mx;
AudioInPort g_audio_in_ports[kAudioInMaxPorts];

int audio_in_port_id(uint64_t raw_handle) {
    uint32_t handle = (uint32_t)raw_handle;
    if ((handle & 0x7f000000u) != kAudioInHandleTag) return -1;
    uint32_t id = handle & 0xffu;
    return id < kAudioInMaxPorts ? (int)id : -1;
}

} // namespace

static void audio_in_reset_ports() {
    std::lock_guard<std::mutex> lk(g_audio_in_mx);
    for (auto& port : g_audio_in_ports) port = {};
}

HLE(audio_in_init) { return 0; }

// sceAudioInOpen(userId, type, index, len, freq, param) -> tagged handle or AudioIn error.
// Public formats are S16 mono (0) and S16 stereo (2); supported rates are 16/48 kHz and the
// hardware grain limit is 1..2048 frames. userId/type/index policy is left to the guest-facing
// service just as in the reference implementation; type is retained in the opaque handle.
HLE(audio_in_open) {
    (void)a0; (void)a2;
    uint32_t grain = (uint32_t)a3;
    uint32_t freq = (uint32_t)a4;
    uint32_t format = (uint32_t)a5;
    if (!grain || grain > 2048) return kAudioInErrInvalidSize;
    if (freq != 16000 && freq != 48000) return kAudioInErrInvalidFreq;
    uint32_t channels;
    if (format == 0) channels = 1;
    else if (format == 2) channels = 2;
    else return kAudioInErrInvalidParam;

    std::lock_guard<std::mutex> lk(g_audio_in_mx);
    for (int id = 0; id < kAudioInMaxPorts; id++) {
        if (g_audio_in_ports[id].in_use) continue;
        g_audio_in_ports[id] = {true, grain, freq, channels, {}};
        return (uint32_t)(kAudioInHandleTag | (((uint32_t)a1 & 0xffu) << 16) | (uint32_t)id);
    }
    return kAudioInErrPortFull;
}

// sceAudioInInput(handle, dst) blocks for one capture grain and returns frames captured. The
// null backend writes exactly grain*channels S16 samples. Fault-safe output preserves the HLE
// process when a guest supplies an inaccessible range.
HLE(audio_in_input) {
    int id = audio_in_port_id(a0);
    if (id < 0) return kAudioInErrInvalidHandle;
    if (!a1) return kAudioInErrInvalidPtr;

    std::chrono::steady_clock::time_point deadline;
    uint32_t grain;
    {
        std::lock_guard<std::mutex> lk(g_audio_in_mx);
        AudioInPort& port = g_audio_in_ports[id];
        if (!port.in_use) return kAudioInErrNotOpened;
        size_t bytes = (size_t)port.grain * port.channels * sizeof(int16_t);
        if (!a2_store_zeros(a1, bytes)) return kAudioInErrInvalidPtr;

        grain = port.grain;
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::nanoseconds((uint64_t)port.grain * 1000000000ull / port.freq);
        if (port.next.time_since_epoch().count() == 0 || port.next < now - duration * 4)
            port.next = now;
        port.next += duration;
        deadline = port.next;
    }
    if (deadline > std::chrono::steady_clock::now()) std::this_thread::sleep_until(deadline);
    return grain;
}

HLE(audio_in_close) {
    int id = audio_in_port_id(a0);
    if (id < 0) return kAudioInErrInvalidHandle;
    std::lock_guard<std::mutex> lk(g_audio_in_mx);
    if (!g_audio_in_ports[id].in_use) return kAudioInErrNotOpened;
    g_audio_in_ports[id] = {};
    return 0;
}

constexpr uint64_t kA2ErrInvalidParam = (uint64_t)(int64_t)(int32_t)0x80268001;
constexpr uint64_t kA2ErrNotReady = (uint64_t)(int64_t)(int32_t)0x80268008;
constexpr uint64_t kA2ErrPortFull = (uint64_t)(int64_t)(int32_t)0x80268012;

// Advance the emulated hardware queue by elapsed 48 kHz grains. Caller holds g_a2_mx. Keeping this
// state independent of the host sink matters because GetQueueLevel is an observable device clock:
// middleware uses it to pace state changes even when the backend happens to consume synchronously.
void audio2_update_queue_locked(A2Context& context,
                                std::chrono::steady_clock::time_point now) {
    if (!context.queued) {
        context.last_queue_update = now;
        return;
    }
    const uint32_t grain = context.grain ? context.grain : 256;
    const auto grain_duration = std::chrono::nanoseconds(
        (long long)grain * 1000000000LL / 48000);
    if (grain_duration.count() <= 0 || context.last_queue_update.time_since_epoch().count() == 0) {
        context.last_queue_update = now;
        return;
    }
    if (now <= context.last_queue_update) return;
    const auto elapsed = now - context.last_queue_update;
    const uint64_t drained = std::min<uint64_t>(
        context.queued, (uint64_t)(elapsed / grain_duration));
    if (!drained) return;
    context.queued -= (uint32_t)drained;
    context.last_queue_update += grain_duration * (long long)drained;
    if (!context.queued) context.last_queue_update = now;
}

// sceAudioOut2Initialize(void) -> 0. Idempotent success (same as sceAudioOutInit).
HLE(audio2_initialize) { A2LOG("sceAudioOut2Initialize"); return 0; }

// sceAudioOut2ContextResetParam(SceAudioOut2ContextParam* param) -> 0.  A reset is not merely a
// memset: titles may retain any default they do not override.  In particular queue_depth=4,
// num_grains=512, and flags=1 are part of the observable ABI.
HLE(audio2_ctx_reset_param) {
    A2LOG("sceAudioOut2ContextResetParam");
    struct ContextParam {
        uint32_t max_ports, max_object_ports, guarantee_object_ports;
        uint32_t queue_depth, num_grains, flags, reserved[10];
    } param{};
    static_assert(sizeof(ContextParam) == 0x40);
    param.max_ports = 256;
    param.max_object_ports = 256;
    param.queue_depth = 4;
    param.num_grains = 512;
    param.flags = 1;
    if (!audio_store_bytes(a0, &param, sizeof param)) return kA2ErrInvalidParam;
    return 0;
}

// sceAudioOut2ContextQueryMemory(const param*, size_t* outSize) -> 0.
// Live: a1 is the out size the guest allocates and passes straight to ContextCreate as
// (mem, memSize). The null backend needs no guest work memory; report a fixed 1 MiB so the
// allocation is real and cheap (the value's only observable effect is that malloc succeeds —
// the garbage value 0x244811c was allocated and accepted in the capture run).
HLE(audio2_ctx_query_memory) {
    A2LOG("sceAudioOut2ContextQueryMemory");
    if (audio2log()) a2_dump("param", a0, 0x40);
    if (!a2_store_u64(a1, 0x100000)) return kA2ErrInvalidParam;
    return 0;
}

// sceAudioOut2GetSpeakerArrayMemorySize(...) -> size (GTA V / PPSA04263, RAGE, issue #1134).
// RETURNS the speaker-array work-memory byte size directly (rax); the guest uses it verbatim as an
// allocation size: `r15 = ret; ptr = allocator->alloc(r15, 0x10)`. Live [RAGE] disassembly at
// eboot+0x2adf25e: arg rdi=8 (speaker/channel config). Stubbed to 0 the guest allocated a ZERO-byte
// speaker-array buffer and then overran it, aborting RAGE audio/streaming init with its int 0x41
// fatal — the deterministic pre-render crash on this title.  The SDK sizing contract is 0x400 bytes
// of base state plus 0x40 per VBAP speaker or 0x100 per ambisonics speaker, with 0x200 extra for 3D.
HLE(audio2_get_speaker_array_memory_size) {
    A2LOG("sceAudioOut2GetSpeakerArrayMemorySize");
    const uint64_t speakers = std::clamp<uint64_t>(a0, 1, 32);
    return 0x400 + speakers * (a2 ? 0x100 : 0x40) + (a1 ? 0x200 : 0);
}

// sceAudioOut2ContextCreate(param*, mem, memSize, Handle* outCtx) -> 0.
HLE(audio2_ctx_create) {
    A2LOG("sceAudioOut2ContextCreate");
    uint32_t grain = 256;
    uint32_t queue_depth = 4;
    if (a0) {
        uint32_t param[5]{};
        if (audio_read_bytes(a0, param, sizeof param)) {
            if (param[3] >= 1 && param[3] <= 64) queue_depth = param[3];
            if (param[4] >= 64 && param[4] <= 4096) grain = param[4];
        }
    }
    std::lock_guard<std::mutex> lk(g_a2_mx);
    for (int i = 0; i < 4; i++) {
        if (g_a2_ctx[i].used) continue;
        const uint32_t generation = audio2_next_generation(g_a2_ctx[i].generation);
        g_a2_ctx[i] = A2Context{};
        g_a2_ctx[i].used = true;
        g_a2_ctx[i].generation = generation;
        g_a2_ctx[i].queue_depth = queue_depth;
        g_a2_ctx[i].grain = grain;
        if (!a2_store_u64(a3, audio2_make_handle(kA2CtxTag, generation, i + 1))) {
            audio2_clear_slot(g_a2_ctx[i]);
            return kA2ErrInvalidParam;
        }
        audio_flow_note_create((uint32_t)i, grain, queue_depth);
        return 0;
    }
    return kA2ErrInvalidParam;
}
HLE(audio2_ctx_destroy) {
    A2LOG("sceAudioOut2ContextDestroy");
    const uint32_t one_based_index = (uint32_t)(a0 & kA2HandleIndexMask);
    if ((a0 & kA2HandleTagMask) != kA2CtxTag || one_based_index < 1 ||
        one_based_index > std::size(g_a2_ctx)) return kA2ErrInvalidParam;
    const uint32_t context_slot = one_based_index - 1;
    std::lock_guard<std::mutex> sink_lk(g_a2_sink_mx[context_slot]);
    bool close_sink = false;
    {
        std::lock_guard<std::mutex> lk(g_a2_mx);
        A2Context* context = audio2_context_locked(a0);
        if (!context) return kA2ErrInvalidParam;
        close_sink = context->sink_opened;
        audio2_clear_slot(*context);
        for (auto& port : g_a2_port_state)
            if (port.used && port.context == a0) audio2_clear_slot(port);
    }
    if (close_sink)
        if (AudioSink* sink = audio_sink()) sink->close(kA2SinkPortBase + (int)context_slot);
    return 0;
}

// sceAudioOut2UserCreate(userId, Handle* out) -> 0. (live: userId=0xff)
HLE(audio2_user_create) {
    A2LOG("sceAudioOut2UserCreate");
    std::lock_guard<std::mutex> lk(g_a2_mx);
    if (!a2_store_u64(a1, kA2UserTag | (uint64_t)++g_a2_users)) return kA2ErrInvalidParam;
    return 0;
}
HLE(audio2_user_destroy) { A2LOG("sceAudioOut2UserDestroy"); return 0; }

// sceAudioOut2PortCreate(ctx, const portParam*, Handle* outPort, ...) -> 0.
HLE(audio2_port_create) {
    A2LOG("sceAudioOut2PortCreate");
    if (audio2log()) a2_dump("portParam", a1, 0x40);
    // AudioOut2PortParam: u16 port_type @0, u32 data_format @4, u32 sampling_freq @8.  The format is
    // NOT a block length: 0x800 means eight-channel float PCM, while 0x100/0x200 mean mono/stereo.
    // Confusing it for `channels * grain` happens to work at grain=256, but fails for every other
    // grain and leaves PortGetState unable to report the format that the guest mixer consumes.
    uint16_t ptype = 0;
    uint32_t data_format = 0;
    uint32_t flags = 0;
    if (a1) {
        struct { uint16_t type, pad; uint32_t format; uint32_t rate, flags; } hdr{};
        if (audio_read_bytes(a1, &hdr, sizeof hdr)) {
            ptype = hdr.type;
            data_format = hdr.format;
            flags = hdr.flags;
        }
    }
    std::lock_guard<std::mutex> lk(g_a2_mx);
    if (!audio2_context_locked(a0)) return kA2ErrInvalidParam;
    // First-free-slot allocation (not a monotonic counter): slots are recycled on PortDestroy, so a
    // title that churns ports over a long session can't exhaust the table. The generation in each
    // returned handle keeps an old identity from aliasing the recycled slot.
    for (uint32_t id = 1; id <= kA2MaxPorts; ++id) {
        if (g_a2_port_state[id - 1].used) continue;
        A2PortState& port = g_a2_port_state[id - 1];
        const uint32_t generation = audio2_next_generation(port.generation);
        port.generation = generation; // failed out-pointer publication still consumes this identity
        const uint64_t handle = audio2_make_handle(kA2PortTag, generation, id);
        if (!a2_store_u64(a2, handle)) return kA2ErrInvalidParam;
        port.used = true;
        port.context = a0;
        port.pcm_ptr = 0;
        port.type = ptype;
        port.data_format = data_format;
        port.flags = flags;
        audio_flow_note_port_create(id - 1, ptype, data_format, flags);
        layout_note_port_create(id - 1);
        return 0;
    }
    return kA2ErrPortFull;
}
HLE(audio2_port_destroy) {
    A2LOG("sceAudioOut2PortDestroy");
    // Clear the slot so a destroyed port stops being mixed (its guest PCM buffer may be freed and
    // reused) and the slot is reusable by the next PortCreate.
    std::lock_guard<std::mutex> lk(g_a2_mx);
    A2PortState* port = audio2_port_locked(a0);
    if (!port) return kA2ErrInvalidParam;
    audio2_clear_slot(*port);
    return 0;
}

// sceAudioOut2PortGetState(port, State* out) -> 0.  Titles read this state before rebuilding their
// speaker mixer, so the active output, channel count, and device volume are observable ABI state.
HLE(audio2_port_get_state) {
    A2LOG("sceAudioOut2PortGetState");
    struct PortState {
        uint16_t output;
        uint8_t num_channels, pad1;
        int16_t volume;
        uint16_t reroute_counter;
        uint32_t flags, pad2;
        uint64_t reserved[6];
    } state{};
    static_assert(sizeof(PortState) == 0x40);
    {
        std::lock_guard<std::mutex> lk(g_a2_mx);
        A2PortState* port = audio2_port_locked(a0);
        if (!port) return kA2ErrInvalidParam;
        state.num_channels = (uint8_t)audio2_format_channels(port->data_format);
    }
    state.output = 1;
    state.volume = 127;
    if (!audio_store_bytes(a1, &state, sizeof state)) return kA2ErrInvalidParam;
    return 0;
}
// sceAudioOut2PortSetAttributes(port, const SceAudioOut2Attribute* attrs, size_t count).
// Attribute = {u32 id, u32 reserved, const void* value, size_t valueSize} (0x18 stride).
// `reserved` is ABI padding and may be indeterminate (GTA V leaves stack-address bits there), so
// it must never be folded into the id. Live-decoded ids:
//   0 = per-grain PCM data pointer (value: a guest pointer qword; port format sizes the samples).
// Unknown ids are skipped fail-visibly under PROSPER_AUDIO2LOG. CONFIDENCE: HIGH (Evergate + GTA V).
HLE(audio2_port_set_attr) {
    A2LOG("sceAudioOut2PortSetAttributes");
    const uint64_t count = a2 <= 32 ? a2 : 32;    // defensive cap; log the clamp fail-visibly
    if (count != a2 && audio2log())
        fprintf(stderr, "[audio2] PortSetAttributes: count %llu clamped to 32\n", (unsigned long long)a2);
    std::lock_guard<std::mutex> lk(g_a2_mx);      // covers port state AND the diagnostic set below
    uint32_t port_slot = 0;
    A2PortState* port = audio2_port_locked(a0, &port_slot);
    if (!port) return kA2ErrInvalidParam;
    if (a2 == 0) return 0;                        // no attributes: a legal no-op on a live port
    for (uint64_t i = 0; i < count; i++) {
        struct { uint32_t id, reserved; uint64_t vptr, vsize; } at{};
        static_assert(sizeof(at) == 0x18);
        if (!audio_read_bytes(a1 + i * 0x18, &at, sizeof at)) break;
        if (at.id == 0 && at.vsize == 8) {
            uint64_t pcm = 0;
            if (audio_read_bytes(at.vptr, &pcm, 8)) {
                port->pcm_ptr = pcm;
                port->pcm_frames = 0;
                port->pcm_pending = pcm != 0;
                const auto* context = audio2_context_locked(port->context);
                const uint32_t frames = context && context->grain >= 64 && context->grain <= 4096
                    ? context->grain : 256;
                const uint32_t channels = audio2_format_channels(port->data_format);
                const uint32_t type = port->data_format & 0x7fu;
                if (pcm && channels >= 1 && channels <= kAudioMaxBedChannels &&
                    (type == 0 || type == 1)) {
                    const size_t sample_bytes = type == 0 ? sizeof(float) : sizeof(int16_t);
                    static thread_local std::vector<uint8_t> source;
                    source.resize(static_cast<size_t>(frames) * channels * sample_bytes);
                    const size_t got = audio_read_bytes_partial(pcm, source.data(), source.size());
                    port->pcm_frames = static_cast<uint32_t>(got / (channels * sample_bytes));
                    for (uint32_t channel = 0; channel < channels; ++channel) {
                        auto& output = port->pcm_channels[channel];
                        output.resize(port->pcm_frames);
                        for (uint32_t frame = 0; frame < port->pcm_frames; ++frame) {
                            const auto* sample = source.data() +
                                (static_cast<size_t>(frame) * channels + channel) * sample_bytes;
                            if (type == 0) {
                                std::memcpy(&output[frame], sample, sizeof(float));
                            } else {
                                int16_t value;
                                std::memcpy(&value, sample, sizeof(value));
                                output[frame] = value * (1.0f / 32768.0f);
                            }
                        }
                    }
                }
                // Reached-ness: the guest handing us a PCM buffer address is the last step before
                // submission, so this separates "never wired up audio" from "wired up, never pushed".
                if (pcm && audio_flow()) ++g_flow_pcm_published;
                layout_note_pcm_ptr(port_slot, pcm);
            }
            if (getenv("PROSPER_AUDIO2_PROBE")) {
                // Keep probe history per AudioOut2 port. Object-heavy titles routinely create
                // more than 16 ports; aliasing this table made their alternating grain buffers
                // look like a state change on every SetAttributes call and flooded the log.
                static uint64_t seen_store[kA2MaxPorts] = {0};
                static uint8_t logged_changes[kA2MaxPorts] = {0};
                if (seen_store[port_slot] != pcm) {
                    seen_store[port_slot] = pcm;
                    if (logged_changes[port_slot] < 4) {
                        ++logged_changes[port_slot];
                        fprintf(stderr, "[audio2-attr] port%llu id=0x%x vptr=0x%llx vsize=%llu -> pcm=0x%llx\n",
                                (unsigned long long)(port_slot + 1), at.id,
                                (unsigned long long)at.vptr, (unsigned long long)at.vsize,
                                (unsigned long long)pcm);
                    }
                }
            }
        } else if (audio2log() || getenv("PROSPER_AUDIO2_PROBE")) {
            static std::set<uint64_t> seen;
            if (seen.insert(at.id).second) {
                uint8_t value[16]{};
                const size_t got = audio_read_bytes_partial(
                    at.vptr, value, std::min<size_t>((size_t)at.vsize, sizeof(value)));
                fprintf(stderr, "[audio2] PortSetAttributes: port%llu unhandled attr id=%u "
                        "size=%llu value:", (unsigned long long)(port_slot + 1), at.id,
                        (unsigned long long)at.vsize);
                for (size_t j = 0; j < got; ++j) fprintf(stderr, " %02x", value[j]);
                fprintf(stderr, "\n");
            }
        }
    }
    return 0;
}

// sceAudioOut2ContextAdvance(ctx) -> 0. Update the public queue clock; PCM submission lives in Push.
HLE(audio2_ctx_advance) {
    A2LOG("sceAudioOut2ContextAdvance");
    uint32_t advance_slot = 0;
    {
        std::lock_guard<std::mutex> lk(g_a2_mx);
        A2Context* context = audio2_context_locked(a0, &advance_slot);
        if (!context) return kA2ErrInvalidParam;
        audio2_update_queue_locked(*context, std::chrono::steady_clock::now());
        if (audio_flow()) {
            std::lock_guard<std::mutex> flk(g_flow_mx);
            ++g_flow_ctx[advance_slot].advances;
        }
    }
    // Reported from Advance as well as Push so a pump loop that advances but never submits still
    // produces interval lines (the `advance=N push=0` signature of case 1).
    audio_flow_report(advance_slot);
    return 0;
}

// sceAudioOut2ContextPush(ctx, flag) -> 0. On hardware Push blocks while the output queue is
// full; the null backend reproduces that as one grain of wall-clock pacing per call (same
// model as RealtimeSilentSink) so CRI's server thread runs at real time, not a hot spin.
HLE(audio2_ctx_push) {
    A2LOG("sceAudioOut2ContextPush");
    uint32_t grain = 256;
    uint32_t context_slot = 0;
    // Reserve one hardware queue slot before mixing the submitted grains. A nonblocking push on a full
    // queue returns NOT_READY; a blocking push waits until the 48 kHz device clock drains a slot.
    for (;;) {
        std::chrono::nanoseconds wait_duration{};
        {
            std::lock_guard<std::mutex> lk(g_a2_mx);
            A2Context* context = audio2_context_locked(a0, &context_slot);
            if (!context) return kA2ErrInvalidParam;
            const auto now = std::chrono::steady_clock::now();
            audio2_update_queue_locked(*context, now);
            const bool was_empty = !context->queued;
            if (audio2_reserve_queue_slot(context->queued, context->queue_depth)) {
                if (was_empty) context->last_queue_update = now;
                grain = context->grain;
                break;
            }
            wait_duration = std::chrono::nanoseconds(
                (long long)(context->grain ? context->grain : 256) * 1000000000LL / 48000);
        }
        if (!a1) {
            if (audio_flow()) {
                std::lock_guard<std::mutex> flk(g_flow_mx);
                ++g_flow_ctx[context_slot].not_ready;
            }
            return kA2ErrNotReady;
        }
        std::this_thread::sleep_for(wait_duration);
    }
    // Mix each active port's current grain into a stereo bed. AudioOut2 data_format encodes the
    // sample type in bits 0..6 (0 = F32, 1 = S16) and channel count in bits 8..15. Read exactly one
    // grain at that width, convert it to the float mix bed, then fold a surround MAIN bed into the
    // stereo host sink. Both sample types are public AudioOut2 formats: for example, 0x200 is F32
    // stereo and 0x201 is S16 stereo.
    static thread_local std::vector<float> bed(4096 * 2);
    static thread_local std::vector<float> tmp;
    static thread_local std::vector<int16_t> tmp_s16;
    bool have_pcm = false;
    bool have_main_stream = false;
    {
        std::lock_guard<std::mutex> lk(g_a2_mx);
        A2Context* c = audio2_context_locked(a0);
        if (!c) return kA2ErrInvalidParam;
        grain = (c->grain >= 64 && c->grain <= 4096) ? c->grain : 256;
        std::memset(bed.data(), 0, sizeof(float) * grain * 2);
        const bool probe = getenv("PROSPER_AUDIO2_PROBE") != nullptr;
        const bool flow = audio_flow();
        const bool layout = audio_layout();
        uint32_t object_ports_with_pcm = 0;
        float object_peak = 0.0f;
        int object_peak_port = 0;
        int pidx_probe = 0;
        for (auto& ps : g_a2_port_state) {
            const int this_pidx = pidx_probe++;
            if (!ps.used || ps.context != a0) continue;
            // A port belonging to this context but carrying no published PCM pointer is a distinct
            // failure from one that was never created: count it rather than skipping invisibly.
            if (!ps.pcm_ptr) {
                if (flow) {
                    std::lock_guard<std::mutex> flk(g_flow_mx);
                    FlowPort& fp = g_flow_port[this_pidx];
                    flow_tag_port(fp, context_slot, ps.type, ps.data_format);
                    ++fp.no_pcm;
                }
                continue;
            }
            const uint32_t data_type = ps.data_format & 0x7fu;
            const uint32_t channels = audio2_format_channels(ps.data_format);
            // #1700: every channel count the fold can place is now mixed — the old `channels > 8`
            // gate discarded a whole MAIN port (2.31 MB/s of real 12-channel content in Dragon
            // Quest VII) while the decoder happily reported 12. What is left is the genuinely
            // unreadable: a sample type prosper cannot size, and a declared width beyond the fold's
            // range, which must be REJECTED rather than clamped — reading a wider grain at the
            // fold's stride would walk the guest's buffer and mix garbage.
            const bool readable = (data_type == 0 || data_type == 1) &&
                                  channels >= 1 && channels <= kAudioMaxBedChannels;
            if (!readable) {
                // A LOUD reject. It is a fatal gap to implement, not a skip to live with, and it
                // previously produced no message at all unless a probe happened to be enabled.
                // Rate-limited per port, not suppressed.
                static uint64_t reject_log[kA2MaxPorts] = {0};
                if ((reject_log[this_pidx]++ % 4096) == 0)
                    fprintf(stderr, "[audio2] port%d DISCARDED: data_format=0x%x is not readable "
                                    "(sample type %u: 0=f32, 1=s16; %u channels, max %u); its guest "
                                    "PCM is being dropped every push\n",
                            this_pidx + 1, ps.data_format, data_type, channels,
                            (unsigned)kAudioMaxBedChannels);
                if (probe) fprintf(stderr, "[audio2-probe] port%d skipped: format=0x%x "
                                           "type=%u channels=%u unsupported\n",
                                           this_pidx + 1, ps.data_format, data_type, channels);
                // A rejected port is exactly where lost audio hides — that is how #1700 itself was
                // found — so under the flow probe, read and MEASURE it anyway without mixing it,
                // whenever the grain can be sized at all. A width the fold cannot place is still a
                // width: the declared count gives a well-defined stride, so the measurement is
                // sound even though the placement is not. Only an unknown sample type is truly
                // unsizable, and that arm records the skip alone.
                const bool sizable = (data_type == 0 || data_type == 1) && channels >= 1;
                if (flow && sizable) {
                    // SAMPLED, not exhaustive, and the cap is load-bearing rather than tidiness.
                    // A FAILING read is the EXPECTED case on this arm — a 20-channel declaration
                    // over a narrower real buffer is exactly what it exists to detect — and
                    // audio_read_bytes_partial binary-searches the readable prefix when the full
                    // range faults. Uncapped, one 255-channel port costs a 4 MiB request, ~22
                    // probe syscalls copying ~2 MiB on average, ~45 MiB of copying per push, at
                    // 188 Hz, WITH g_a2_mx HELD. The pre-#1700 16-channel bound hid that; removing
                    // the bound without a cap would have turned a diagnostic that is meant to be
                    // cheap enough to leave on into one that changes the timing it measures.
                    // A signal statistic does not need the whole grain, so bound the span and say
                    // so: `frames`/`bytes` on this port's line then describe what was MEASURED.
                    constexpr size_t kRejectProbeBytes = 64u * 1024u;
                    const size_t sample_bytes = data_type == 0 ? sizeof(float) : sizeof(int16_t);
                    const size_t frame_bytes = sample_bytes * channels;
                    size_t rej_frames_req = kRejectProbeBytes / frame_bytes;
                    if (rej_frames_req == 0) rej_frames_req = 1;      // one frame minimum, always
                    if (rej_frames_req > grain) rej_frames_req = grain;
                    const size_t rej_samples = rej_frames_req * channels;
                    size_t rej_frames = 0;
                    if (data_type == 0) {
                        if (tmp.size() < rej_samples) tmp.resize(rej_samples);
                        rej_frames = audio_read_bytes_partial(ps.pcm_ptr, tmp.data(),
                                        sizeof(float) * rej_samples) / frame_bytes;
                    } else {
                        if (tmp_s16.size() < rej_samples) tmp_s16.resize(rej_samples);
                        rej_frames = audio_read_bytes_partial(ps.pcm_ptr, tmp_s16.data(),
                                        sizeof(int16_t) * rej_samples) / frame_bytes;
                    }
                    // One lock scope covering tag + counters + samples: tagging, unlocking for the
                    // guest read, then re-locking lets a report land in the window and clear `seen`,
                    // so the measurement never prints.
                    std::lock_guard<std::mutex> flk(g_flow_mx);
                    FlowPort& fp = g_flow_port[this_pidx];
                    flow_tag_port(fp, context_slot, ps.type, ps.data_format);
                    ++fp.skip_fmt;
                    ++fp.reads;
                    fp.frames += rej_frames;
                    fp.bytes += rej_frames * frame_bytes;
                    // Short against what was REQUESTED, not against the grain: the cap is our own
                    // choice, so counting it as a short guest read would report a guest defect.
                    if (rej_frames < rej_frames_req) ++fp.short_read;
                    for (size_t k = 0, nf = rej_frames * channels; k < nf; ++k)
                        flow_add_sample(fp, data_type == 0 ? tmp[k]
                                                          : (float)tmp_s16[k] * (1.0f / 32768.0f));
                } else if (flow) {
                    // Nothing to measure for a grain we cannot even size; record the skip.
                    std::lock_guard<std::mutex> flk(g_flow_mx);
                    FlowPort& fp = g_flow_port[this_pidx];
                    flow_tag_port(fp, context_slot, ps.type, ps.data_format);
                    ++fp.skip_fmt;
                }
                continue;
            }
            have_main_stream |= ps.type == 0;
            // The stamp probes writes to the guest source, independently of the owned submission
            // used by this mix. Opt-in, one port, bounded. See PROSPER_AUDIO_STAMP above.
            audio_stamp_step((uint32_t)this_pidx, ps.pcm_ptr, channels, data_type, grain);
            if (!ps.pcm_pending) {
                if (flow) {
                    std::lock_guard<std::mutex> flk(g_flow_mx);
                    FlowPort& fp = g_flow_port[this_pidx];
                    flow_tag_port(fp, context_slot, ps.type, ps.data_format);
                    ++fp.no_grain;
                }
                continue;
            }
            ps.pcm_pending = false;
            const size_t frames_got = std::min(ps.pcm_frames, grain);
            auto sample_at = [&](size_t frame, uint32_t channel) {
                return ps.pcm_channels[channel][frame];
            };
            // Per-channel layout measurement, before any routing decision, so the fold-down for a
            // wide bed rests on the guest's measured content rather than on an assumed enumeration.
            if (layout)
                layout_add_grain(this_pidx, context_slot, ps.type, ps.data_format, channels, grain,
                                 frames_got, sample_at);
            // Measure EVERY port that yielded bytes, before the MAIN-only routing filter, so signal
            // that prosper discards is attributable to the discard rather than to a silent guest.
            if (flow) {
                std::lock_guard<std::mutex> flk(g_flow_mx);
                FlowPort& fp = g_flow_port[this_pidx];
                flow_tag_port(fp, context_slot, ps.type, ps.data_format);
                ++fp.reads;
                fp.frames += frames_got;
                fp.bytes += frames_got * channels *
                            (data_type == 0 ? sizeof(float) : sizeof(int16_t));
                if (frames_got < grain) ++fp.short_read;
                for (size_t k = 0, nf = frames_got * channels; k < nf; ++k) {
                    const float v = sample_at(k / channels, k % channels);
                    flow_add_sample(fp, v);   // NaN counted, then treated as the silence it becomes
                }
                if (ps.type != 0) ++fp.skip_not_main; else ++fp.mixed;
            }
            if (probe) {
                static uint64_t call_ct[kA2MaxPorts] = {0};
                const size_t nf = frames_got * channels;
                size_t nan_ct = 0; float amax = 0.0f;
                for (size_t k = 0; k < nf; k++) {
                    float v = sample_at(k / channels, k % channels);
                    if (v != v) nan_ct++; else { float a = v < 0 ? -v : v; if (a > amax) amax = a; } }
                // Object-heavy engines keep hundreds of valid but silent ports. Reporting every
                // silent buffer once per 64 grains both hides the active route and perturbs its
                // real-time pump. Sample only ports that actually contain a signal.
                if ((call_ct[this_pidx]++ % 64) == 0 && amax > 0.0f)
                    fprintf(stderr, "[audio2-probe] port%d type=%u fmt=%s ch=%u pcm=0x%llx "
                            "frames=%zu nan=%zu/%zu |max|=%.4g\n", this_pidx + 1, ps.type,
                            data_type == 0 ? "f32" : "s16", channels,
                            (unsigned long long)ps.pcm_ptr, frames_got, nan_ct, nf, amax);
                if ((ps.type & 0xff00u) == 0x0100u) {
                    object_ports_with_pcm++;
                    if (amax > object_peak) {
                        object_peak = amax;
                        object_peak_port = this_pidx + 1;
                    }
                }
            }
            // Only the MAIN port (type 0) drives the host speaker output. Ordinary non-main types
            // route to separate hardware devices (personal/pad speaker, chat, vibration). Object
            // ports also remain unmixed here until their attributes and speaker routing are
            // implemented, but probe mode still measures them above so discarded object audio is
            // visible rather than mistaken for decoder attenuation.
            if (ps.type != 0) {
                continue;
            }
            const float port_gain = (ps.flags & kA2PortFlag20DbHeadroom) ? kA2HeadroomGain : 1.0f;
            // One gain pair per source channel, from the pure (unit-tested) layout function. A
            // channel prosper cannot place gets {0,0}; that is a real gap, so say so out loud
            // rather than letting it look like a quiet channel.
            AudioStereoGain gains[kAudioMaxBedChannels];
            const unsigned unplaced = audio_stereo_downmix(channels, gains, kAudioMaxBedChannels);
            if (unplaced) {
                static uint64_t unplaced_log[kA2MaxPorts] = {0};
                if ((unplaced_log[this_pidx]++ % 4096) == 0)
                    fprintf(stderr, "[audio2] port%d: %u of %u MAIN bed channels have no known "
                                    "stereo position (data_format=0x%x) and contribute nothing; "
                                    "run with PROSPER_AUDIO_LAYOUT=1 to measure their role\n",
                            this_pidx + 1, unplaced, channels, ps.data_format);
            }
            // Flatten the matrix into one tap list per side, keeping ONLY the non-zero gains. This
            // is not an optimization — it is required for correctness. Multiplying every channel by
            // both gains reads channels the fold does not place, and `Inf * 0.0f` and `NaN * 0.0f`
            // are both NaN, so one Inf on a surround channel would poison the OPPOSITE side and an
            // unplaced height channel holding uninitialized guest memory would poison BOTH — the
            // output path then clamps NaN to zero, silencing a port whose content is fine. The old
            // inline chain only ever read a channel on the side it contributed to; preserve exactly
            // that, so a zero-gain channel is never even loaded.
            struct Tap { uint32_t ch; float gain; };
            Tap left_taps[kAudioMaxBedChannels], right_taps[kAudioMaxBedChannels];
            uint32_t left_n = 0, right_n = 0;
            for (uint32_t c = 0; c < channels; ++c) {
                if (gains[c].left  != 0.0f) left_taps[left_n++]   = {c, gains[c].left};
                if (gains[c].right != 0.0f) right_taps[right_n++] = {c, gains[c].right};
            }
            for (size_t fno = 0; fno < frames_got; fno++) {
                float left = 0.0f, right = 0.0f;
                for (uint32_t t = 0; t < left_n; ++t)
                    left += sample_at(fno, left_taps[t].ch) * left_taps[t].gain;
                for (uint32_t t = 0; t < right_n; ++t)
                    right += sample_at(fno, right_taps[t].ch) * right_taps[t].gain;
                bed[fno * 2 + 0] += left * port_gain;
                bed[fno * 2 + 1] += right * port_gain;
            }
            if (frames_got) have_pcm = true;
        }
        if (probe) {
            static uint64_t object_probe_calls = 0;
            if ((object_probe_calls++ % 64) == 0 && object_peak > 0.0f)
                fprintf(stderr, "[audio2-probe] objects: pcm_ports=%u peak_port=%d |max|=%.4g\n",
                        object_ports_with_pcm, object_peak_port, object_peak);
        }
        if (flow) {
            std::lock_guard<std::mutex> flk(g_flow_mx);
            FlowCtx& f = g_flow_ctx[context_slot];
            ++f.pushes;
            if (have_pcm) ++f.pushes_with_pcm;
            // Measure the bed as it stands after mixing (pre-clamp): this is exactly the signal the
            // host sink is about to receive, so a zero here with non-zero per-port peaks localizes
            // the loss to routing rather than to the guest.
            for (uint32_t i = 0; i < grain * 2; ++i) flow_add_bed_sample(f, bed[i]);
        }
    }
    audio_layout_report();   // no locks held here; g_layout_mx is a leaf
    // From this point through host output or silent pacing, serialize with Destroy for this exact
    // sink slot. Revalidate the full generation under g_a2_mx after acquiring the slot mutex.
    std::unique_lock<std::mutex> sink_lk(g_a2_sink_mx[context_slot]);
    auto pace_silently = [&]() -> uint64_t {
        std::chrono::steady_clock::time_point target;
        {
            std::lock_guard<std::mutex> lk(g_a2_mx);
            A2Context* c = audio2_context_locked(a0);
            if (!c) return kA2ErrInvalidParam;
            uint32_t g2 = c->grain ? c->grain : 256;
            auto dur = std::chrono::nanoseconds((long long)g2 * 1000000000LL / 48000);
            auto now = std::chrono::steady_clock::now();
            // (Re)sync if unset or far behind (post-stall) to avoid burst catch-up — same policy
            // as RealtimeSilentSink::output.
            if (c->next.time_since_epoch().count() == 0 || c->next < now - dur * 4) c->next = now;
            c->next += dur;
            target = c->next;
        }
        std::this_thread::sleep_until(target);
        return 0;
    };

    AudioSink* sink = audio_sink();
    if (sink && have_main_stream) {
        // Keep an established stream continuous when a port misses a submission. The cleared bed
        // supplies silence for that interval; replaying its last grain produces a 187.5 Hz buzz at
        // the common 256-frame grain, while skipping output starves the host device queue.
        // Forward the mixed grain to the host device; the sink paces one grain per call in real
        // time (same contract as the v1 sceAudioOutOutput path), so no extra sleep on this path.
        // A FAILED device open must fall through to the silent wall-clock pacing below instead:
        // the SDL3 sink's output() is a no-op on a null stream (no sleep), and returning early on
        // every push would hot-spin the guest's pump thread on hosts with no audio device.
        AudioPortInfo info; info.freq = 48000; info.channels = 2; info.fmt = AudioFmt::F32;
        info.grain = (int)grain;
        bool open_ok = false;
        {
            std::lock_guard<std::mutex> lk(g_a2_mx);
            A2Context* context = audio2_context_locked(a0);
            if (!context) return kA2ErrInvalidParam;
            if (!context->sink_opened) {
                context->sink_opened = true;
                context->sink_open_ok = sink->open(kA2SinkPortBase + (int)context_slot, info);
            }
            open_ok = context->sink_open_ok;
        }
        const int sink_port = kA2SinkPortBase + (int)context_slot;
        if (audio_flow()) {
            std::lock_guard<std::mutex> flk(g_flow_mx);
            FlowCtx& f = g_flow_ctx[context_slot];
            f.sink_opened = true;
            f.sink_open_ok = open_ok;
            if (open_ok) { ++f.sink_grains; f.sink_bytes += (uint64_t)grain * 2 * sizeof(float); }
        }
        if (open_ok) {
            for (uint32_t i = 0; i < grain * 2; i++) {
                float v = bed[i];
                if (v != v) v = 0.0f;                        // NaN -> 0; Inf is caught by the clamp below
                bed[i] = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
            }
            audio_observe_output(sink_port, bed.data(), (int)grain, info);
            sink->output(sink_port, bed.data(), (int)grain);
            audio_flow_report(context_slot);
            return 0;
        }
        if (audio_flow()) {
            std::lock_guard<std::mutex> flk(g_flow_mx);
            ++g_flow_ctx[context_slot].silent_paced;
        }
        const uint64_t paced = pace_silently();
        audio_flow_report(context_slot);
        return paced;
    }
    // No sink / no data yet: keep the silent real-time pacing so the guest's pump thread does not
    // hot-spin (on hardware Push blocks while the output queue is full).
    if (audio_flow()) {
        std::lock_guard<std::mutex> flk(g_flow_mx);
        ++g_flow_ctx[context_slot].silent_paced;
    }
    const uint64_t paced = pace_silently();
    audio_flow_report(context_slot);
    return paced;
}

// Targeted control-surface tracing. AudioOut2 ports feed a pre-mastering speaker bed, so these
// calls are relevant whenever valid decoded PCM reaches the bed but the audible master is wrong.
void audio2_control_probe(const char* name, uint64_t a0, uint64_t a1, uint64_t a2,
                          uint64_t a3, uint64_t a4, uint64_t a5) {
    if (!getenv("PROSPER_AUDIO2_CONTROL_PROBE")) return;
    static std::atomic<uint32_t> count{0};
    const uint32_t n = count.fetch_add(1);
    if (n >= 128) return;
    fprintf(stderr, "[audio2-control] %s(0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx)\n",
            name, (unsigned long long)a0, (unsigned long long)a1,
            (unsigned long long)a2, (unsigned long long)a3,
            (unsigned long long)a4, (unsigned long long)a5);
    const uint64_t args[6] = {a0, a1, a2, a3, a4, a5};
    for (uint32_t i = 0; i < 6; ++i) {
        if (args[i] < 0x10000 || (args[i] & 0xffff000000000000ull)) continue;
        uint8_t bytes[48]{};
        const size_t got = audio_read_bytes_partial(args[i], bytes, sizeof(bytes));
        if (!got) continue;
        fprintf(stderr, "[audio2-control]   a%u[0..%zu]:", i, got);
        for (size_t k = 0; k < got; ++k) fprintf(stderr, " %02x", bytes[k]);
        fprintf(stderr, "\n");
    }
}

HLE(audio2_ctx_set_attr) {
    audio2_control_probe("sceAudioOut2ContextSetAttributes", a0, a1, a2, a3, a4, a5);
    std::lock_guard<std::mutex> lk(g_a2_mx);
    if (!audio2_context_locked(a0)) return kA2ErrInvalidParam;
    return 0;
}
HLE(audio2_get_speaker_info) {
    static std::atomic<uint32_t> probes{0};
    if (probes.fetch_add(1) < 8)
        audio2_control_probe("sceAudioOut2GetSpeakerInfo", a0, a1, a2, a3, a4, a5);
    // The host sink exposed by this HLE is stereo. Report its real speaker mask and positions so
    // guest mixers can construct their final matrix; leaving this output untouched reports zero
    // available speakers and causes engines to attenuate/misroute an otherwise healthy mix.
    struct SpeakerPosition { int16_t azimuth, elevation; };
    struct SpeakerInfo {
        uint8_t type;
        uint8_t reserved0;
        int16_t reserved1;
        uint32_t available_bits;
        uint32_t flags;
        uint32_t reserved2;
        SpeakerPosition positions[16];
    } info{};
    static_assert(sizeof(SpeakerInfo) == 0x50);
    info.type = 0;                 // conventional speaker array
    info.available_bits = 0x3;    // front-left and front-right
    info.positions[0] = {-30, 0};
    info.positions[1] = { 30, 0};
    if (!audio_store_bytes(a0, &info, sizeof(info))) return kA2ErrInvalidParam;
    return 0;
}

bool audio2_speaker_array_valid(uint64_t handle) {
    const uint64_t index = handle & 0xff;
    return (handle & ~0xffull) == kA2SpeakerArrayTag && index >= 1 && index <= 32 &&
           g_a2_speaker_arrays[index - 1];
}

HLE(audio2_speaker_array_create) {
    // Signature: (SpeakerArrayHandle* out, const void* vbap_params, const void* ambi_params).
    // The work-memory and speaker geometry live in those parameter objects; this backend only needs
    // an opaque identity because it computes deterministic stereo coefficients below.
    if (getenv("PROSPER_AUDIO2_PROBE")) {
        fprintf(stderr, "[audio2-speaker] create out=0x%llx vbap=0x%llx ambi=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2);
        const uint64_t params[2] = {a1, a2};
        for (uint32_t p = 0; p < 2; ++p) {
            uint8_t bytes[64]{};
            const size_t got = audio_read_bytes_partial(params[p], bytes, sizeof(bytes));
            if (!got) continue;
            fprintf(stderr, "[audio2-speaker]   param%u[0..%zu]:", p, got);
            for (size_t i = 0; i < got; ++i) fprintf(stderr, " %02x", bytes[i]);
            fprintf(stderr, "\n");
        }
    }
    std::lock_guard<std::mutex> lk(g_a2_mx);
    for (uint64_t i = 0; i < 32; ++i) {
        if (g_a2_speaker_arrays[i]) continue;
        const uint64_t handle = kA2SpeakerArrayTag | (i + 1);
        if (!a2_store_u64(a0, handle)) return kA2ErrInvalidParam;
        g_a2_speaker_arrays[i] = true;
        return 0;
    }
    return kA2ErrPortFull;
}

HLE(audio2_speaker_array_destroy) {
    std::lock_guard<std::mutex> lk(g_a2_mx);
    if (!audio2_speaker_array_valid(a0)) return kA2ErrInvalidParam;
    g_a2_speaker_arrays[(a0 & 0xff) - 1] = false;
    return 0;
}

// The position/spread/downmix arguments use the SysV guest ABI's independent XMM argument sequence;
// handle/output/count/height therefore arrive in RDI/RSI/RDX/RCX. The current deterministic stereo
// fallback does not consume the float inputs, so use the normal integer HLE bridge. This is also
// required on Windows, whose import stubs translate the guest's integer registers but do not yet
// marshal XMM arguments into a typed Microsoft-ABI call.
HLE(audio2_get_speaker_array_coefficients) {
    const uint64_t handle = a0;
    const uint64_t coefficients = a1;
    const uint32_t count = (uint32_t)a2;
    const uint8_t height_aware = (uint8_t)a3;
    if (getenv("PROSPER_AUDIO2_PROBE")) {
        static std::atomic<uint32_t> calls{0};
        const uint32_t call = calls.fetch_add(1);
        if (call < 64)
            fprintf(stderr, "[audio2-speaker] coeff #%u handle=0x%llx out=0x%llx "
                    "count=%u height=%u\n", call + 1, (unsigned long long)handle,
                    (unsigned long long)coefficients, count, height_aware);
    }
    {
        std::lock_guard<std::mutex> lk(g_a2_mx);
        if (!audio2_speaker_array_valid(handle)) return kA2ErrInvalidParam;
    }
    if ((!coefficients && count) || count > 256) return kA2ErrInvalidParam;
    std::vector<float> values(count, 0.0f);
    if (count > 0) values[0] = 1.0f;
    if (count > 1) values[1] = 1.0f;
    if (values.empty()) return 0;
    return audio_store_bytes(coefficients, values.data(), values.size() * sizeof(float))
        ? 0 : kA2ErrInvalidParam;
}

HLE(audio2_get_speaker_array_ambisonics_coefficients) {
    if (getenv("PROSPER_AUDIO2_PROBE")) {
        static std::atomic<uint32_t> calls{0};
        const uint32_t call = calls.fetch_add(1);
        if (call < 64)
            fprintf(stderr, "[audio2-speaker] ambi #%u handle=0x%llx channel=%llu "
                    "out=0x%llx count=%llu\n", call + 1, (unsigned long long)a0,
                    (unsigned long long)a1, (unsigned long long)a2,
                    (unsigned long long)a3);
    }
    std::lock_guard<std::mutex> lk(g_a2_mx);
    if (!audio2_speaker_array_valid(a0) || (!a2 && a3) || a3 > 256) return kA2ErrInvalidParam;
    std::vector<float> values((size_t)a3, 0.0f);
    if (!values.empty()) values[0] = (a1 == 0 || a1 == 64) ? 0.70710677f : 1.0f;
    if (values.empty()) return 0;
    return audio_store_bytes(a2, values.data(), values.size() * sizeof(float))
        ? 0 : kA2ErrInvalidParam;
}

HLE(audio2_mastering_init) {
    audio2_control_probe("sceAudioOut2MasteringInit", a0, a1, a2, a3, a4, a5);
    return 0;
}
HLE(audio2_mastering_term) {
    audio2_control_probe("sceAudioOut2MasteringTerm", a0, a1, a2, a3, a4, a5);
    return 0;
}
HLE(audio2_mastering_set_param) {
    audio2_control_probe("sceAudioOut2MasteringSetParam", a0, a1, a2, a3, a4, a5);
    return 0;
}
HLE(audio2_mastering_get_state) {
    audio2_control_probe("sceAudioOut2MasteringGetState", a0, a1, a2, a3, a4, a5);
    return 0;
}

// Return the device queue state after draining whole grains according to the 48 kHz hardware clock.
// Both outputs are optional in the SDK contract.
HLE(audio2_ctx_get_queue_level) {
    A2LOG("sceAudioOut2ContextGetQueueLevel");
    static std::atomic<uint32_t> control_probes{0};
    if (control_probes.fetch_add(1) < 8)
        audio2_control_probe("sceAudioOut2ContextGetQueueLevel", a0, a1, a2, a3, a4, a5);
    uint32_t queued = 0, available = 4;
    {
        std::lock_guard<std::mutex> lk(g_a2_mx);
        A2Context* context = audio2_context_locked(a0);
        if (!context) return kA2ErrInvalidParam;
        audio2_update_queue_locked(*context, std::chrono::steady_clock::now());
        queued = context->queued;
        available = queued < context->queue_depth ? context->queue_depth - queued : 0;
    }
    if (a1 && !a2_store_u32(a1, queued)) return kA2ErrInvalidParam;
    if (a2 && !a2_store_u32(a2, available)) return kA2ErrInvalidParam;
    return 0;
}

HLE(audio2_get_system_state) {
    A2LOG("sceAudioOut2GetSystemState");
    static std::atomic<uint32_t> control_probes{0};
    if (control_probes.fetch_add(1) < 8)
        audio2_control_probe("sceAudioOut2GetSystemState", a0, a1, a2, a3, a4, a5);
    // { float loudness; u32 pad; u64 reserved[7]; }
    if (!a2_store_zeros(a0, 0x40)) return kA2ErrInvalidParam;
    return 0;
}

// Generic logging probes for the not-yet-exercised remainder of the surface. These functions can
// be called once per grain, so sample each export independently. Their detailed behavior remains a
// stub, but handle and ownership validation is observable and must match the implemented paths.
#define A2_PROBE_LOG(str)                                                \
    A2LOG(str);                                                          \
    static std::atomic<uint32_t> control_probes{0};                      \
    if (control_probes.fetch_add(1) < 8)                                 \
        audio2_control_probe(str, a0, a1, a2, a3, a4, a5)
HLE(audio2_ctx_bed_write) {
    A2_PROBE_LOG("sceAudioOut2ContextBedWrite");
    std::lock_guard<std::mutex> lk(g_a2_mx);
    return audio2_context_locked(a0) ? 0 : kA2ErrInvalidParam;
}
HLE(audio2_port_register) {
    A2_PROBE_LOG("sceAudioOut2PortRegister");
    std::lock_guard<std::mutex> lk(g_a2_mx);
    A2Context* context = audio2_context_locked(a0);
    A2PortState* port = audio2_port_locked(a1);
    return context && port && port->context == a0 ? 0 : kA2ErrInvalidParam;
}
HLE(audio2_port_unregister) {
    A2_PROBE_LOG("sceAudioOut2PortUnregister");
    std::lock_guard<std::mutex> lk(g_a2_mx);
    A2Context* context = audio2_context_locked(a0);
    A2PortState* port = audio2_port_locked(a1);
    return context && port && port->context == a0 ? 0 : kA2ErrInvalidParam;
}
#undef A2_PROBE_LOG

// --- libSceAjm (Audio Job Manager — compressed-audio decode: ATRAC9/MP3/AAC) --------------------
// The core owns AJM handles, builders, synchronous batch execution, guest-memory copies, and result
// sidebands. ATRAC9 is decoded by vendored LibAtrac9; MP3 is delegated through ajm_decoder.hpp to an
// optional general-purpose host codec frontend. Unsupported codecs fail truthfully in the sideband.
// CONFIDENCE: HIGH for the exercised ATRAC9 (Blasphemous 2) and MP3 (GTA V) batch-2 paths.
namespace {
    std::atomic<uint32_t> g_ajm_next{1};   // one non-zero counter for context/instance/batch handles
    // AJM returns signed 32-bit SCE errors. Preserve that ABI in the full HLE return register,
    // matching the AudioOut error constants above rather than leaving the upper half zeroed.
    constexpr uint64_t AJM_ERR_INVALID_CONTEXT   = (uint64_t)(int64_t)(int32_t)0x80930002u;
    constexpr uint64_t AJM_ERR_INVALID_INSTANCE  = (uint64_t)(int64_t)(int32_t)0x80930003u;
    constexpr uint64_t AJM_ERR_INVALID_BATCH     = (uint64_t)(int64_t)(int32_t)0x80930004u;
    constexpr uint64_t AJM_ERR_INVALID_PARAMETER = (uint64_t)(int64_t)(int32_t)0x80930005u;
    // sceAjmInitialize's `config` word is a revision selector carried in its HIGH dword: PS4-style
    // callers pass 0, and PS5 titles pass a non-zero revision. Surveyed across all 39 dumps in the
    // local corpus: 43 modules import sceAjmInitialize, 41 decode to a constant, and the high dword
    // is only ever 0, 2 or 3 — 0x200000000 on PPSA01826 (The Pathless), 0x300000000 on PPSA05143
    // (Little Nightmares III) and PPSA09804 (GRIS, via Wwise's AkSoundEngine.prx). No module passes
    // a constant with a non-zero low dword, so validate that and accept the revision rather than
    // allow-listing values one title at a time.
    constexpr uint64_t AJM_INITIALIZE_PS5_CONFIG = 0x200000000ull;
    constexpr uint64_t AJM_INITIALIZE_CONFIG_RESERVED_MASK = 0xffffffffull;

    // The four pointer-returning builder exports serialize a batch from 8/16-byte chunks. These
    // layout values are shared by Sony's PS4-inherited AJM ABI and the PS5 3.20 export surface.
    enum : uint32_t {
        AJM_IDENT_JOB = 0,
        AJM_IDENT_INPUT_RUN = 1,
        AJM_IDENT_INPUT_CONTROL = 2,
        AJM_IDENT_CONTROL_FLAGS = 3,
        AJM_IDENT_RUN_FLAGS = 4,
        AJM_IDENT_RETURN_ADDRESS = 6,
        AJM_IDENT_INLINE = 7,
        AJM_IDENT_OUTPUT_RUN = 17,
        AJM_IDENT_OUTPUT_CONTROL = 18,
    };
    constexpr uint32_t AJM_INSTANCE_STATISTICS = 0x80000;
    constexpr uint64_t AJM_CONTROL_FLAGS_MASK = 0x000060000000e7ffull;
    constexpr uint64_t AJM_STATISTICS_FLAGS_MASK = 0x00000000c0018007ull;
    constexpr uint64_t AJM_RUN_FLAGS_MASK = 0x0000e00000001fffull;
    constexpr size_t AJM_MAX_BUILDER_BYTES = 64u * 1024u * 1024u;

    struct AjmJobChunk {
        uint32_t header;
        uint32_t size;
    };
    struct AjmFlagsChunk {
        uint32_t header;
        uint32_t flags_low;
    };
    struct AjmBufferChunk {
        uint32_t header;
        uint32_t size;
        uint64_t address;
    };
    struct AjmGuestBuffer {
        uint64_t address;
        uint64_t size;
    };
    static_assert(sizeof(AjmJobChunk) == 8);
    static_assert(sizeof(AjmFlagsChunk) == 8);
    static_assert(sizeof(AjmBufferChunk) == 16);
    static_assert(sizeof(AjmGuestBuffer) == 16);

    uint32_t ajm_chunk_header(uint32_t ident, uint32_t payload = 0) {
        return (ident & 0x3fu) | ((payload & 0xfffffu) << 6u);
    }

    template <typename T>
    void ajm_append(std::vector<uint8_t>& bytes, const T& value) {
        const size_t offset = bytes.size();
        bytes.resize(offset + sizeof(value));
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }

    void ajm_append_buffer(std::vector<uint8_t>& bytes, uint32_t ident,
                           uint64_t address, uint32_t size) {
        ajm_append(bytes, AjmBufferChunk{ajm_chunk_header(ident), size, address});
    }

    void ajm_append_flags(std::vector<uint8_t>& bytes, uint32_t ident, uint64_t flags) {
        ajm_append(bytes, AjmFlagsChunk{ajm_chunk_header(ident, (uint32_t)(flags >> 32)),
                                        (uint32_t)flags});
    }

    uint64_t ajm_write_job(uint64_t destination, uint32_t instance,
                           const std::vector<uint8_t>& payload) {
        if (!destination || payload.size() > UINT32_MAX ||
            payload.size() + sizeof(AjmJobChunk) > AJM_MAX_BUILDER_BYTES) return 0;
        std::vector<uint8_t> job;
        job.reserve(sizeof(AjmJobChunk) + payload.size());
        ajm_append(job, AjmJobChunk{ajm_chunk_header(AJM_IDENT_JOB, instance),
                                    (uint32_t)payload.size()});
        job.insert(job.end(), payload.begin(), payload.end());
        return audio_store_bytes(destination, job.data(), job.size())
            ? destination + job.size() : 0;
    }
}
// --- AJM decode state (batch-2.0 path; see ajm2_decode_batch below) ----------------------------
namespace {
struct AjmDecodeInst {
    uint32_t codec = UINT32_MAX;          // AjmCodecType supplied to InstanceCreate
    uint64_t flags = 0;
    std::unique_ptr<ajm::StreamDecoder> host_dec; // optional MP3/AAC frontend codec
    uint8_t  config[4] = {0};
    bool     have_config = false;
    std::unique_ptr<Atrac9Decoder> at9_dec; // persistent per instance: preserves MDCT overlap across blocks
    uint32_t skip_samples = 0;            // gapless program: priming frames to drop at program start
    uint64_t total_samples = 0;           // gapless program: trimmed frames to deliver (0 = no program)
    uint32_t skip_remaining = 0;          // frames still to drop (counts down from skip_samples)
    uint64_t gapless_delivered = 0;       // trimmed frames delivered for the CURRENT program
    uint64_t decoded_samples = 0;         // cumulative delivered sample-frames (no-program sideband)
    // PCM decoded but not yet delivered to a guest output buffer. Lives with the decoder (NOT per
    // batch): the decoder state already spans batches, so a spill must too — dropping it at a batch
    // boundary would lose audio the guest was told (via iSizeConsumed) it had received.
    std::vector<int16_t> carry;
};
struct AjmDecJob {
    uint32_t instance = 0;
    uint64_t out_addr = 0, result_addr = 0;
    uint32_t out_size = 0;
    // SplitBuffer shape: two outputs model a wrapped ring's two halves. PCM past the first
    // buffer's capacity spills into the second; zero when the job has one output.
    uint64_t out2_addr = 0;
    uint32_t out2_size = 0;
    // sceAjmBatchJobDecodeSplit (#2981): the guest passes ARRAYS of {ptr,size} buffers —
    // AkSoundEngine's live callsite builds up to four split input fragments and one or two
    // output buffers on its stack, PS4-BatchJobRunSplitBufferRa shape.
    uint64_t in_addr[4] = {0, 0, 0, 0};
    uint32_t in_size[4] = {0, 0, 0, 0};
    uint32_t num_in = 0;
};
// SCE_AJM_ERROR_INVALID_PARAMETER — the AJM error space (see the constants above); -1 is not a value
// the guest's error mapping recognizes.
constexpr int32_t kAjm2ErrDecode = (int32_t)0x80930005;
// Monotonic milliseconds for [ajm2] diagnostics: pad-script presses are scheduled in wall seconds,
// so timestamped lifecycle logs let a press at a known time be matched to its AJM traffic (#1097).
uint64_t ajm2_log_ms() {
    static const auto t0 = std::chrono::steady_clock::now();
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
}

// Optional live codec capture. Keeping compressed input and the exact returned PCM in separate,
// per-instance files lets the same stream be decoded independently when diagnosing a codec or ABI
// mismatch. Disabled unless explicitly requested; AJM execution is serialized by g_ajm2_mx.
void ajm2_dump_decode(uint32_t instance, std::span<const uint8_t> input,
                      const int16_t* pcm, uint32_t pcm_bytes) {
    const char* base = getenv("PROSPER_AJM_DUMP");
    if (!base || !*base) return;
    char path[1024];
    auto append = [&](const char* suffix, const void* data, size_t size) {
        if (!data || !size) return;
        const int n = std::snprintf(path, sizeof(path), "%s.inst%u.%s", base, instance, suffix);
        if (n <= 0 || static_cast<size_t>(n) >= sizeof(path)) return;
        if (FILE* f = std::fopen(path, "ab")) {
            std::fwrite(data, 1, size, f);
            std::fclose(f);
        }
    };
    append("mp3", input.data(), input.size());
    append("s16le", pcm, pcm_bytes);
}
std::mutex g_ajm2_mx;
std::map<uint32_t, AjmDecodeInst> g_ajm2_inst;             // instance id -> codec + decode state
std::map<uint64_t, std::vector<AjmDecJob>> g_ajm2_jobs;   // batchInfo -> queued decode jobs
} // namespace

// sceAjmInitialize(u64 config, u32* out_context): create a context. PS4-style callers pass zero;
// PS5 titles pass a revision in the high dword — 0x200000000 on PPSA01826 (The Pathless),
// 0x300000000 on PPSA05143 (Little Nightmares III) and PPSA09804 (GRIS, via Wwise). Allow-listing
// individual revisions is what made this fail on an unseen one, and the failure is not contained:
// a guest that reads SCE_AJM_ERROR_INVALID_PARAMETER as "no audio subsystem" skips creating its
// audio-thread object and then dereferences the null singleton it never built.
//
// Accepting an unknown revision is NOT the #1949 mistake (returning success for something prosper
// does not implement). The two AJM batch ABIs are already discriminated by the import the guest
// resolves, not by this word — prosper registers both sceAjmBatchStartBuffer and sceAjmBatchStart —
// and the downstream decode paths really are implemented, so the guest gets behaviour rather than a
// silent wait. Rejecting, by contrast, is not fail-visible: it surfaces as a SIGSEGV inside
// sceKernelCreateEventFlag, far from the cause.
//
// An unrecognized revision is reported once PER REVISION, and only after the context store succeeds,
// so the line cannot claim "accepted" for a call that then failed. See #1971.
// CONFIDENCE: MED — the reserved low dword holds across all 41 constant call sites in the corpus,
// but two sites compute the low dword at runtime, so it is unfalsified rather than established. The
// meaning of the revision number itself is not known, and prosper's context does not use it.
HLE(ajm_initialize) {
    if (!a1 || (a0 & AJM_INITIALIZE_CONFIG_RESERVED_MASK) != 0) return AJM_ERR_INVALID_PARAMETER;
    if (!a2_store_u32(a1, g_ajm_next.fetch_add(1))) return AJM_ERR_INVALID_PARAMETER;
    if (a0 != 0 && a0 != AJM_INITIALIZE_PS5_CONFIG) {
        // Per-revision, not once globally: a single shared slot lets the first unknown revision
        // consume the notice and hide every later one.
        static std::mutex mx;
        static std::vector<uint64_t> seen;
        bool first = false;
        {
            std::lock_guard<std::mutex> lk(mx);
            if (std::find(seen.begin(), seen.end(), a0) == seen.end() && seen.size() < 16) {
                seen.push_back(a0);
                first = true;
            }
        }
        if (first)
            fprintf(stderr, "[ajm] sceAjmInitialize: unrecognized config revision 0x%llx (accepted)\n",
                    (unsigned long long)a0);
    }
    return 0;
}
HLE(ajm_finalize)         { return 0; }
// sceAjmModuleRegister(u32 context, AjmCodecType codec, s64 reserved): register a codec on the context.
HLE(ajm_module_register)  {
    if (getenv("PROSPER_AUDIOLOG")) fprintf(stderr, "[ajm] ModuleRegister ctx=%llu codec=%llu\n",
                                            (unsigned long long)a0, (unsigned long long)a1);
    return a0 ? 0 : AJM_ERR_INVALID_CONTEXT;
}
HLE(ajm_module_unregister){ return 0; }
// sceAjmInstanceCreate(u32 context, codec, flags, u32* instance): a decoder instance handle.
HLE(ajm_instance_create) {
    if (getenv("PROSPER_AUDIOLOG")) fprintf(stderr, "[ajm] InstanceCreate ctx=%llu codec=%llu flags=0x%llx\n",
                                            (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2);
    if (!a0) return AJM_ERR_INVALID_CONTEXT;
    if (!a3) return AJM_ERR_INVALID_PARAMETER;
    const uint32_t id = g_ajm_next.fetch_add(1);
    if (!a2_store_u32(a3, id)) return AJM_ERR_INVALID_PARAMETER;
    AjmDecodeInst instance{};
    instance.codec = (uint32_t)a1;
    instance.flags = a2;
    if (ajm::DecoderBackend* backend = ajm::decoder_backend())
        instance.host_dec = backend->create((ajm::Codec)instance.codec, instance.flags);
    {
        std::lock_guard<std::mutex> lk(g_ajm2_mx);
        g_ajm2_inst.emplace(id, std::move(instance));
    }
    return 0;
}
// sceAjmInstanceDestroy(u32 context, u32 instance).
HLE(ajm_instance_destroy) {
    if (!a0) return AJM_ERR_INVALID_CONTEXT;
    if (!a1) return AJM_ERR_INVALID_INSTANCE;
    {   // Release this instance's persistent codec state. Instance ids are monotonic, so without
        // this both LibAtrac9 and host decoder state would grow for the process lifetime.
        std::lock_guard<std::mutex> lk(g_ajm2_mx);
        g_ajm2_inst.erase((uint32_t)a1);
    }
    return 0;
}
// sceAjmBatchInitialize(void* pBatchBuffer, size_t szBatchBuffer, SceAjmBatchInfo* pBatchInfo): prepare
// the caller's batch info so its job builders write into pBatchBuffer starting at offset 0. Was
// UNIMPLEMENTED (Evergate calls it first, NID MmpF1XsQiHw): the generic stub returned 0 without touching
// pBatchInfo, so the guest built its audio-decode batch on a garbage buffer/cursor, never ran a decode
// job, and never opened an AudioOut port -> total silence. GTA V confirms the full 40-byte layout:
// { void* pBuffer; size_t offset; size_t size; void* lastGoodJob; void* lastGoodJobReturnAddress; }.
// CONFIDENCE: HIGH — live Evergate construction plus GTA V's initialized 40-byte object and builders.
HLE(ajm_batch_initialize) {
    if (getenv("PROSPER_AUDIOLOG")) { static int n; if (n++ < 8)
        fprintf(stderr, "[ajm] BatchInitialize buf=0x%llx size=%llu info=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2); }
    if (!a0 || !a1 || !a2) return AJM_ERR_INVALID_PARAMETER;
    uint64_t info[5] = { a0, 0, a1, 0, 0 };
    if (!audio_store_bytes(a2, info, sizeof info)) return AJM_ERR_INVALID_PARAMETER;
    {   // Re-initializing a batch discards anything queued on it: a guest that abandons a batch and
        // rebuilds it must not have the old jobs decoded into output buffers it may since have freed.
        std::lock_guard<std::mutex> lk(g_ajm2_mx);
        g_ajm2_jobs.erase(a2);
    }
    return 0;
}
// --- AJM batch walker (diagnostic) -------------------------------------------------------------------
// prosper's AJM builders write jobs into the guest batch buffer with a known chunk format (see
// ajm_write_job). Under PROSPER_AUDIOLOG, walk the submitted batch and log each job's chunk shape
// (instance, flags, input/output buffers) — the wire-format documentation the future REAL decode
// needs. NOTE: no decode runs yet; jobs are parsed and logged only. ATRAC9 batch decode via the
// vendored LibAtrac9 is tracked in #1065.
struct AjmChunkRef { uint32_t ident; uint32_t size; uint64_t address; };

void ajm_execute_batch(uint64_t batch_addr, uint64_t batch_size) {
    const bool log = getenv("PROSPER_AUDIOLOG") != nullptr;
    if (!log) return;   // diagnostic-only until #1065 wires real decode
    if (!batch_addr || !batch_size || batch_size > AJM_MAX_BUILDER_BYTES) return;
    std::vector<uint8_t> buf(batch_size, 0);
    const size_t got = audio_read_bytes_partial(batch_addr, buf.data(), buf.size());
    if (got < 8) return;
    int logged = 0;
    size_t cur = 0;
    while (cur + 8 <= got) {
        uint32_t jhdr = 0, jsize = 0;
        std::memcpy(&jhdr, buf.data() + cur, 4);
        std::memcpy(&jsize, buf.data() + cur + 4, 4);
        if ((jhdr & 0x3fu) != AJM_IDENT_JOB) break;         // not a job header -> end of batch
        const uint32_t instance = (jhdr >> 6) & 0xfffffu;
        const size_t payload_end = cur + 8 + jsize;
        if (payload_end > got) break;
        std::vector<AjmChunkRef> chunks;
        uint64_t run_flags = 0, ctrl_flags = 0;
        size_t pc = cur + 8;
        while (pc + 4 <= payload_end) {
            uint32_t chdr = 0; std::memcpy(&chdr, buf.data() + pc, 4);
            const uint32_t cident = chdr & 0x3fu;
            if (cident == AJM_IDENT_CONTROL_FLAGS || cident == AJM_IDENT_RUN_FLAGS) {
                if (pc + 8 > payload_end) break;
                uint32_t flo = 0; std::memcpy(&flo, buf.data() + pc + 4, 4);
                const uint64_t f = ((uint64_t)((chdr >> 6) & 0xfffffu) << 32) | flo;
                if (cident == AJM_IDENT_RUN_FLAGS) run_flags = f; else ctrl_flags = f;
                pc += 8;
            } else if (cident == AJM_IDENT_INLINE) {
                if (pc + 8 > payload_end) break;   // header straddles the payload end
                uint32_t isz = 0; std::memcpy(&isz, buf.data() + pc + 4, 4);
                pc += 8 + (((size_t)isz + 7u) & ~size_t{7u});
            } else {                                        // buffer chunk (16 bytes)
                if (pc + 16 > payload_end) break;
                uint32_t csize = 0; uint64_t caddr = 0;
                std::memcpy(&csize, buf.data() + pc + 4, 4);
                std::memcpy(&caddr, buf.data() + pc + 8, 8);
                chunks.push_back({cident, csize, caddr});
                pc += 16;
            }
        }
        auto find = [&](uint32_t id) -> const AjmChunkRef* {
            for (const auto& c : chunks) if (c.ident == id) return &c; return nullptr; };
        const AjmChunkRef *in_run = find(AJM_IDENT_INPUT_RUN),  *out_run = find(AJM_IDENT_OUTPUT_RUN);
        const AjmChunkRef *in_ctl = find(AJM_IDENT_INPUT_CONTROL), *out_ctl = find(AJM_IDENT_OUTPUT_CONTROL);
        if (log && logged++ < 32) {
            fprintf(stderr, "[ajm] JOB inst=%u run_flags=0x%llx ctrl_flags=0x%llx chunks=%zu"
                    " in_run=%s out_run=%s in_ctl=%s out_ctl=%s\n", instance,
                    (unsigned long long)run_flags, (unsigned long long)ctrl_flags, chunks.size(),
                    in_run?"y":"-", out_run?"y":"-", in_ctl?"y":"-", out_ctl?"y":"-");
            if (in_ctl && in_ctl->size && in_ctl->size <= 64) {
                uint8_t cb[64] = {}; audio_read_bytes_partial(in_ctl->address, cb, in_ctl->size);
                fprintf(stderr, "[ajm]   in_ctl[%u]:", in_ctl->size);
                for (uint32_t i = 0; i < in_ctl->size; i++) fprintf(stderr, " %02x", cb[i]);
                fprintf(stderr, "\n");
            }
            if (in_run) fprintf(stderr, "[ajm]   in_run addr=0x%llx size=%u  out_run addr=0x%llx size=%u  out_ctl size=%u\n",
                    (unsigned long long)in_run->address, in_run->size,
                    (unsigned long long)(out_run?out_run->address:0), out_run?out_run->size:0, out_ctl?out_ctl->size:0);
        }
        (void)in_run; (void)out_run; (void)out_ctl; (void)instance;
        cur = payload_end;
    }
}

// sceAjmBatchStartBuffer(context, batch, size, prio, AjmBatchError* err, u32* out_batch_id): accept a
// decode batch and report it started (out_batch_id filled). We don't run the jobs; BatchWait completes
// it. The AjmBatchError out (a4) is left as the caller's value (its layout isn't needed for no-error).
HLE(ajm_batch_start) {
    if (getenv("PROSPER_AUDIOLOG")) fprintf(stderr, "[ajm] BatchStart ctx=%llu batch=0x%llx size=%llu prio=%llu\n",
            (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2, (unsigned long long)a3);
    if (!a0) return AJM_ERR_INVALID_CONTEXT;
    if (!a5) return AJM_ERR_INVALID_PARAMETER;
    ajm_execute_batch(a1, a2);   // diagnostic walk/log of the submitted jobs (no decode yet: #1065)
    if (!a2_store_u32(a5, g_ajm_next.fetch_add(1))) return AJM_ERR_INVALID_PARAMETER;
    return 0;
}
HLE(ajm_batch_wait)       {
    if (getenv("PROSPER_AUDIOLOG")) { static int n; if (n++ < 40)
        fprintf(stderr, "[ajm] BatchWait ctx=%llu batch=%llu\n", (unsigned long long)a0, (unsigned long long)a1); }
    return a0 ? 0 : AJM_ERR_INVALID_CONTEXT; }   // batch completed
HLE(ajm_batch_cancel) {
    if (!a0) return AJM_ERR_INVALID_CONTEXT;
    if (!a1) return AJM_ERR_INVALID_BATCH;
    return 0;
}
HLE(ajm_batch_errordump)  { return 0; }

// Build one control job and return the next free byte in the caller's batch buffer. Returning zero
// here is not a harmless stub result: the guest chains the returned cursor into its next builder.
HLE8(ajm_batch_job_control_buffer_ra) {
    if (getenv("PROSPER_AUDIOLOG")) { static int n; if (n++ < 40)
        fprintf(stderr, "[ajm] JobControlRa batch=0x%llx inst=%llu flags=0x%llx in=0x%llx/%llu out=0x%llx/%llu ra=0x%llx\n",
            (unsigned long long)a0,(unsigned long long)a1,(unsigned long long)a2,(unsigned long long)a3,
            (unsigned long long)a4,(unsigned long long)a5,(unsigned long long)a6,(unsigned long long)a7); }
    if (a4 > UINT32_MAX || a6 > UINT32_MAX) return 0;
    std::vector<uint8_t> payload;
    payload.reserve(a7 ? 56 : 40);
    if (a7) ajm_append_buffer(payload, AJM_IDENT_RETURN_ADDRESS, a7, 0);
    ajm_append_buffer(payload, AJM_IDENT_INPUT_CONTROL, a3, (uint32_t)a4);
    const uint64_t mask = (uint32_t)a1 == AJM_INSTANCE_STATISTICS
        ? AJM_STATISTICS_FLAGS_MASK : AJM_CONTROL_FLAGS_MASK;
    ajm_append_flags(payload, AJM_IDENT_CONTROL_FLAGS, a2 & mask);
    ajm_append_buffer(payload, AJM_IDENT_OUTPUT_CONTROL, a5, (uint32_t)a6);
    return ajm_write_job(a0, (uint32_t)a1, payload);
}

// Store caller data inline after an AJM inline header. The reported batch address points at the
// copied payload; the next cursor is rounded up to AJM's required 8-byte boundary.
HLE(ajm_batch_job_inline_buffer) {
    if (!a0 || !a3 || a2 > UINT32_MAX || a2 > AJM_MAX_BUILDER_BYTES - sizeof(AjmJobChunk))
        return 0;
    const size_t data_size = (size_t)a2;
    const size_t aligned_size = (data_size + 7u) & ~size_t{7u};
    std::vector<uint8_t> bytes(sizeof(AjmJobChunk) + aligned_size, 0);
    const AjmJobChunk header{ajm_chunk_header(AJM_IDENT_INLINE), (uint32_t)aligned_size};
    std::memcpy(bytes.data(), &header, sizeof(header));
    if (data_size && !audio_read_bytes(a1, bytes.data() + sizeof(header), data_size)) return 0;
    if (!audio_store_bytes(a0, bytes.data(), bytes.size()) ||
        !a2_store_u64(a3, a0 + sizeof(AjmJobChunk))) return 0;
    return a0 + bytes.size();
}

HLE10(ajm_batch_job_run_buffer_ra) {
    if (getenv("PROSPER_AUDIOLOG")) { static int n; if (n++ < 40)
        fprintf(stderr, "[ajm] JobRunRa batch=0x%llx inst=%llu flags=0x%llx in=0x%llx/%llu out=0x%llx/%llu sb=0x%llx/%llu ra=0x%llx\n",
            (unsigned long long)a0,(unsigned long long)a1,(unsigned long long)a2,(unsigned long long)a3,(unsigned long long)a4,
            (unsigned long long)a5,(unsigned long long)a6,(unsigned long long)a7,(unsigned long long)a8,(unsigned long long)a9); }
    if (a4 > UINT32_MAX || a6 > UINT32_MAX || a8 > UINT32_MAX) return 0;
    std::vector<uint8_t> payload;
    payload.reserve(a9 ? 72 : 56);
    if (a9) ajm_append_buffer(payload, AJM_IDENT_RETURN_ADDRESS, a9, 0);
    ajm_append_buffer(payload, AJM_IDENT_INPUT_RUN, a3, (uint32_t)a4);
    ajm_append_flags(payload, AJM_IDENT_RUN_FLAGS, a2 & AJM_RUN_FLAGS_MASK);
    ajm_append_buffer(payload, AJM_IDENT_OUTPUT_RUN, a5, (uint32_t)a6);
    ajm_append_buffer(payload, AJM_IDENT_OUTPUT_CONTROL, a7, (uint32_t)a8);
    return ajm_write_job(a0, (uint32_t)a1, payload);
}

HLE10(ajm_batch_job_run_split_buffer_ra) {
    if (getenv("PROSPER_AUDIOLOG")) { static int n; if (n++ < 40)
        fprintf(stderr, "[ajm] JobRunSplitRa batch=0x%llx inst=%llu\n", (unsigned long long)a0,(unsigned long long)a1); }
    if (a4 > UINT32_MAX || a6 > UINT32_MAX || a8 > UINT32_MAX) return 0;
    const uint64_t chunk_count = a4 + a6 + 2u + (a9 ? 1u : 0u);
    if (chunk_count > (AJM_MAX_BUILDER_BYTES - sizeof(AjmJobChunk)) /
                      sizeof(AjmBufferChunk)) return 0;
    if ((a4 && !a3) || (a6 && !a5)) return 0;

    std::vector<uint8_t> payload;
    payload.reserve((size_t)chunk_count * sizeof(AjmBufferChunk));
    if (a9) ajm_append_buffer(payload, AJM_IDENT_RETURN_ADDRESS, a9, 0);
    for (uint64_t i = 0; i < a4; ++i) {
        AjmGuestBuffer buffer{};
        if (!audio_read_bytes(a3 + i * sizeof(buffer), &buffer, sizeof(buffer)) ||
            buffer.size > UINT32_MAX) return 0;
        ajm_append_buffer(payload, AJM_IDENT_INPUT_RUN, buffer.address, (uint32_t)buffer.size);
    }
    ajm_append_flags(payload, AJM_IDENT_RUN_FLAGS, a2 & AJM_RUN_FLAGS_MASK);
    for (uint64_t i = 0; i < a6; ++i) {
        AjmGuestBuffer buffer{};
        if (!audio_read_bytes(a5 + i * sizeof(buffer), &buffer, sizeof(buffer)) ||
            buffer.size > UINT32_MAX) return 0;
        ajm_append_buffer(payload, AJM_IDENT_OUTPUT_RUN, buffer.address, (uint32_t)buffer.size);
    }
    ajm_append_buffer(payload, AJM_IDENT_OUTPUT_CONTROL, a7, (uint32_t)a8);
    return ajm_write_job(a0, (uint32_t)a1, payload);
}

// --- AJM batch-2.0 compressed-audio decode (Blasphemous 2 / GTA V) -----------------------------
// B2's FMOD decodes ATRAC9 and GTA V decodes MP3 through the newer BatchJob* builders (distinct from
// the RunBufferRa model above). ABI recovered from live PROSPER_AUDIOLOG captures:
//   sceAjmBatchJobInitialize(batchInfo, instance, cfgPtr, cfgSize=8, resultPtr, ...)
//        cfgPtr[0..3] = the 4-byte ATRAC9 ConfigData (starts 0xFE; e.g. FE 72 09 F0 = 48k stereo).
//   sceAjmBatchJobSetGaplessDecode(batchInfo, instance, {u32 total, u32 skip}, 1, resultPtr, ...)
//   sceAjmBatchJobDecode(batchInfo, instance, in, inSize, out, outSize, resultPtr, retAddr, 0, resultPtr)
//        decode inSize compressed bytes at `in` into interleaved S16 PCM at `out` (<= outSize).
//   sceAjmBatchStart(context, batchInfo, priority, ...) -> run all jobs queued on batchInfo.
// prosper accepts the batches, decodes each job through vendored LibAtrac9 or an optional host codec,
// writes PCM into the guest output buffer, and fills the result sideband. The guest then mixes it into
// its AudioOut2 MAIN port. CONFIDENCE: HIGH — live B2 and GTA V evidence plus focused codec tests.
namespace {

// Write the AJM decode result sideband (published contract), into the 32-byte decode buffer:
//   SceAjmSidebandResult { s32 iResult; s32 iCodecResult; }          (0 / 0 = success)
//   SceAjmSidebandStream { u32 iSizeConsumed; u32 iSizeProduced;     (input bytes used / PCM bytes out)
//                          u64 uiTotalDecodedSamples; }
//   SceAjmSidebandMFrame { u32 numFrames; u32 reserved; }            (codec frames decoded by this job)
// uiTotalDecodedSamples is load-bearing, not padding: with it left zero the guest's mixer (FMOD) stops
// after a single batch, so it carries the instance's running sample-frame total.
bool ajm2_write_result(uint64_t result_addr, int32_t err, uint32_t consumed, uint32_t produced,
                       uint64_t total_samples = 0, uint32_t decoded_frames = 0) {
    if (!result_addr) return true;
    struct Sideband { int32_t iResult; int32_t iCodecResult; uint32_t iSizeConsumed;
                      uint32_t iSizeProduced; uint64_t uiTotalDecodedSamples;
                      uint32_t numFrames; uint32_t reserved; };
    static_assert(sizeof(Sideband) == 32, "AJM decode sideband must include the MFrame result");
    Sideband sb{ err, 0, consumed, produced, total_samples, decoded_frames, 0 };
    return audio_store_bytes(result_addr, &sb, sizeof sb);
}

// Decode a batch's queued jobs. Jobs sharing an instance are consecutive stream blocks whose input
// buffers are contiguous in guest memory, so we treat each instance's jobs as ONE continuous ATRAC9
// stream: decode whole superframes in order (frames within a superframe aren't independently
// decodable, so partial-superframe decoding would desync the decoder), and route the decoded PCM
// into the jobs' output buffers in sequence, filling each completely. Decoding is bounded by the
// TOTAL output capacity across the instance's jobs; the compressed bytes actually consumed are
// reported per job via iSizeConsumed, so the guest re-submits whatever it provided beyond capacity.
void ajm2_decode_batch(std::vector<AjmDecJob>& jobs) {
    for (size_t ji = 0; ji < jobs.size();) {
        const uint32_t inst_id = jobs[ji].instance;
        size_t je = ji; while (je < jobs.size() && jobs[je].instance == inst_id) ++je;  // [ji,je) same instance
        auto it = g_ajm2_inst.find(inst_id);
        // InstanceCreate normally constructs optional codecs. Also retry lazily in case a test or
        // embedding frontend installed its backend after creating the guest instance.
        if (it != g_ajm2_inst.end() && !it->second.host_dec &&
            it->second.codec != UINT32_MAX) {
            if (ajm::DecoderBackend* backend = ajm::decoder_backend())
                it->second.host_dec = backend->create((ajm::Codec)it->second.codec, it->second.flags);
        }
        if (it != g_ajm2_inst.end() && it->second.host_dec && it->second.host_dec->valid()) {
            AjmDecodeInst& instance = it->second;
            const bool log = getenv("PROSPER_AUDIOLOG") != nullptr;
            for (size_t k = ji; k < je; ++k) {
                AjmDecJob& job = jobs[k];
                if (!instance.host_dec->valid()) {
                    ajm2_write_result(job.result_addr, kAjm2ErrDecode, 0, 0,
                                      instance.decoded_samples);
                    continue;
                }
                // BatchJobDecodeSplit (#2981): stage the second fragment as the contiguous
                // continuation of the first — the StreamDecoder seam takes a single span.
                // FFmpeg reads AV_INPUT_BUFFER_PADDING_SIZE (64) bytes past the packet it is
                // handed; the guest buffer makes no such promise, so stage zero padding after
                // the fragment. Without this the batch path parses heap tail as a packet header
                // (exposed by the Ajm Opus path, #2981 — the stream seam already padded).
                size_t in_total = 0;
                for (uint32_t f = 0; f < job.num_in; ++f) in_total += job.in_size[f];
                std::vector<uint8_t> input(in_total + ajm::kStreamInputPadding, 0);
                size_t got = 0;
                for (uint32_t f = 0; f < job.num_in; ++f)
                    got += audio_read_bytes_partial(job.in_addr[f], input.data() + got,
                                                    job.in_size[f]);
                const uint32_t out_total =
                    job.out_size + (job.out2_size ? job.out2_size : 0);
                std::vector<int16_t> pcm(out_total / sizeof(int16_t));
                const ajm::DecodeResult decoded = instance.host_dec->decode(
                    std::span<const uint8_t>(input.data(), got), std::span<int16_t>(pcm));
                const uint32_t frame_bytes = decoded.channels * sizeof(int16_t);
                // A streaming parser may consume a compressed prefix before it has one complete
                // frame and therefore before it can report the stream's channel count. That is a
                // successful zero-output job: requiring frame_bytes here would report an error
                // after advancing persistent parser state, prompting the guest to resend bytes we
                // already consumed. Channel alignment becomes meaningful only once PCM is emitted.
                const bool pcm_shape_valid = decoded.produced_bytes == 0 ||
                    (frame_bytes != 0 && decoded.produced_bytes % frame_bytes == 0);
                const bool valid = decoded.ok && decoded.consumed_bytes <= got &&
                    decoded.produced_bytes <= out_total &&
                    decoded.produced_bytes <= pcm.size() * sizeof(int16_t) &&
                    pcm_shape_valid;
                // A failed result publishes zero consumption. The backend may already have
                // advanced parser/codec state, so it must become terminal before the guest retries
                // the same compressed prefix. This also covers a malformed backend result rejected
                // by the guest-facing validation above.
                if (!valid) instance.host_dec->invalidate();
                int32_t err = valid ? 0 : kAjm2ErrDecode;
                uint32_t consumed = valid ? decoded.consumed_bytes : 0;
                uint32_t produced = valid ? decoded.produced_bytes : 0;
                // Destination split, resolved before any store: a wrapped ring's two halves are
                // one logical buffer. The split point is the first descriptor's capacity aligned
                // DOWN to whole sample-frames — a mid-frame split would transpose L/R for the
                // entire second half of every wrapping job (#2988 review B). produced is already
                // frame-aligned (pcm_shape_valid), so only the boundary needs alignment.
                uint32_t first = produced;
                if (produced > job.out_size && job.out2_size)
                    first = job.out_size - (job.out_size % frame_bytes);
                if (produced - first > job.out2_size) {
                    // The aligned split cannot fit the second half in the second descriptor —
                    // only reachable when the guest's descriptors are frame-misaligned AND
                    // produced nears out_total (the bound is out2_size + out_size%frame_bytes).
                    // Fail visibly rather than overrun the guest buffer or drop samples
                    // silently (#2988 review B2). The terminal invalidate is deliberate: a
                    // misaligned descriptor pair is guest state, not decoder corruption, but
                    // per-job rejection would re-fail every retry — and an aligned ring (every
                    // shape GRIS submits) never reaches this branch.
                    instance.host_dec->invalidate();
                    err = kAjm2ErrDecode;
                    consumed = produced = 0;
                    first = 0;
                }
                if (produced && !audio_store_bytes(job.out_addr, pcm.data(), first)) {
                    instance.host_dec->invalidate();
                    err = kAjm2ErrDecode;
                    consumed = produced = 0;
                }
                if (!err && produced > first &&
                    !audio_store_bytes(job.out2_addr, pcm.data() + first / sizeof(int16_t),
                                       produced - first)) {
                    instance.host_dec->invalidate();
                    err = kAjm2ErrDecode;
                    consumed = produced = 0;
                }
                if (!err) ajm2_dump_decode(inst_id,
                    std::span<const uint8_t>(input.data(), consumed), pcm.data(), produced);
                if (!err && produced) instance.decoded_samples += produced / frame_bytes;
                const bool result_published = ajm2_write_result(
                    job.result_addr, err, consumed, produced, instance.decoded_samples,
                    err ? 0 : decoded.decoded_frames);
                if (!result_published) {
                    // The codec and guest PCM may already have advanced, but the guest did not
                    // receive the progress sideband. Preserve the same terminal invariant as every
                    // other unpublishable result before another queued job reaches the decoder.
                    instance.host_dec->invalidate();
                }
                if (log)
                    fprintf(stderr, "[ajm2] t=%llums decode inst=%u codec=%u in=%zu/%uB "
                            "out=%u%s -> %u consumed, %u PCM, %uch/%uHz total=%llu%s\n",
                            (unsigned long long)ajm2_log_ms(), inst_id, instance.codec,
                            got, (unsigned)in_total, job.out_size,
                            job.out2_size ? "+2nd" : "", consumed, produced, decoded.channels,
                            decoded.sample_rate, (unsigned long long)instance.decoded_samples,
                            !result_published ? " RESULT_STORE_ERR" : (err ? " ERR" : ""));
            }
            ji = je;
            continue;
        }
        // A non-null invalid host decoder is deliberately terminal. Do not fall through to the
        // ATRAC9 path or erase the cumulative sample count on a later batch's error sideband.
        if (it != g_ajm2_inst.end() && it->second.host_dec) {
            for (size_t k = ji; k < je; ++k)
                ajm2_write_result(jobs[k].result_addr, kAjm2ErrDecode, 0, 0,
                                  it->second.decoded_samples);
            if (getenv("PROSPER_AUDIOLOG"))
                fprintf(stderr, "[ajm2] decode inst=%u SKIP n=%zu host_dec INVALID\n",
                        inst_id, je - ji);
            ji = je;
            continue;
        }
        Atrac9Decoder* dec = (it != g_ajm2_inst.end() && it->second.at9_dec && it->second.at9_dec->valid())
                             ? it->second.at9_dec.get() : nullptr;
        if (!dec) {
            const uint64_t total = it != g_ajm2_inst.end()
                ? (it->second.total_samples ? it->second.gapless_delivered
                                            : it->second.decoded_samples)
                : 0;
            for (size_t k = ji; k < je; ++k)
                ajm2_write_result(jobs[k].result_addr, kAjm2ErrDecode, 0, 0, total);
            if (getenv("PROSPER_AUDIOLOG"))
                fprintf(stderr, "[ajm2] decode inst=%u SKIP n=%zu no at9/host (inst_%s)\n",
                        inst_id, je - ji,
                        it == g_ajm2_inst.end() ? "MISSING" : "present");
            ji = je;
            continue;
        }
        const int ch = dec->channels();
        const int sfb = dec->superframe_bytes();
        const int sfs = dec->superframe_samples();
        if (ch <= 0 || sfb <= 0 || sfs <= 0) {
            for (size_t k = ji; k < je; ++k) ajm2_write_result(jobs[k].result_addr, kAjm2ErrDecode, 0, 0); ji = je; continue; }
        const uint32_t frame_bytes = (uint32_t)ch * sizeof(int16_t);   // one interleaved sample-frame
        const uint32_t sf_out_bytes = (uint32_t)sfs * frame_bytes;
        std::vector<int16_t> pcm((size_t)sfs * ch);
        std::vector<int16_t>& carry = it->second.carry;   // spans batches: the decoder state does too
        std::vector<uint8_t> in;
        const bool log = getenv("PROSPER_AUDIOLOG") != nullptr;

        for (size_t k = ji; k < je; ++k) {
            AjmDecJob& job = jobs[k];
            AjmDecodeInst& I = it->second;
            // A prior job may have advanced ATRAC9 state and then failed to publish its result.
            // Such an instance is terminal: later jobs must report zero progress instead of
            // decoding against state the guest could not observe.
            if (!dec) {
                ajm2_write_result(job.result_addr, kAjm2ErrDecode, 0, 0,
                                  I.total_samples ? I.gapless_delivered : I.decoded_samples);
                continue;
            }
            // A programmed gapless decode (#1097): drop `skip` priming frames at the program start,
            // deliver EXACTLY `total` trimmed frames, then pin the reported total there with no
            // further PCM. FMOD's channel-end and codec-slot recycling key off that exact landing;
            // overshooting it (whole raw superframes, cumulative counter) left every one-shot SFX
            // "still playing" forever, so codec slots were never recycled and the pool exhausted
            // after ~32 sounds -- every later sound silent.
            const bool prog = I.total_samples > 0;
            // Over-allocate by a full superframe: LibAtrac9's bit reader is unbounded, so a truncated
            // or corrupt block's parse may step past the nominal superframe. decode_superframe still
            // rejects such a parse, but the read must land in memory we own.
            size_t in_total = 0;
            for (uint32_t f = 0; f < job.num_in; ++f) in_total += job.in_size[f];
            in.assign(in_total + (size_t)sfb, 0);
            size_t got = 0;
            for (uint32_t f = 0; f < job.num_in; ++f)
                got += audio_read_bytes_partial(job.in_addr[f], in.data() + got, job.in_size[f]);
            // Only ever hand the guest whole sample-frames: a byte-granular partial write would shift
            // the interleave and swap L/R for the rest of the stream.
            const uint32_t out_cap = job.out_size - (job.out_size % frame_bytes);
            uint32_t in_cur = 0, produced = 0, decoded_codec_frames = 0;
            int32_t err = 0;
            // Deliver `nframes` at `src`: end-trim against the program total, then write what the
            // output has room for. Returns the number of frames written; `spill` gets the within-
            // total remainder (room starvation), while beyond-total frames are discarded.
            auto deliver = [&](const int16_t* src, uint32_t nframes,
                               const int16_t** spill, uint32_t* spill_frames) -> uint32_t {
                *spill = nullptr; *spill_frames = 0;
                if (prog) {
                    const uint64_t left = I.total_samples > I.gapless_delivered
                                              ? I.total_samples - I.gapless_delivered : 0;
                    if (nframes > left) nframes = (uint32_t)left;
                }
                const uint32_t room_frames = (out_cap - produced) / frame_bytes;
                const uint32_t w = std::min(nframes, room_frames);
                if (w) {
                    if (!audio_store_bytes(job.out_addr + produced, src,
                                           w * frame_bytes)) { err = kAjm2ErrDecode; return 0; }
                    produced += w * frame_bytes;
                    if (prog) I.gapless_delivered += w;   // only meaningful for a gapless program
                }
                *spill = src + (size_t)w * ch;
                *spill_frames = nframes - w;
                return w;
            };
            // 1. Drain carry-over PCM (decoded earlier, not yet delivered) into this output first.
            //    Carry holds post-skip frames, so only the total/room trims apply here.
            if (!carry.empty() && produced < out_cap) {
                const uint32_t cframes = (uint32_t)(carry.size() / ch);
                const int16_t* spill = nullptr; uint32_t spill_frames = 0;
                deliver(carry.data(), cframes, &spill, &spill_frames);
                if (!err) {
                    if (spill_frames)
                        carry.erase(carry.begin(),
                                    carry.end() - (size_t)spill_frames * ch);
                    else
                        carry.clear();
                }
            }
            // 2. Decode this block's superframes into the remaining space. Stop BEFORE decoding once
            //    the output is full (so `iSizeConsumed` never covers PCM the guest did not receive)
            //    or the gapless program is complete (post-EOS input produces nothing).
            while (!err && produced < out_cap && in_cur + (uint32_t)sfb <= got &&
                   !(prog && I.gapless_delivered >= I.total_samples)) {
                if (dec->decode_superframe(in.data() + in_cur, pcm.data(),
                                           (int)(got - in_cur)) < 0) { err = kAjm2ErrDecode; break; }
                in_cur += (uint32_t)sfb;
                ++decoded_codec_frames;
                const int16_t* block = pcm.data();
                uint32_t nframes = (uint32_t)sfs;
                if (prog && I.skip_remaining) {     // priming skip: stream-order, decode-time drop
                    const uint32_t drop = std::min(I.skip_remaining, nframes);
                    I.skip_remaining -= drop;
                    block += (size_t)drop * ch;
                    nframes -= drop;
                }
                const int16_t* spill = nullptr; uint32_t spill_frames = 0;
                deliver(block, nframes, &spill, &spill_frames);
                if (err) break;
                if (spill_frames)                                       // spilled: carry forward
                    carry.insert(carry.end(), spill, spill + (size_t)spill_frames * ch);
            }
            I.decoded_samples += produced / frame_bytes;
            const bool result_published = ajm2_write_result(
                job.result_addr, err, in_cur, produced,
                prog ? I.gapless_delivered : I.decoded_samples, decoded_codec_frames);
            if (!result_published) {
                // Decoder/trim/carry state and guest PCM may already have advanced, but the guest
                // did not receive the progress sideband. Keep cumulative totals for diagnostics
                // and terminal error sidebands, while preventing any retry from advancing further.
                I.at9_dec.reset();
                I.carry.clear();
                dec = nullptr;
            }
            if (log)
                fprintf(stderr, "[ajm2] t=%llums decode inst=%u in=%zu/%uB out=%u -> %u consumed, %u PCM, total=%llu carry=%zu%s\n",
                        (unsigned long long)ajm2_log_ms(), inst_id, got, (unsigned)in_total, job.out_size,
                        in_cur, produced, (unsigned long long)it->second.decoded_samples,
                        carry.size() * sizeof(int16_t),
                        !result_published ? " RESULT_STORE_ERR" : (err ? " ERR" : ""));
        }
        ji = je;
    }
}
} // namespace

HLE10(ajm_batch_job_initialize) {
    // Read the 4-byte ATRAC9 config and (re)create this instance's decoder on a config change. Same
    // config -> keep the existing decoder so streaming MDCT overlap is preserved across blocks/batches.
    uint8_t cfg[4] = {0};
    const bool have = a2 && audio_read_bytes(a2, cfg, 4) && cfg[0] == 0xFE;
    std::lock_guard<std::mutex> lk(g_ajm2_mx);
    AjmDecodeInst& inst = g_ajm2_inst[(uint32_t)a1];
    const bool config_changed =
        have && (!inst.have_config || std::memcmp(inst.config, cfg, 4) != 0);
    if (getenv("PROSPER_AUDIOLOG")) {   // #1097: every re-init, with the state it inherits
        static std::atomic<uint64_t> n{0};
        const uint64_t k = n.fetch_add(1);
        if (k < 400 || (k % 100) == 0)
            fprintf(stderr, "[ajm2] t=%llums JobInitialize inst=%llu config=%02x%02x%02x%02x "
                    "changed=%d inherited_decoded=%llu\n",
                    (unsigned long long)ajm2_log_ms(), (unsigned long long)a1,
                    cfg[0], cfg[1], cfg[2], cfg[3], config_changed ? 1 : 0,
                    (unsigned long long)inst.decoded_samples);
    }
    if (config_changed) {
        std::memcpy(inst.config, cfg, 4);
        inst.have_config = true;
        inst.codec = (uint32_t)ajm::Codec::Atrac9;
        inst.at9_dec = std::make_unique<Atrac9Decoder>();
        if (!inst.at9_dec->init(cfg)) inst.at9_dec.reset();
        // A different stream format on a reused slot: drop the old stream's pending PCM and
        // program state; the guest programs a fresh gapless spec for the new sound (#1097).
        inst.carry.clear();
        inst.skip_remaining = 0;
        inst.gapless_delivered = 0;
        inst.total_samples = 0;
        if (getenv("PROSPER_AUDIOLOG"))
            fprintf(stderr, "[ajm2] JobInitialize inst=%llu config=%02x%02x%02x%02x decoder=%s\n",
                    (unsigned long long)a1, cfg[0], cfg[1], cfg[2], cfg[3],
                    inst.at9_dec ? "ok" : "init-failed");
    }
    return 0;
}
HLE10(ajm_batch_job_gapless) {
    // gaplessPtr = { u32 totalSamples, u32 skipSamples }. Recorded for future front/end trim; not yet
    // applied (skip is ~256 encoder-priming samples, inaudible for playback bring-up).
    uint32_t g[2] = {0, 0};
    if (a2) audio_read_bytes(a2, g, sizeof g);
    std::lock_guard<std::mutex> lk(g_ajm2_mx);
    AjmDecodeInst& inst = g_ajm2_inst[(uint32_t)a1];
    inst.total_samples = g[0];
    inst.skip_samples = g[1];
    // A gapless spec (re)starts the decode PROGRAM (#1097): the skip applies from here, the
    // delivered counter restarts (FMOD re-arms at a loop boundary and expects the next pass to
    // count 0..total again), and pending spill PCM belongs to the previous program -- a reused
    // codec slot must never emit the prior sound's tail into the new one.
    inst.skip_remaining = g[1];
    inst.gapless_delivered = 0;
    inst.carry.clear();
    if (getenv("PROSPER_AUDIOLOG")) {   // #1097 lifecycle evidence
        static std::atomic<uint64_t> n{0};
        const uint64_t k = n.fetch_add(1);
        if (k < 400 || (k % 100) == 0)
            fprintf(stderr, "[ajm2] t=%llums SetGapless inst=%llu total=%u skip=%u\n",
                    (unsigned long long)ajm2_log_ms(), (unsigned long long)a1, g[0], g[1]);
    }
    return 0;
}
HLE10(ajm_batch_job_decode) {
    // Queue a decode op on this batch; executed in order at BatchStart. a2/a3 = in/inSize,
    // a4/a5 = out/outSize, a6 = result sideband.
    if (getenv("PROSPER_AUDIOLOG")) { static std::atomic<uint32_t> n{0}; if (n.fetch_add(1) < 16)
        fprintf(stderr, "[ajm2] JobDecode batch=0x%llx inst=%llu in=0x%llx/%llu "
                        "out=0x%llx/%llu result=0x%llx ra=0x%llx tail=0x%llx,0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1,
                (unsigned long long)a2, (unsigned long long)a3,
                (unsigned long long)a4, (unsigned long long)a5,
                (unsigned long long)a6, (unsigned long long)a7,
                (unsigned long long)a8, (unsigned long long)a9); }
    if (!a0 || !a2 || !a4) return AJM_ERR_INVALID_PARAMETER;
    // Bound the guest-supplied sizes with the same cap the older AJM builders use. Unbounded values
    // would request a multi-gigabyte zero-filled staging buffer per job (a bad_alloc thrown through a
    // guest frame), and are never legitimate for a decode block.
    if (a3 > AJM_MAX_BUILDER_BYTES || a5 > AJM_MAX_BUILDER_BYTES) return AJM_ERR_INVALID_PARAMETER;
    std::lock_guard<std::mutex> lk(g_ajm2_mx);
    g_ajm2_jobs[a0].push_back({(uint32_t)a1, a4, a6, (uint32_t)a5, 0, 0,
                               {a2, 0, 0, 0}, {(uint32_t)a3, 0, 0, 0}, 1});
    return 0;
}
HLE10(ajm_batch_job_decode_split) {
    // sceAjmBatchJobDecodeSplit (#2981) — GRIS/AkSoundEngine ABI, proven by live-callsite
    // disassembly (the guest builds the call at eboot AkSoundEngine+... before `call stub`):
    //   a0 = SceAjmBatchInfo*            a1 = SceAjmInstance
    //   a2 = AjmBuffer* pIn  ({u64 ptr, u64 size} entries, built on the caller's stack)
    //   a3 = numIn (1..4; the callsite loop caps at 4)
    //   a4 = AjmBuffer* pOut ({ptr,size} entries)   a5 = numOut (1 or 2)
    //   a6 = sideband result (instance_obj+0x78)    a7/a8 = sideband bookkeeping (0x708 cap)
    //   a9 = return address
    // The earlier mapping treated a2/a3 as a raw (ptr,size) input pair and read dispatcher
    // stack residue as a 24 GB "outSize", silently rejecting EVERY job — the direct cause of
    // total Wwise silence in this title. The real payload is whole media chunks (e.g. a
    // 1472-byte ring-buffer block holding many framed Opus packets), not 3-byte fragments.
    if (!a0 || !a2 || !a4 || !a6 || !a3 || !a5 || a3 > 4 || a5 > 2)
        return AJM_ERR_INVALID_PARAMETER;
    AjmDecJob job;
    job.instance = (uint32_t)a1;
    job.result_addr = a6;
    // Output: first buffer's {ptr, size}; cap the staging like every other builder path.
    uint64_t out_desc[2 * 2] = {0, 0, 0, 0};
    if (audio_read_bytes_partial(a4, out_desc, sizeof(uint64_t) * 2 * a5) !=
        sizeof(uint64_t) * 2 * a5)
        return AJM_ERR_INVALID_PARAMETER;
    job.out_addr = out_desc[0];
    job.out_size = (uint32_t)std::min<uint64_t>(out_desc[1], AJM_MAX_BUILDER_BYTES);
    if (!job.out_addr || !job.out_size) return AJM_ERR_INVALID_PARAMETER;
    if (a5 > 1) {   // wrapped ring: second half
        job.out2_addr = out_desc[2];
        job.out2_size = (uint32_t)std::min<uint64_t>(out_desc[3], AJM_MAX_BUILDER_BYTES);
        if (!job.out2_addr || !job.out2_size) return AJM_ERR_INVALID_PARAMETER;
    }
    // Inputs: concatenate every {ptr,size} entry into one contiguous stream.
    uint64_t in_desc[2 * 4] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (audio_read_bytes_partial(a2, in_desc, sizeof(uint64_t) * 2 * a3) !=
        sizeof(uint64_t) * 2 * a3)
        return AJM_ERR_INVALID_PARAMETER;
    job.num_in = (uint32_t)a3;
    for (uint32_t f = 0; f < job.num_in; ++f) {
        job.in_addr[f] = in_desc[2 * f];
        const uint64_t sz = in_desc[2 * f + 1];
        if (!job.in_addr[f] || sz > AJM_MAX_BUILDER_BYTES) return AJM_ERR_INVALID_PARAMETER;
        job.in_size[f] = (uint32_t)sz;
    }
    if (getenv("PROSPER_AUDIOLOG")) { static std::atomic<uint32_t> n{0}; if (n.fetch_add(1) < 16)
        fprintf(stderr, "[ajm2] JobDecodeSplit batch=0x%llx inst=%llu in=%zxB(%u frag) "
                        "out=0x%llx/%u result=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1,
                (size_t)(job.in_size[0] + job.in_size[1] + job.in_size[2] + job.in_size[3]),
                job.num_in, (unsigned long long)job.out_addr, job.out_size,
                (unsigned long long)job.result_addr); }
    std::lock_guard<std::mutex> lk(g_ajm2_mx);
    g_ajm2_jobs[a0].push_back(std::move(job));
    return 0;
}
HLE10(ajm_batch_job_clear_context) {
    // Wwise calls this on restart/seek. A no-op left the host decoder holding its partial-packet
    // accumulator and libopus prediction state, so the next fragment spliced onto stale bytes and
    // the 2-byte prefix was read from the middle of an old packet (#2981 review 6b). Drop the
    // decoder: the executor lazily recreates it from the registered codec on the next batch, and
    // the per-instance spill carry dies with the reset.
    const uint32_t inst = (uint32_t)a1;
    std::lock_guard<std::mutex> lk(g_ajm2_mx);
    if (auto it = g_ajm2_inst.find(inst); it != g_ajm2_inst.end()) {
        it->second.host_dec.reset();
        it->second.carry.clear();
    }
    return 0;
}
HLE10(ajm_batch_job_set_resample_ex) { return 0; } // resample params: the host backend resamples
// GetResampleInfo/GetStatistics answer SCE_OK without writing their out-parameters. GRIS runs with
// this and its audio verifies on this head (36,765 decode jobs, 0 errors, port-17 music 124.9 s of
// 124.9 s audible), so the values are ignored or benign in practice — but a title that
// READS them gets untouched stack (the #2951 shape). Zero-filling needs the out-argument layout,
// which no live traffic has pinned yet.
// CONFIDENCE: LOW that returning success-without-writes is safe beyond GRIS.
HLE10(ajm_batch_job_get_resample_info) { return 0; }
HLE10(ajm_batch_job_get_statistics) { return 0; }
HLE10(ajm_batch_start2) {
    // Run every decode job queued on this batchInfo (a1), in order, then clear it. Synchronous:
    // BatchWait then just returns success. GTA V passes its u32 batch-id output in a4 and immediately
    // waits on it; leaving the initialized sentinel (0xffffffff) there breaks the lifecycle even when
    // decode itself succeeded. a3 is the optional batch-error object, unused on a successful start.
    if (getenv("PROSPER_AUDIOLOG")) { static std::atomic<uint32_t> n{0}; if (n.fetch_add(1) < 16)
        fprintf(stderr, "[ajm2] BatchStart ctx=%llu batch=0x%llx priority=%llu "
                        "a3=0x%llx a4=0x%llx a5=0x%llx a6=0x%llx a7=0x%llx a8=0x%llx a9=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5,
                (unsigned long long)a6, (unsigned long long)a7, (unsigned long long)a8,
                (unsigned long long)a9); }
    std::lock_guard<std::mutex> lk(g_ajm2_mx);
    // The batch id is the caller-visible proof that BatchStart accepted this submission. Validate
    // and publish it before any persistent decoder, PCM output, or result sideband advances. On an
    // inaccessible non-null pointer, leave the queued jobs intact so the guest can retry safely.
    if (a4 && !a2_store_u32(a4, g_ajm_next.fetch_add(1))) return AJM_ERR_INVALID_PARAMETER;
    auto it = g_ajm2_jobs.find(a1);
    if (it != g_ajm2_jobs.end()) {
        ajm2_decode_batch(it->second);
        g_ajm2_jobs.erase(it);      // erase, not clear: the map is keyed by a guest pointer
    }
    return 0;
}

// --- libSceNgs2 silent lifecycle ---------------------------------------------------------------
// Dead Cells is the first title to exercise NGS2. PROSPER_NGS2_TRACE preserves and logs all six
// guest arguments for this PS5 surface; normal execution uses the same handlers without logging.
namespace {

int ngs2_trace_level() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("PROSPER_NGS2_TRACE"); v = (e && *e) ? atoi(e) : 0; if (v < 0) v = 0; }
    return v;
}
bool ngs2_trace_enabled() { return ngs2_trace_level() >= 1; }

void ngs2_trace_call(const char* name, std::atomic<uint64_t>& calls,
                     uint64_t a0, uint64_t a1, uint64_t a2,
                     uint64_t a3, uint64_t a4, uint64_t a5) {
    const uint64_t n = calls.fetch_add(1) + 1;
    if (ngs2_trace_enabled() && (n <= 8 || (n & (n - 1)) == 0)) {
        fprintf(stderr, "[ngs2] %s #%llu (0x%llx, 0x%llx, 0x%llx, 0x%llx, 0x%llx, 0x%llx)\n",
                name, (unsigned long long)n,
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
    }
}

constexpr uint64_t kNgs2SystemTag = 0x4e47533253000000ull; // "NGS2S"
constexpr uint64_t kNgs2RackTag   = 0x4e47533252000000ull; // "NGS2R"
constexpr uint64_t kNgs2VoiceTag  = 0x4e47533256000000ull; // "NGS2V"
constexpr uint64_t kNgs2TagMask   = 0xffffffffffffff00ull;
constexpr uint64_t kNgs2VoiceMask = 0xffffffffff000000ull;
constexpr uint64_t kNgs2ErrInvalidOut    = (uint64_t)(int64_t)(int32_t)0x804a0053;
constexpr uint64_t kNgs2ErrInvalidSystem = (uint64_t)(int64_t)(int32_t)0x804a0230;
constexpr uint64_t kNgs2ErrInvalidRack   = (uint64_t)(int64_t)(int32_t)0x804a0261;
constexpr uint64_t kNgs2ErrInvalidVoice  = (uint64_t)(int64_t)(int32_t)0x804a0302;

struct Ngs2RackState {
    bool used = false;
    uint64_t system = 0;
    uint32_t rack_id = 0;
    uint32_t max_voices = 64;
};

std::mutex g_ngs2_mx;
bool g_ngs2_systems[4]{};
// NGS2 produces num_grain_samples frames per SystemRender; the guest hands us a buffer that can be far
// larger (Dead Cells passes 37888 frames) and only consumes one grain, streaming one grain-sized block
// per render. We fill only this many frames of an oversized buffer — filling the whole capacity makes a
// streaming voice's read cursor race ~9x ahead of the ~8-block queue the guest keeps topped up, i.e. the
// rhythmic underrun. Real NGS2 grains are small (SDK default 256, max 512), so this 4096 default is a
// safe upper bound that only trims oversized buffers. sceNgs2SystemCreate reads the real value from the
// SceNgs2SystemOption (num_grain_samples @ +0x70) when the title supplies one; Dead Cells passes null.
uint32_t g_ngs2_grain = 4096;
Ngs2RackState g_ngs2_racks[32];
std::mutex g_ngs2_zero_mx;
std::vector<uint8_t> g_ngs2_zeros;

// --- NGS2 sampler PCM mixer (Dead Cells) -------------------------------------------------------
// Dead Cells decodes its audio to interleaved 16-bit PCM in-engine (Heaps/Haxe) and streams it into
// NGS2 sampler voices; NGS2 mixes those voices in SystemRender and the guest hands the result to
// sceAudioOut. The previous headless backend zero-filled SystemRender, so the game played silence.
// This mixes the real voices. Protocol observed live (PROSPER_NGS2_TRACE=2), voice param IDs:
//   0x40010000  waveform format   { u32 waveform_type@0, u32 channels@4, u32 sample_rate@8 }
//   0x40010001  waveform block    { u64 data_ptr@0, ... }   (interleaved PCM at data_ptr)
//   0x40001300  matrix/gain       (per-render float matrix; unity for v1)
// waveform_type 0x12 == signed-16 LE PCM (the only type Dead Cells uses); ATRAC9 is deferred until a
// title that needs it exists. CONFIDENCE: MED — format/data/render layouts are live-verified; block
// length and one-shot/loop semantics are played from the current pointer and re-based when the guest
// re-points (fault-safe reads bound every access), pending an ngs2.h WaveformBlock cross-check.
constexpr uint32_t kNgs2ParamWaveformFormat = 0x40010000;
constexpr uint32_t kNgs2ParamWaveformBlock  = 0x40010001;
constexpr uint32_t kNgs2ParamMatrixLevels   = 0x40001300;   // per-render output level matrix (gains)
constexpr uint32_t kNgs2ParamVoiceCallback  = 0x00000007;   // Ngs2VoiceCallbackParam (block-done handler)
constexpr uint32_t kNgs2WaveformTypePcmS16  = 0x12;
// A voice stops when it has been mixed this many consecutive renders without a fresh waveform block.
// Sampler SFX are re-fed continuously while audible (observed: a play is followed by a stream of block
// params); one that stops being fed has ended, so expiring it bounds voice accumulation (voice-pool
// handles otherwise read a persistent/reused buffer forever and pile up into clipping over time).
constexpr int kNgs2VoiceIdleRendersMax = 3;

struct Ngs2Voice {
    bool     playing      = false;
    bool     played_audio = false; // has produced any non-silent sample since the last (re)trigger
    int      idle         = 0;     // renders mixed since the last waveform-block feed
    uint64_t data_ptr  = 0;        // guest base of the current interleaved-PCM block
    uint32_t channels  = 2;        // source channel count
    uint32_t rate      = 48000;    // source sample rate (Hz)
    // Per-output-channel gain from the voice's level matrix (kNgs2ParamMatrixLevels). Sized for the
    // full PS5 surround layout (up to 7.1); prosper currently mixes only the front L/R pair (see
    // ngs2_mix_voices), so gain[2..7] are parsed and stored but not yet summed into a surround bus.
    // SURROUND-TODO: when a multichannel output bus is added, mix all channels through these gains.
    float    gain[8]   = {1,1,1,1,1,1,1,1};
    uint32_t wtype     = kNgs2WaveformTypePcmS16;
    double   cursor    = 0.0;      // fractional source-frame position within blocks.front()
    // Fed waveform blocks awaiting playback, in feed order (front = currently playing). A streaming
    // guest feeds a fixed-size block per grain and reuses a small ring of non-contiguous pool buffers,
    // so the mixer must play them one block at a time and never read past a block boundary (doing so
    // samples the next, unrelated pool buffer -> the garbled "digital noise").
    std::deque<uint64_t> blocks;
    // Waveform-block completion callback (voice param 0x0007, Ngs2VoiceCallbackParam). A streaming guest
    // registers it and expects NGS2 to invoke it as each fed block finishes, so it can queue the next.
    uint64_t cb_fn = 0, cb_data = 0;   // guest handler function pointer + user data
    uint32_t cb_flags = 0;
    uint64_t played_samples = 0;       // wall-clock-paced total source frames played; reported as
                                       // VoiceGetState's num_decoded_samples so the guest's decode/feed
                                       // loop paces to real time instead of to our (faster) render rate.
    uint64_t t0_ns          = 0;       // steady-clock ns at stream start (0 = not started / resync).
};
std::map<uint32_t, Ngs2Voice> g_ngs2_voices;   // key: voice handle & 0xffffff (rack_slot<<8 | voice_id)

bool ngs2_mix_disabled() {
    static const bool off = [] { const char* e = getenv("PROSPER_NGS2_SILENT"); return e && *e && *e != '0'; }();
    return off;
}

// Parse a voice-param chain and fold format/data updates into the voice state. `handle` is the full
// guest voice handle; the low 24 bits key g_ngs2_voices. Runs under g_ngs2_mx.
void ngs2_apply_voice_params(uint64_t handle, uint64_t list) {
    if ((handle & kNgs2VoiceMask) != kNgs2VoiceTag || !list) return;
    Ngs2Voice& v = g_ngs2_voices[(uint32_t)(handle & 0xffffff)];
    uint64_t p = list;
    for (int i = 0; i < 32 && p; ++i) {
        struct { uint16_t size; int16_t next; uint32_t id; } head{};
        if (!audio_read_bytes(p, &head, sizeof(head)) || head.size < sizeof(head) || head.size > 0x400) break;
        if (head.id == kNgs2ParamWaveformFormat) {
            uint32_t fmt[3] = {};   // { waveform_type, channels, sample_rate }
            if (audio_read_bytes(p + 8, fmt, sizeof(fmt))) {
                // A waveform-format param starts a NEW sound on this voice: the guest reuses a small pool
                // of voices, reformatting one each time it (re)triggers a click/SFX, and streams a fresh
                // track's blocks after re-formatting the music voice. Reset the stream state so the new
                // sound plays from the start instead of queueing behind the previous sound's leftover
                // blocks and stale clock (that pile-up is the "buffer mess" on the second click).
                v.blocks.clear();
                v.cursor = 0.0;
                v.played_samples = 0;
                v.t0_ns = 0;
                v.wtype = fmt[0];
                if (fmt[1] >= 1 && fmt[1] <= 8)          v.channels = fmt[1];
                if (fmt[2] >= 8000 && fmt[2] <= 192000)  v.rate     = fmt[2];
                if (getenv("PROSPER_NGS2_TRACE"))
                    fprintf(stderr, "[ngs2] FORMAT voice=0x%llx waveform_type=0x%x ch=%u rate=%u\n",
                            (unsigned long long)handle, fmt[0], fmt[1], fmt[2]);
            }
        } else if (head.id == kNgs2ParamMatrixLevels) {
            // Per-render output level matrix. Observed layout (VoiceControl capture): a leading count
            // word (+0x10) and a ramp/config float (+0x14, ~1e4), then the front L/R output levels as
            // two floats at +0x18 and +0x1c (e.g. 0.707, 1.0). Apply them as this voice's front L/R
            // gains so the game's own mix balance holds (this is what keeps the summed bus in range —
            // prosper was mixing every voice at unity before). Values outside a sane [0,4] range are
            // ignored (the matrix also carries non-gain fields like ramp lengths).
            // SURROUND-TODO: the full matrix maps source channels to ALL output ports (up to 7.1); we
            // read only the front pair. Parse the rest into gain[2..7] once a surround bus exists.
            float lv[2] = {};
            if (audio_read_bytes(p + 0x18, lv, sizeof(lv))) {
                if (lv[0] >= 0.0f && lv[0] <= 4.0f) v.gain[0] = lv[0];
                if (lv[1] >= 0.0f && lv[1] <= 4.0f) v.gain[1] = lv[1];
            }
        } else if (head.id == kNgs2ParamWaveformBlock) {
            uint64_t ptr = 0;
            if (audio_read_bytes(p + 8, &ptr, 8) && ptr >= 0x200000000ull) {
                // Enqueue the block for in-order playback (bounded so a runaway feed can't grow it
                // without limit). The mixer plays through the queue one block at a time.
                if (v.blocks.size() < 64) v.blocks.push_back(ptr);
                v.data_ptr     = ptr;  // last fed block (FEED trace / diagnostics)
                v.playing      = true; // streaming
                v.idle         = 0;    // fed this render -> not idle
                if (getenv("PROSPER_NGS2_TRACE")) {
                    int16_t probe[64] = {}; size_t g = audio_read_bytes_partial(ptr, probe, sizeof probe);
                    int pk = 0; for (size_t j = 0; j < g / 2; j++) { int a = probe[j] < 0 ? -probe[j] : probe[j]; if (a > pk) pk = a; }
                    fprintf(stderr, "[ngs2] FEED voice=0x%llx ptr=0x%llx got=%zu probe_peak=%d ch=%u rate=%u\n",
                            (unsigned long long)handle, (unsigned long long)ptr, g, pk, v.channels, v.rate);
                }
            }
        } else if (head.id == kNgs2ParamVoiceCallback) {
            // Ngs2VoiceCallbackParam: header(8), callback fn(8), callback_data(8), flags(4), reserved(4).
            // The guest registers a per-voice waveform-block completion handler here; NGS2 invokes it as
            // each fed block finishes so the guest queues the next (see ngs2_fire_pending_callbacks).
            uint64_t fn = 0, data = 0; uint32_t fl = 0;
            audio_read_bytes(p + 8,  &fn,   8);
            audio_read_bytes(p + 16, &data, 8);
            audio_read_bytes(p + 24, &fl,   4);
            v.cb_fn = fn; v.cb_data = data; v.cb_flags = fl;
            if (getenv("PROSPER_NGS2_TRACE"))
                fprintf(stderr, "[ngs2] CALLBACK voice=0x%llx fn=0x%llx data=0x%llx flags=0x%x\n",
                        (unsigned long long)handle, (unsigned long long)fn, (unsigned long long)data, fl);
        }
        if (head.next == 0) break;
        p += (uint64_t)head.next;
    }
}

// --- NGS2 waveform-block completion callbacks --------------------------------------------------
// A streaming guest (Dead Cells) registers a per-voice callback (voice param 0x0007) and expects
// NGS2 to invoke it as each fed waveform block finishes, so it can queue the next block. Without it
// the stream stalls after the initial ring fill and no music is ever produced. The registered handler
// is GUEST code; recovered from Dead Cells' handler at eboot+0x1740f50 it reads a
// SceNgs2VoiceCallbackInfo*:
//     +0x00 : callbackData  (the game's per-stream context pointer we were handed at registration)
//     +0x10 : event flags   (bit0 = "waveform block consumed"; the handler no-ops if it is clear)
//     +0x18 : a pointer whose byte +0x48 must be non-zero for the handler to run its state update
// The handler then re-enters sceNgs2VoiceGetState and reads num_decoded_samples (state+0x10) to
// advance its stream clock. So the callback must run (a) with the caller's guest %fs restored (its
// nested imports resolve through the guest TCB) and (b) OUTSIDE g_ngs2_mx (it re-enters our HLE).
namespace {
struct Ngs2PendingCallback { uint64_t fn, data; uint32_t flags; };
// Populated (under g_ngs2_mx) while mixing, drained after the lock drops on the same SystemRender
// (guest) thread. Thread-local: each guest audio thread renders and fires its own system's voices.
thread_local std::vector<Ngs2PendingCallback> t_ngs2_pending_cb;
thread_local uint64_t t_ngs2_render_guest_fs = 0;  // caller's guest %fs, captured at SystemRender entry

// Firing a guest callback needs the FSGSBASE swap and the SystemRender entry trampoline, both of which
// only exist on the Linux guest-execution path (the trampoline is registered under the same guard). On
// other platforms the callbacks are simply not fired (music streaming is a Linux capability for now), so
// nothing here — including the queuing in ngs2_mix_voices — is compiled elsewhere.
#if defined(__linux__)
inline uint64_t ngs2_rd_fsbase() { uint64_t v; __asm__ volatile("rdfsbase %0" : "=r"(v)); return v; }
inline void     ngs2_wr_fsbase(uint64_t v) { __asm__ volatile("wrfsbase %0" : : "r"(v)); }

void ngs2_fire_pending_callbacks() {
    if (t_ngs2_pending_cb.empty()) return;
    std::vector<Ngs2PendingCallback> pend;
    pend.swap(t_ngs2_pending_cb);
    const uint64_t guest_fs = t_ngs2_render_guest_fs;
    uint64_t host_fs = 0; bool swapped = false;
    if (guest_fs) { host_fs = ngs2_rd_fsbase(); ngs2_wr_fsbase(guest_fs); swapped = true; }
    for (const auto& cb : pend) {
        if (!cb.fn) continue;
        // Minimal, valid SceNgs2VoiceCallbackInfo on the host stack (guest code reads it directly; the
        // guest runs in-process so a host address is a valid guest pointer). obj carries the +0x48
        // "active" byte the handler checks via eboot+0x173f490.
        alignas(16) uint8_t obj[0x50] = {}; obj[0x48] = 1;
        alignas(16) uint8_t info[0x40] = {};
        *(uint64_t*)(info + 0x00) = cb.data;          // callbackData -> game stream context
        *(uint32_t*)(info + 0x10) = cb.flags | 1u;    // event flags, bit0 (block-consumed) set
        *(uint64_t*)(info + 0x18) = (uint64_t)(uintptr_t)obj;
        ((void (*)(uint64_t))(uintptr_t)cb.fn)((uint64_t)(uintptr_t)info);
    }
    if (swapped) ngs2_wr_fsbase(host_fs);
}
#endif
} // namespace

// Mix all playing voices of `system` into one render buffer of `frames` interleaved frames at
// `out_rate`/`out_channels`, accumulating float samples into `mix` (size frames*out_channels).
// Reads source PCM fault-safely in one bulk read per voice; a short/unmapped source just stops that
// voice. Linear resample from each voice's source rate; source stereo→out stereo direct, mono→dup.
void ngs2_mix_voices(uint64_t system, float* mix, uint32_t frames, uint32_t out_channels, uint32_t out_rate) {
    // De-duplicate the 16-voice sampler pool: Dead Cells cycles a fixed pool and reuses a small set of
    // shared PCM buffers, so after the pool wraps, several stale voice handles still reference the SAME
    // buffer. NGS2 stops those on hardware; prosper lacks the (undocumented) one-shot-end/stop state, so
    // without this the same click renders on top of itself a few ms apart (audible doubling once >16
    // clicks have fired). Render each distinct buffer only once, choosing the FRESHEST voice (smallest
    // cursor = most recently (re)triggered); older duplicates at that buffer are skipped this pass.
    // CONFIDENCE: MED — a pragmatic proxy for real voice lifecycle; identical-buffer voices are truly
    // redundant into a mono/stereo bus, so this cannot drop a distinct sound.
    // A streamed waveform block is one grain of frames (the guest feeds one block per grain), so the
    // retire boundary and the per-block read length both track num_grain_samples rather than a constant.
    // For Dead Cells (grain 4096) this is the validated value; a title with a different grain retires and
    // fires its completion callback at the correct rate instead of ~grain/4096x off.
    const uint64_t kBlockFrames = g_ngs2_grain ? g_ngs2_grain : 4096;
    const uint64_t now_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    std::set<uint64_t> rendered_front;         // render each distinct front block once (pool reuse across voices)
    for (auto& [key, v] : g_ngs2_voices) {
        if (!v.playing) continue;
        if (v.blocks.empty()) {
            // Not fed this render. A callback (streaming) voice is re-fed by its block-completion callback,
            // so it may sit briefly between feeds; give it a much longer idle window than a one-shot before
            // expiring (a genuinely dead stream still stops, just later). HOLD the playback clock across the
            // gap by rebasing t0 so (now - t0)*rate == played_samples: the pause neither advances nor
            // rewinds the stream position, and playback resumes seamlessly. (Zeroing t0 here instead would,
            // on resume, make target=0 << played_samples and clamp advance to 0 for played_samples/rate
            // seconds — a multi-second freeze whenever the queue momentarily drains.)
            if (v.rate) v.t0_ns = now_ns - (uint64_t)((double)v.played_samples / (double)v.rate * 1e9);
            const int idle_max = v.cb_fn ? 240 : kNgs2VoiceIdleRendersMax;
            if (++v.idle > idle_max) v.playing = false;
            continue;
        }
        v.idle = 0;
        if (v.wtype != kNgs2WaveformTypePcmS16) {           // only S16 PCM implemented (ATRAC9 deferred)
            if (getenv("PROSPER_NGS2_TRACE")) {
                static uint32_t skipped = 0;
                if ((skipped++ & 0xff) == 0)
                    fprintf(stderr, "[ngs2] SKIP non-PCM voice=0x%x waveform_type=0x%x (music/codec?)\n", key, v.wtype);
            }
            continue;
        }
        if (!rendered_front.insert(v.blocks.front()).second) continue;  // duplicate front block this pass

        const double step = (double)v.rate / (double)(out_rate ? out_rate : 48000);
        const uint32_t src_ch = v.channels ? v.channels : 2;
        // Gather the source frames this render needs by reading whole blocks in queue order. Reading one
        // block at a time (never a single span across two pool buffers) is what stops the mixer sampling
        // an unrelated neighbouring block. cursor is the fractional read position within blocks.front().
        const uint64_t need_frames = (uint64_t)(v.cursor + (double)frames * step) + 2;
        std::vector<int16_t> src;
        src.reserve((size_t)std::min<uint64_t>(need_frames, 1u << 20) * src_ch);
        for (size_t bi = 0; bi < v.blocks.size() && src.size() < need_frames * src_ch; ++bi) {
            std::vector<int16_t> blk((size_t)kBlockFrames * src_ch, 0);
            const size_t got = audio_read_bytes_partial(v.blocks[bi], blk.data(), blk.size() * 2);
            const size_t bf  = got / ((size_t)src_ch * 2);   // valid frames in this block
            src.insert(src.end(), blk.begin(), blk.begin() + bf * src_ch);
            if (bf < kBlockFrames) break;                    // short/unmapped block -> stream truncated
        }
        const uint64_t have_frames = src.size() / src_ch;
        if (have_frames == 0) { v.blocks.clear(); v.playing = false; continue; }
        for (uint32_t f = 0; f < frames; ++f) {
            const double sp = v.cursor + (double)f * step;
            const uint64_t i0 = (uint64_t)sp;
            const double frac = sp - (double)i0;
            bool ran_out = false;
            for (uint32_t oc = 0; oc < out_channels; ++oc) {
                // Channel routing. SURROUND-LIMITATION: prosper targets a 2-channel (front L/R) bus,
                // so we map source channel oc→oc for stereo and fold mono to both. A true PS5 surround
                // (up to 7.1) output would route every source channel to its speaker via the full level
                // matrix; here only gain[0]/gain[1] (front L/R) are applied. Extend when a surround bus
                // and the full matrix parse (gain[2..7]) land.
                const uint32_t sc = (src_ch == 1) ? 0 : (oc < src_ch ? oc : src_ch - 1);
                const uint64_t a = (i0)     * src_ch + sc;
                const uint64_t b = (i0 + 1) * src_ch + sc;
                if (b >= have_frames * src_ch) { ran_out = true; break; }
                const double s = (1.0 - frac) * src[a] + frac * src[b];
                mix[f * out_channels + oc] += (float)(s / 32768.0) * (oc < 8 ? v.gain[oc] : 1.0f);
            }
            if (ran_out) break;   // consumed all valid source this render
        }
        // Advance the read cursor by WALL-CLOCK real time, not by frames*step. The guest over-calls
        // SystemRender (measured ~4x real time) while its decoder feeds blocks at real time; advancing by
        // frames*step every call would drain the queue ~4x faster than it is filled (the rhythmic under-
        // run). Pacing to real time makes consumption match the real-time feed, so the queue holds its
        // depth. The full frames*step window is still MIXED from the cursor each call — the guest double-
        // buffers and outputs one render per real-time grain — but the cursor only moves by elapsed time.
        if (v.t0_ns == 0) { v.t0_ns = now_ns; }
        const double target = (double)(now_ns - v.t0_ns) * 1e-9 * (double)v.rate;
        double advance = target - (double)v.played_samples;
        if (advance < 0.0) advance = 0.0;
        if (advance > (double)frames * step) advance = (double)frames * step;  // at most one grain/call
        v.cursor += advance;
        v.played_samples += (uint64_t)advance;
        // Retire fully-consumed blocks; each completed block fires the guest's block-completion callback
        // so it queues the next one. Fired after the lock drops (see ngs2_fire_pending_callbacks).
        while (v.cursor >= (double)kBlockFrames && !v.blocks.empty()) {
            v.blocks.pop_front();
            v.cursor -= (double)kBlockFrames;
#if defined(__linux__)
            if (v.cb_fn) t_ngs2_pending_cb.push_back({v.cb_fn, v.cb_data, v.cb_flags});
#endif
        }
    }
}

bool ngs2_read_u32(uint64_t src, uint32_t& value) {
    return audio_read_bytes(src, &value, sizeof value);
}

bool ngs2_read_bytes(uint64_t src, void* dst, size_t size) {
    return audio_read_bytes(src, dst, size);
}

bool ngs2_zero_bytes(uint64_t dst, size_t size) {
    // Do not use thread_local here: guest execution swaps %fs, and adding host TLS to prosper_core
    // perturbs Messenger before its first syscall. Process-global zero storage is sufficient because
    // NGS2 rendering is serialized by its audio thread; the lock also makes that contract explicit.
    std::lock_guard<std::mutex> lock(g_ngs2_zero_mx);
    if (g_ngs2_zeros.size() < size) g_ngs2_zeros.resize(size, 0);
    return audio_store_bytes(dst, g_ngs2_zeros.data(), size);
}

uint32_t ngs2_max_voices(uint64_t option) {
    uint32_t voices = 0;
    // SceNgs2RackOption: size[8], name[16], flags, maxGrainSamples, maxVoices.
    if (option) ngs2_read_u32(option + 0x20, voices);
    return voices ? voices : 64;
}

bool ngs2_valid_system(uint64_t handle) {
    const uint64_t slot = handle & 0xff;
    return (handle & kNgs2TagMask) == kNgs2SystemTag && slot >= 1 && slot <= 4 &&
           g_ngs2_systems[slot - 1];
}

Ngs2RackState* ngs2_rack(uint64_t handle) {
    const uint64_t slot = handle & 0xff;
    if ((handle & kNgs2TagMask) != kNgs2RackTag || slot < 1 || slot > 32) return nullptr;
    Ngs2RackState& rack = g_ngs2_racks[slot - 1];
    return rack.used ? &rack : nullptr;
}

} // namespace

#define NGS2_LOG(label) do { static std::atomic<uint64_t> calls{0}; \
    ngs2_trace_call(label, calls, a0, a1, a2, a3, a4, a5); } while (0)

// NGS2 is implemented as a silent backend: the guest still receives real buffer requirements and
// opaque lifecycle handles, while SystemRender produces silence. Layouts/prototypes agree between
// the 3.20 export table, Kyty, shadPS4, and the Dead Cells live trace above. CONFIDENCE: HIGH on
// argument/output positions and handle flow; MED on the deliberately private work-buffer sizes.
HLE(ngs2_system_query_buffer) {
    NGS2_LOG("sceNgs2SystemQueryBufferSize");
    if (!a1 || !a2_store_zeros(a1, 0x40) || !a2_store_u64(a1 + 8, 0x1000))
        return kNgs2ErrInvalidOut;
    return 0;
}

HLE(ngs2_system_create) {
    NGS2_LOG("sceNgs2SystemCreate");
    // a0 = const SceNgs2SystemOption* (optional). num_grain_samples is at +0x70 (size@0, name[64]@8,
    // job_scheduler_options[4]@0x48, flags@0x68, max_grain_samples@0x6c, num_grain_samples@0x70). Honour
    // it when present and sane, else keep the streaming-safe default. Dead Cells passes a0 == null.
    // CONFIDENCE: MED — offset from the published SceNgs2SystemOption layout; not yet seen non-null live.
    if (a0) { uint32_t g = 0; if (audio_read_bytes(a0 + 0x70, &g, 4) && g >= 64 && g <= 8192) g_ngs2_grain = g; }
    if (!a1 || !a2) return kNgs2ErrInvalidOut;
    std::lock_guard<std::mutex> lock(g_ngs2_mx);
    for (uint64_t i = 0; i < 4; ++i) {
        if (g_ngs2_systems[i]) continue;
        g_ngs2_systems[i] = true;
        if (!a2_store_u64(a2, kNgs2SystemTag | (i + 1))) {
            g_ngs2_systems[i] = false;
            return kNgs2ErrInvalidOut;
        }
        return 0;
    }
    return kNgs2ErrInvalidSystem;
}

HLE(ngs2_rack_query_buffer) {
    NGS2_LOG("sceNgs2RackQueryBufferSize");
    const uint64_t size = 0x1000ull + (uint64_t)ngs2_max_voices(a1) * 0x40ull;
    if (!a2 || !a2_store_zeros(a2, 0x40) || !a2_store_u64(a2 + 8, size))
        return kNgs2ErrInvalidOut;
    return 0;
}

HLE(ngs2_rack_create) {
    NGS2_LOG("sceNgs2RackCreate");
    if (!a3 || !a4) return kNgs2ErrInvalidOut;
    std::lock_guard<std::mutex> lock(g_ngs2_mx);
    if (!ngs2_valid_system(a0)) return kNgs2ErrInvalidSystem;
    for (uint64_t i = 0; i < 32; ++i) {
        if (g_ngs2_racks[i].used) continue;
        g_ngs2_racks[i] = {true, a0, (uint32_t)a1, ngs2_max_voices(a2)};
        if (!a2_store_u64(a4, kNgs2RackTag | (i + 1))) {
            g_ngs2_racks[i] = {};
            return kNgs2ErrInvalidOut;
        }
        return 0;
    }
    return kNgs2ErrInvalidRack;
}

HLE(ngs2_rack_get_voice) {
    NGS2_LOG("sceNgs2RackGetVoiceHandle");
    if (!a2) return kNgs2ErrInvalidOut;
    std::lock_guard<std::mutex> lock(g_ngs2_mx);
    Ngs2RackState* rack = ngs2_rack(a0);
    if (!rack) return kNgs2ErrInvalidRack;
    if (a1 >= rack->max_voices || a1 > 0xff) return kNgs2ErrInvalidVoice;
    const uint64_t rack_slot = a0 & 0xff;
    return a2_store_u64(a2, kNgs2VoiceTag | (rack_slot << 8) | a1) ? 0 : kNgs2ErrInvalidOut;
}

// PROSPER_NGS2_TRACE=2 additionally dumps the voice-command param chain: each entry is a
// Ngs2VoiceParamHead { uint16 size; int16 next; uint32 id; payload... } (Sony's documented
// voice-param list shape). This is capture-first RE for the real sampler implementation —
// the play/setup commands carry the waveform format (codec/rate/channels) and data pointers
// we need before mixing can be implemented faithfully.
void ngs2_dump_param_chain(const char* tag, uint64_t list) {
    if (ngs2_trace_level() < 2 || !list) return;
    uint64_t p = list;
    for (int i = 0; i < 16 && p; ++i) {
        struct { uint16_t size; int16_t next; uint32_t id; } head{};
        if (!ngs2_read_bytes(p, &head, sizeof(head))) break;
        fprintf(stderr, "[ngs2]   %s param[%d] @0x%llx size=0x%x next=%d id=0x%08x\n",
                tag, i, (unsigned long long)p, head.size, head.next, head.id);
        if (head.size == 0 || head.size > 0x400) break;
        a2_dump("payload", p, head.size < 0x60 ? head.size : 0x60);
        // Fingerprint any guest pointer the payload carries (waveform data address): dump its first
        // bytes so PCM (sample data) vs a codec container (RIFF/'RIFF', 'AT9 ', ATRAC9 config) is
        // decidable from the magic. Guest heap/direct pointers live in the 0x2_00000000+ aperture.
        for (uint16_t off = 8; off + 8 <= head.size; off += 8) {
            uint64_t val = 0;
            if (!ngs2_read_bytes(p + off, &val, 8)) break;
            if (val >= 0x200000000ull && val < 0x8000000000ull) {
                fprintf(stderr, "[ngs2]   %s param[%d] +0x%x -> guest ptr 0x%llx:\n",
                        tag, i, off, (unsigned long long)val);
                a2_dump("wavedata", val, 0x40);
            }
        }
        if (head.next == 0) break;
        p += (uint64_t)head.next;
    }
}

HLE(ngs2_voice_control) {
    NGS2_LOG("sceNgs2VoiceControl");
    ngs2_dump_param_chain("VoiceControl", a1);
    if ((a0 & kNgs2VoiceMask) != kNgs2VoiceTag) return kNgs2ErrInvalidVoice;
    if (!ngs2_mix_disabled()) { std::lock_guard<std::mutex> lock(g_ngs2_mx); ngs2_apply_voice_params(a0, a1); }
    return 0;
}

HLE(ngs2_voice_run_commands) {
    NGS2_LOG("sceNgs2VoiceRunCommands");
    ngs2_dump_param_chain("RunCommands", a1);
    if ((a0 & kNgs2VoiceMask) != kNgs2VoiceTag) return kNgs2ErrInvalidVoice;
    if (!ngs2_mix_disabled()) { std::lock_guard<std::mutex> lock(g_ngs2_mx); ngs2_apply_voice_params(a0, a1); }
    return 0;
}

HLE(ngs2_voice_get_state) {
    NGS2_LOG("sceNgs2VoiceGetState");
    if ((a0 & kNgs2VoiceMask) != kNgs2VoiceTag) return kNgs2ErrInvalidVoice;
    if (!a1 || !a2 || a2 > 0x1000 || !a2_store_zeros(a1, (size_t)a2)) return kNgs2ErrInvalidOut;
    // SceNgs2SamplerVoiceState (0x30): state_flags@0x00, envelope_height@0x04(f), peak_height@0x08(f),
    // reserved@0x0c, num_decoded_samples@0x10(u64), decoded_data_size@0x18(u64), user_data@0x20,
    // waveform_data@0x28. The block-completion callback re-enters here and reads num_decoded_samples to
    // advance its stream clock, so it must reflect real playback progress or the stream never advances.
    std::lock_guard<std::mutex> lock(g_ngs2_mx);
    auto it = g_ngs2_voices.find((uint32_t)(a0 & 0xffffff));
    const bool playing = it != g_ngs2_voices.end() && it->second.playing && it->second.data_ptr;
    a2_store_u32(a1, playing ? 0x3u : 0u);          // state_flags: Playing / Empty
    if (a2 >= 0x30) {
        a2_store_u32(a1 + 0x04, 0x3f800000u);       // envelope_height = 1.0f
        if (it != g_ngs2_voices.end()) {
            const uint64_t decoded = it->second.played_samples;
            const uint32_t ch = it->second.channels ? it->second.channels : 2;
            a2_store_u64(a1 + 0x10, decoded);        // num_decoded_samples (per-channel frames)
            a2_store_u64(a1 + 0x18, decoded * ch * 2); // decoded_data_size (bytes, s16)
        }
    }
    return 0;
}

// State-flag bits the guest tests (Kyty Ngs2GetStateFlags): Empty=0, Playing=0x3, Paused=0x5,
// Stopped=0xb. Report Playing for a voice we are actively mixing so the game's audio state machine
// treats the voice as live; else Empty.
uint32_t ngs2_voice_state_flags(uint64_t handle) {
    if ((handle & kNgs2VoiceMask) != kNgs2VoiceTag) return 0;
    auto it = g_ngs2_voices.find((uint32_t)(handle & 0xffffff));
    return (it != g_ngs2_voices.end() && it->second.playing && it->second.data_ptr) ? 0x3u : 0u;
}

// sceNgs2VoiceGetStateFlags(voice, uint32_t* outFlags) -> 0. Was UNREGISTERED (#TBD): the generic
// stub returned 0 and left *outFlags uninitialized, so the guest read garbage voice state and its
// audio scheduler misbehaved (the uninit-out class). Return real Playing/Empty flags.
HLE(ngs2_voice_get_state_flags) {
    NGS2_LOG("sceNgs2VoiceGetStateFlags");
    if ((a0 & kNgs2VoiceMask) != kNgs2VoiceTag) return kNgs2ErrInvalidVoice;
    if (!a1) return kNgs2ErrInvalidOut;
    uint32_t flags; { std::lock_guard<std::mutex> lock(g_ngs2_mx); flags = ngs2_voice_state_flags(a0); }
    return a2_store_u32(a1, flags) ? 0 : kNgs2ErrInvalidOut;
}

// sceNgs2RackDestroy(rack, Ngs2ContextBufferInfo* out) -> 0. Was UNREGISTERED. Release the rack and
// drop its voices so their handles do not leak into later mixes; the out context-buffer info, if
// present, is zeroed (we own no returned host buffer).
HLE(ngs2_rack_destroy) {
    NGS2_LOG("sceNgs2RackDestroy");
    std::lock_guard<std::mutex> lock(g_ngs2_mx);
    const uint64_t slot = a0 & 0xff;
    if ((a0 & kNgs2TagMask) != kNgs2RackTag || slot < 1 || slot > 32) return kNgs2ErrInvalidRack;
    g_ngs2_racks[slot - 1] = {};
    for (auto it = g_ngs2_voices.begin(); it != g_ngs2_voices.end(); )
        it = ((it->first >> 8) == slot) ? g_ngs2_voices.erase(it) : std::next(it);
    if (a1 && !a2_store_zeros(a1, 0x40)) return kNgs2ErrInvalidOut;
    return 0;
}

// sceNgs2GeomApply(...) -> 0. Was UNREGISTERED (NID eF8yRCC6W64, seen unimplemented in boot logs):
// applies 3D listener/source geometry to a voice's pan matrix. prosper mixes without 3D panning, so
// this is a faithful no-op success (the guest keeps its pre-initialized pan) — never leave it to the
// generic stub, which the guest cannot distinguish from a real error path.
HLE(ngs2_geom_apply) {
    NGS2_LOG("sceNgs2GeomApply");
    return 0;
}

HLE(ngs2_geom_reset_source) {
    NGS2_LOG("sceNgs2GeomResetSourceParam");
    return a0 && a2_store_zeros(a0, 0xa8) ? 0 : kNgs2ErrInvalidOut;
}

HLE(ngs2_system_render) {
    NGS2_LOG("sceNgs2SystemRender");
    {
        std::lock_guard<std::mutex> lock(g_ngs2_mx);
        if (!ngs2_valid_system(a0)) return kNgs2ErrInvalidSystem;
    }
    struct RenderBufferInfo { uint64_t buffer, size; uint32_t waveform_type, channels; };
    if (!a1 || a2 == 0 || a2 > 16) return kNgs2ErrInvalidOut;
    for (uint64_t i = 0; i < a2; ++i) {
        RenderBufferInfo info{};
        if (!ngs2_read_bytes(a1 + i * sizeof(info), &info, sizeof(info)) ||
            !info.buffer || info.size > 64ull * 1024 * 1024)
            return kNgs2ErrInvalidOut;

        // The render buffer is the mix destination. Output format: waveform_type 0x12 == S16, anything
        // else (the NGS2 render default) == F32. Frame count derives from size and the output stride.
        const bool     out_s16      = info.waveform_type == kNgs2WaveformTypePcmS16;
        const uint32_t out_channels = (info.channels >= 1 && info.channels <= 8) ? info.channels : 2;
        const uint32_t out_bps      = out_s16 ? 2 : 4;
        uint32_t       frames       = (uint32_t)(info.size / ((uint64_t)out_channels * out_bps));
        // Fill only num_grain_samples frames of an oversized buffer (see g_ngs2_grain); the rest stays
        // zeroed. Filling the whole capacity makes a streaming voice's read cursor race ahead of the
        // guest's block queue and produce the rhythmic underrun.
        if (g_ngs2_grain && frames > g_ngs2_grain) frames = g_ngs2_grain;

        if (ngs2_mix_disabled() || frames == 0) {
            if (!ngs2_zero_bytes(info.buffer, (size_t)info.size)) return kNgs2ErrInvalidOut;
            continue;
        }

        std::vector<float> mix((size_t)frames * out_channels, 0.0f);
        int playing = 0;
        {
            std::lock_guard<std::mutex> lock(g_ngs2_mx);
            for (auto& [k, vc] : g_ngs2_voices) if (vc.playing && vc.data_ptr) playing++;
            ngs2_mix_voices(a0, mix.data(), frames, out_channels, 48000 /* NGS2 system rate */);
        }
        if (ngs2_trace_enabled()) {
            float pk = 0; for (float f : mix) { float a = f < 0 ? -f : f; if (a > pk) pk = a; }
            static uint64_t rc = 0; static float maxpk = 0; static int maxpv = 0;
            if (pk > maxpk) maxpk = pk; if (playing > maxpv) maxpv = playing;
            if ((++rc & 0x3f) == 1 || (pk > 0 && playing > 0))
                fprintf(stderr, "[ngs2] render buf#%llu out=%s ch=%u frames=%u size=%llu playing_voices=%d(max%d) mix_peak=%.4f(max%.4f)\n",
                        (unsigned long long)i, out_s16 ? "s16" : "f32", out_channels, frames,
                        (unsigned long long)info.size, playing, maxpv, pk, maxpk);
        }
        // Serialize the float mix into the guest buffer's native format (clamped), then commit it.
        std::vector<uint8_t> out((size_t)info.size, 0);
        for (size_t s = 0; s < mix.size(); ++s) {
            float f = mix[s];
            f = f > 1.0f ? 1.0f : (f < -1.0f ? -1.0f : f);
            if (out_s16) { int16_t v = (int16_t)(f * 32767.0f); memcpy(&out[s * 2], &v, 2); }
            else         { memcpy(&out[s * 4], &f, 4); }
        }
        if (!audio_store_bytes(info.buffer, out.data(), out.size())) return kNgs2ErrInvalidOut;
    }
    return 0;
}

// SystemRender fires guest waveform-block callbacks (queued during mixing). Guest code must run with
// the caller's guest %fs, recovered from the import-stub frame via the shared entry trampoline (it
// forwards entry %rsp as a 7th argument), and after mixing has released g_ngs2_mx. Non-Linux platforms
// register the plain handler; their guest-callback dispatch would need prosper_call_guest_sysv instead.
#if defined(__linux__)
extern "C" uint64_t ngs2_system_render_c(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                         uint64_t a4, uint64_t a5, uint64_t entry_rsp);
PROSPER_ASM_TRAMPOLINE(ngs2_system_render_entry, ngs2_system_render_c)
extern "C" void ngs2_system_render_entry();
extern "C" uint64_t ngs2_system_render_c(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                         uint64_t a4, uint64_t a5, uint64_t entry_rsp) {
    t_ngs2_render_guest_fs = prosper::callback_guest_fs_from_entry_stack(entry_rsp);
    const uint64_t r = ngs2_system_render(a0, a1, a2, a3, a4, a5);
    ngs2_fire_pending_callbacks();
    t_ngs2_render_guest_fs = 0;
    return r;
}
#endif

HLE(ngs2_geom_reset_listener) {
    NGS2_LOG("sceNgs2GeomResetListenerParam");
    return a0 && a2_store_zeros(a0, 0xa0) ? 0 : kNgs2ErrInvalidOut;
}

HLE(ngs2_geom_calc_listener) {
    NGS2_LOG("sceNgs2GeomCalcListener");
    return a0 && a1 && a2_store_zeros(a1, 0x60) ? 0 : kNgs2ErrInvalidOut;
}

#undef NGS2_LOG

// --- libSceVoice (voice chat, #1158). prosper has no voice-capture backend (no microphone / network
// voice path). GTA V (PPSA04263, RAGE) initialises voice at boot and, when it believes voice is up,
// builds a voice context and sizes its ring buffers by DIVIDING by voice-parameter fields (bitrate,
// samples-per-frame, ...) it obtains from the voice API — e.g. `x * 8000 / bitrate` at eboot+0x26043b0
// and `x / ctx[+0x2dc0]` at eboot+0x2905... . prosper's generic unimplemented stubs returned 0 WITHOUT
// writing those outputs, so several divisors were 0 → a cascade of guest integer #DE → SIGFPE.
//
// Report the subsystem as unavailable from sceVoiceInit: GTA treats a failed voice init as "voice
// chat off" (a real state — e.g. voice disabled in system settings / no headset region) and skips its
// entire voice-buffer setup, so none of the divide-by-zero math runs and boot continues to the title.
// This is the honest model for a host with no voice hardware, and voice is not needed to reach the
// title screen. CONFIDENCE: MED — verified live to clear the SIGFPE cascade and reach the GTA V title;
// a faithful null-backend voice implementation (init succeeds, ports return valid silent parameters)
// is the alternative if a title ever needs working voice (tracked separately).
HLE(voice_init_unavailable) {   // sceVoiceInit / sceVoiceInitHQ -> report voice unavailable
    // Negative SCE_VOICE-class error (facility 0x8041; the guest only sign-checks the return via `js`).
    // Chosen as a HARD, non-retryable init failure: it is NOT the ALREADY_INITIALIZED code (0x80410004,
    // which some engines treat as success), so the guest takes its "voice off" branch and does not retry.
    return (uint64_t)(int64_t)(int32_t)0x80410002;
}

void register_audio_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("sceAudioOutInit", audio_init);
    R("sceAudioOutOpen", audio_open);
    // libSceVoice: report voice unavailable so titles with no host voice backend skip voice-chat
    // setup instead of #DE'ing on never-written voice-parameter divisors (GTA V, #1158). Both the
    // standard and HQ init entry points get the same "unavailable" answer.
    R("sceVoiceInit", voice_init_unavailable);
    R("sceVoiceInitHQ", voice_init_unavailable);
    R("sceAudioOutOutput", audio_output);
    R("sceAudioOutOutputs", audio_outputs);
    R("sceAudioOutSetVolume", audio_set_volume);
    R("sceAudioOutClose", audio_close);
    R("sceAudioOutGetPortState", audio_get_port_state);
    // libSceAudioIn inherited core: deterministic null microphone with real-time capture pacing.
    R("sceAudioInInit", audio_in_init);
    R("sceAudioInOpen", audio_in_open);
    R("sceAudioInInput", audio_in_input);
    R("sceAudioInClose", audio_in_close);
    // libSceAudioOut2 (PS5) — see the probe block above.
    R("sceAudioOut2Initialize", audio2_initialize);
    R("sceAudioOut2ContextResetParam", audio2_ctx_reset_param);
    R("sceAudioOut2ContextQueryMemory", audio2_ctx_query_memory);
    R("sceAudioOut2ContextCreate", audio2_ctx_create);
    R("sceAudioOut2ContextDestroy", audio2_ctx_destroy);
    R("sceAudioOut2ContextAdvance", audio2_ctx_advance);
    R("sceAudioOut2ContextPush", audio2_ctx_push);
    R("sceAudioOut2ContextGetQueueLevel", audio2_ctx_get_queue_level);
    R("sceAudioOut2ContextSetAttributes", audio2_ctx_set_attr);
    R("sceAudioOut2ContextBedWrite", audio2_ctx_bed_write);
    R("sceAudioOut2UserCreate", audio2_user_create);
    R("sceAudioOut2UserDestroy", audio2_user_destroy);
    R("sceAudioOut2PortCreate", audio2_port_create);
    R("sceAudioOut2PortDestroy", audio2_port_destroy);
    R("sceAudioOut2PortGetState", audio2_port_get_state);
    R("sceAudioOut2PortSetAttributes", audio2_port_set_attr);
    R("sceAudioOut2PortRegister", audio2_port_register);
    R("sceAudioOut2PortUnregister", audio2_port_unregister);
    R("sceAudioOut2GetSystemState", audio2_get_system_state);
    R("sceAudioOut2GetSpeakerInfo", audio2_get_speaker_info);
    R("sceAudioOut2GetSpeakerArrayMemorySize", audio2_get_speaker_array_memory_size);  // GTA V (#1134)
    R("sceAudioOut2SpeakerArrayCreate", audio2_speaker_array_create);
    R("sceAudioOut2SpeakerArrayDestroy", audio2_speaker_array_destroy);
    R("sceAudioOut2GetSpeakerArrayCoefficients", audio2_get_speaker_array_coefficients);
    R("sceAudioOut2GetSpeakerArrayAmbisonicsCoefficients",
      audio2_get_speaker_array_ambisonics_coefficients);
    R("sceAudioOut2MasteringInit", audio2_mastering_init);
    R("sceAudioOut2MasteringTerm", audio2_mastering_term);
    R("sceAudioOut2MasteringSetParam", audio2_mastering_set_param);
    R("sceAudioOut2MasteringGetState", audio2_mastering_get_state);
    // libSceAjm (#187): headless decode-lifecycle (valid handles, no actual decode -> silence).
    R("sceAjmInitialize", ajm_initialize);            R("sceAjmFinalize", ajm_finalize);
    R("sceAjmModuleRegister", ajm_module_register);   R("sceAjmModuleUnregister", ajm_module_unregister);
    R("sceAjmInstanceCreate", ajm_instance_create);   R("sceAjmInstanceDestroy", ajm_instance_destroy);
    R("sceAjmBatchInitialize", ajm_batch_initialize);
    R("sceAjmBatchStartBuffer", ajm_batch_start);     R("sceAjmBatchWait", ajm_batch_wait);
    R("sceAjmBatchCancel", ajm_batch_cancel);
    R("sceAjmBatchJobControlBufferRa", ajm_batch_job_control_buffer_ra);
    R("sceAjmBatchJobInlineBuffer", ajm_batch_job_inline_buffer);
    R("sceAjmBatchJobRunBufferRa", ajm_batch_job_run_buffer_ra);
    R("sceAjmBatchJobRunSplitBufferRa", ajm_batch_job_run_split_buffer_ra);
    R("sceAjmBatchErrorDump", ajm_batch_errordump);
    R("sceAjmBatchJobInitialize", ajm_batch_job_initialize);
    R("sceAjmBatchJobDecode", ajm_batch_job_decode);
    R("sceAjmBatchJobDecodeSplit", ajm_batch_job_decode_split);
    R("sceAjmBatchJobClearContext", ajm_batch_job_clear_context);
    R("sceAjmBatchJobSetResampleParametersEx", ajm_batch_job_set_resample_ex);
    R("sceAjmBatchJobGetResampleInfo", ajm_batch_job_get_resample_info);
    R("sceAjmBatchJobGetStatistics", ajm_batch_job_get_statistics);
    R("sceAjmBatchJobSetGaplessDecode", ajm_batch_job_gapless);
    R("sceAjmBatchStart", ajm_batch_start2);
    Hle::register_fn("pgFAiLR5qT4", ngs2_system_query_buffer, "sceNgs2SystemQueryBufferSize");
    Hle::register_fn("koBbCMvOKWw", ngs2_system_create, "sceNgs2SystemCreate");
    Hle::register_fn("0eFLVCfWVds", ngs2_rack_query_buffer, "sceNgs2RackQueryBufferSize");
    Hle::register_fn("cLV4aiT9JpA", ngs2_rack_create, "sceNgs2RackCreate");
    Hle::register_fn("MwmHz8pAdAo", ngs2_rack_get_voice, "sceNgs2RackGetVoiceHandle");
    Hle::register_fn("uu94irFOGpA", ngs2_voice_control, "sceNgs2VoiceControl");
    Hle::register_fn("AbYvTOZ8Pts", ngs2_voice_run_commands, "sceNgs2VoiceRunCommands");
    Hle::register_fn("-TOuuAQ-buE", ngs2_voice_get_state, "sceNgs2VoiceGetState");
    Hle::register_fn("rEh728kXk3w", ngs2_voice_get_state_flags, "sceNgs2VoiceGetStateFlags");
    Hle::register_fn("lCqD7oycmIM", ngs2_rack_destroy, "sceNgs2RackDestroy");
    Hle::register_fn("eF8yRCC6W64", ngs2_geom_apply, "sceNgs2GeomApply");
    Hle::register_fn("0lbbayqDNoE", ngs2_geom_reset_source, "sceNgs2GeomResetSourceParam");
#if defined(__linux__)
    // The trampoline forwards entry %rsp so the handler can recover the caller's guest %fs and fire
    // waveform-block completion callbacks (guest code) after mixing. See ngs2_system_render_c.
    Hle::register_fn("i0VnXM-C9fc", (HleFn)ngs2_system_render_entry, "sceNgs2SystemRender");
#else
    Hle::register_fn("i0VnXM-C9fc", ngs2_system_render, "sceNgs2SystemRender");
#endif
    Hle::register_fn("7Lcfo8SmpsU", ngs2_geom_reset_listener, "sceNgs2GeomResetListenerParam");
    Hle::register_fn("1WsleK-MTkE", ngs2_geom_calc_listener, "sceNgs2GeomCalcListener");
    #undef R
}

} // namespace prosper
