// test_audio — behavioral tests for the sceAudioOut and NGS2 HLEs in hle_audio.cpp.
//
// Agentic-first: no device, no PATH/DLL setup. Installs a fake AudioSink, drives the real HLE
// entrypoints through the dispatch table (so registration + arg decoding + forwarding are all
// exercised), and asserts the port lifecycle, format decoding, PCM forwarding, volume and error
// paths. Exit code is truth.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/audio/audio.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(cond) do { if (!(cond)) { printf("  [FAIL] %s:%d  %s\n", __FILE__, __LINE__, #cond); fails++; } } while (0)

// Records every backend event so the test can assert on them.
struct CapturingSink : AudioSink {
    struct OpenEv  { int port; AudioPortInfo info; };
    struct OutEv   { int port; int frames; std::vector<uint8_t> pcm; };
    struct VolEv   { int port; uint32_t mask; std::vector<int> vols; };
    std::vector<OpenEv> opens;
    std::vector<OutEv>  outs;
    std::vector<VolEv>  vols_;
    std::vector<int>    closes;

    bool open(int port, const AudioPortInfo& info) override { opens.push_back({port, info}); return true; }
    void output(int port, const void* pcm, int frames) override {
        OutEv e; e.port = port; e.frames = frames;
        // Copy the delivered grain so we can verify byte-for-byte (uses the port's most-recent open).
        int fb = frames * (opens.empty() ? 4 : audio_frame_bytes(opens.back().info));
        e.pcm.assign((const uint8_t*)pcm, (const uint8_t*)pcm + fb);
        outs.push_back(std::move(e));
    }
    void set_volume(int port, uint32_t mask, const int* v) override {
        VolEv e; e.port = port; e.mask = mask;
        for (int c = 0; c < 8; c++) if (mask & (1u << c)) e.vols.push_back(v[c]);
        vols_.push_back(std::move(e));
    }
    void close(int port) override { closes.push_back(port); }
};

// Holds output in flight while another thread destroys its AudioOut2 context. The private gate
// also makes a racing close observable before it can complete, so the test can distinguish proper
// slot serialization from a clear-first/close-later implementation without touching HLE internals.
struct BlockingSink : AudioSink {
    std::mutex gate, state_mx;
    std::condition_variable state_cv;
    bool output_entered = false;
    bool release_output = false;
    bool output_finished = false;
    bool close_attempted = false;
    bool close_finished = false;

    bool open(int, const AudioPortInfo&) override { return true; }
    void output(int, const void*, int) override {
        std::lock_guard<std::mutex> operation(gate);
        std::unique_lock<std::mutex> state(state_mx);
        output_entered = true;
        state_cv.notify_all();
        state_cv.wait(state, [&] { return release_output; });
        output_finished = true;
        state_cv.notify_all();
    }
    void close(int) override {
        {
            std::lock_guard<std::mutex> state(state_mx);
            close_attempted = true;
            state_cv.notify_all();
        }
        std::lock_guard<std::mutex> operation(gate);
        std::lock_guard<std::mutex> state(state_mx);
        close_finished = true;
        state_cv.notify_all();
    }
    bool wait_for_output() {
        std::unique_lock<std::mutex> state(state_mx);
        return state_cv.wait_for(state, std::chrono::seconds(2), [&] { return output_entered; });
    }
    bool wait_for_close_attempt(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> state(state_mx);
        return state_cv.wait_for(state, timeout, [&] { return close_attempted; });
    }
    void unblock_output() {
        std::lock_guard<std::mutex> state(state_mx);
        release_output = true;
        state_cv.notify_all();
    }
    bool completed_in_order() {
        std::lock_guard<std::mutex> state(state_mx);
        return output_finished && close_attempted && close_finished;
    }
};

// Typed shims over the dispatch table.
static HleFn FN(const char* n) { return Hle::lookup(nid_hash(n)); }
static int64_t call(const char* n, uint64_t a0 = 0, uint64_t a1 = 0, uint64_t a2 = 0,
                    uint64_t a3 = 0, uint64_t a4 = 0, uint64_t a5 = 0) {
    HleFn f = FN(n);
    if (!f) { printf("  [FAIL] not registered: %s\n", n); fails++; return -999; }
    return (int64_t)f(a0, a1, a2, a3, a4, a5);
}
static int64_t call_raw(const char* nid, uint64_t a0 = 0, uint64_t a1 = 0, uint64_t a2 = 0,
                        uint64_t a3 = 0, uint64_t a4 = 0, uint64_t a5 = 0) {
    HleFn f = Hle::lookup(nid);
    if (!f) { printf("  [FAIL] raw NID not registered: %s\n", nid); fails++; return -999; }
    return (int64_t)f(a0, a1, a2, a3, a4, a5);
}
static uint64_t PTR(const void* p) { return (uint64_t)(uintptr_t)p; }

// AudioSignalStats is the decision rule PROSPER_AUDIO_FLOW rests on when classifying a silent
// title: "the guest submitted nothing" vs "submitted exactly-zero PCM" vs "submitted real signal".
// A wrong answer here sends the next investigation at the wrong subsystem entirely, so pin the
// distinction that peak and RMS alone cannot make — quiet-but-real must NOT read as silent.
static void test_signal_stats() {
    // An all-zero buffer: the signature of a guest that submitted a cleared grain.
    AudioSignalStats zero;
    for (int i = 0; i < 512; i++) zero.add(0.0f);
    CHECK(zero.samples == 512);
    CHECK(zero.nonzero == 0);
    CHECK(zero.silent());
    CHECK(zero.peak == 0.0);
    CHECK(zero.rms() == 0.0);

    // A genuinely QUIET but real mix, far below any threshold one would pick for "audible".
    // This is the case an RMS-threshold rule gets wrong, and the assertion that fails if the
    // measure is changed to one.
    AudioSignalStats quiet;
    for (int i = 0; i < 512; i++) quiet.add(i % 2 ? 3.0e-7f : -3.0e-7f);
    CHECK(quiet.nonzero == 512);
    CHECK(!quiet.silent());
    CHECK(quiet.peak < 1.0e-6);
    CHECK(quiet.rms() < 1.0e-6);
    CHECK(quiet.rms() > 0.0);              // but exactly, measurably non-zero

    // NaN is the one input class where "non-zero" and "non-zero peak" disagree: `a > peak` is false
    // for NaN, so a peak-based rule reports an all-NaN buffer as silent while the nonzero count
    // reports it as full. Pins the semantics against a peak==0.0 implementation, which satisfies
    // every other case here. The probe counts NaN separately and then treats it as the silence the
    // sink's clamp turns it into, so this distinction must stay visible at the accumulator.
    AudioSignalStats nans;
    const float nan_v = std::nan("");
    for (int i = 0; i < 64; i++) nans.add(nan_v);
    CHECK(nans.samples == 64);
    CHECK(nans.nonzero == 64);             // a peak-based rule would say 0 here
    CHECK(!nans.silent());
    CHECK(nans.peak == 0.0);               // NaN never raises the running peak

    // Real programme material: peak and RMS both meaningful, nonzero tracks the samples.
    AudioSignalStats loud;
    for (int i = 0; i < 4; i++) loud.add(i == 2 ? -0.5f : 0.25f);
    CHECK(loud.samples == 4);
    CHECK(loud.nonzero == 4);
    CHECK(!loud.silent());
    CHECK(std::fabs(loud.peak - 0.5) < 1e-9);
    CHECK(std::fabs(loud.rms() - std::sqrt((0.0625 * 3 + 0.25) / 4.0)) < 1e-9);

    // A lone transient in an otherwise silent buffer: peak alone would call this a healthy signal,
    // so the report must expose that only one sample of many is actually non-zero.
    AudioSignalStats transient;
    transient.add(1.0f);
    for (int i = 0; i < 999; i++) transient.add(0.0f);
    CHECK(transient.peak == 1.0);
    CHECK(transient.nonzero == 1);
    CHECK(transient.samples == 1000);
    CHECK(!transient.silent());
    CHECK(transient.rms() < 0.04);         // ~0.0316: far below the peak it reports

    // An empty accumulator must not divide by zero, and reset must return to that state.
    AudioSignalStats empty;
    CHECK(empty.rms() == 0.0);
    CHECK(empty.silent());
    loud.reset();
    CHECK(loud.samples == 0 && loud.nonzero == 0 && loud.peak == 0.0 && loud.rms() == 0.0);
}

// audio_stereo_downmix is the whole correctness surface of multichannel MAIN output: the mix loop
// does nothing but apply it. It is a pure function of the channel count, so pin the layout contract
// here rather than only through a live boot.
static void test_stereo_downmix() {
    constexpr float kC = 0.70710678f;   // -3 dB, used for centre and surround
    constexpr float kL = 0.5f;          // -6 dB, LFE
    AudioStereoGain g[kAudioMaxBedChannels];
    auto near = [](float a, float b) { return std::fabs(a - b) < 1e-6f; };

    // Mono feeds both sides; stereo is identity. Nothing is unplaced in either.
    CHECK(audio_stereo_downmix(1, g, kAudioMaxBedChannels) == 0);
    CHECK(near(g[0].left, 1.0f) && near(g[0].right, 1.0f));
    CHECK(audio_stereo_downmix(2, g, kAudioMaxBedChannels) == 0);
    CHECK(near(g[0].left, 1.0f) && near(g[0].right, 0.0f));
    CHECK(near(g[1].left, 0.0f) && near(g[1].right, 1.0f));

    // 3ch adds FC (both sides), 4ch is quad (a second L/R pair), 5ch is FL/FR/FC/SL/SR.
    CHECK(audio_stereo_downmix(3, g, kAudioMaxBedChannels) == 0);
    CHECK(near(g[2].left, kC) && near(g[2].right, kC));
    CHECK(audio_stereo_downmix(4, g, kAudioMaxBedChannels) == 0);
    CHECK(near(g[2].left, kC) && near(g[2].right, 0.0f));
    CHECK(near(g[3].left, 0.0f) && near(g[3].right, kC));
    CHECK(audio_stereo_downmix(5, g, kAudioMaxBedChannels) == 0);
    CHECK(near(g[2].left, kC) && near(g[2].right, kC));
    CHECK(near(g[3].left, kC) && near(g[3].right, 0.0f));
    CHECK(near(g[4].left, 0.0f) && near(g[4].right, kC));

    // 6..8: FL FR FC LFE then surround pairs. 7ch's single rear-centre splits across both sides.
    CHECK(audio_stereo_downmix(6, g, kAudioMaxBedChannels) == 0);
    CHECK(near(g[2].left, kC) && near(g[2].right, kC));
    CHECK(near(g[3].left, kL) && near(g[3].right, kL));
    CHECK(near(g[4].left, kC) && near(g[4].right, 0.0f));
    CHECK(near(g[5].left, 0.0f) && near(g[5].right, kC));
    CHECK(audio_stereo_downmix(7, g, kAudioMaxBedChannels) == 0);
    CHECK(near(g[6].left, kC * 0.5f) && near(g[6].right, kC * 0.5f));
    CHECK(audio_stereo_downmix(8, g, kAudioMaxBedChannels) == 0);
    CHECK(near(g[6].left, kC) && near(g[6].right, 0.0f));
    CHECK(near(g[7].left, 0.0f) && near(g[7].right, kC));

    // 9..16 (#1700): the bed is no longer DISCARDED — its first eight channels fold exactly as an
    // 8-channel bed does, which is where every measured title puts its content. The remaining
    // channels have no measured side, so they are reported as unplaced and contribute nothing;
    // that count is the fail-visible signal, and it must not silently become zero.
    for (unsigned ch = 9; ch <= kAudioMaxBedChannels; ++ch) {
        AudioStereoGain wide[kAudioMaxBedChannels];
        CHECK(audio_stereo_downmix(ch, wide, kAudioMaxBedChannels) == ch - 8);
        AudioStereoGain eight[kAudioMaxBedChannels];
        CHECK(audio_stereo_downmix(8, eight, kAudioMaxBedChannels) == 0);
        for (unsigned c = 0; c < 8; ++c)
            CHECK(near(wide[c].left, eight[c].left) && near(wide[c].right, eight[c].right));
        for (unsigned c = 8; c < ch; ++c)
            CHECK(wide[c].left == 0.0f && wide[c].right == 0.0f);
    }

    // Every placed bed keeps left/right balanced, and every channel is placed on exactly one side or
    // on both — never left half-placed. The pairwise check is what a summed one would miss: two
    // channels swapped between sides sum identically, and a total is also blind to a channel that
    // reaches one side only by accident.
    //
    // Note what this file DOES pin, because it is easy to undersell: the explicit per-index
    // assertions above fix index -> side for widths 6, 7 and 8, so a build that swapped the fold's
    // left and right fails here. What no test in this repo can settle is whether that mapping is
    // RIGHT AGAINST HARDWARE — prosper's live probe groups a bed's channels by side but cannot
    // orient the groups, so the orientation rests on the index-0 = FrontLeft convention. That is a
    // narrower and more useful claim than "nothing here can see a swap", and #1720 owns it.
    for (unsigned ch = 1; ch <= kAudioMaxBedChannels; ++ch) {
        AudioStereoGain m[kAudioMaxBedChannels];
        CHECK(audio_stereo_downmix(ch, m, kAudioMaxBedChannels) == (ch > 8 ? ch - 8 : 0));
        float sum_l = 0.0f, sum_r = 0.0f;
        unsigned left_only = 0, right_only = 0, both = 0, neither = 0;
        for (unsigned c = 0; c < ch; ++c) {
            sum_l += m[c].left;
            sum_r += m[c].right;
            const bool l = m[c].left != 0.0f, r = m[c].right != 0.0f;
            if (l && r) ++both; else if (l) ++left_only; else if (r) ++right_only; else ++neither;
            // Both-sided channels are centre-like and must be centred, not merely non-zero.
            if (l && r) CHECK(near(m[c].left, m[c].right));
        }
        CHECK(near(sum_l, sum_r));
        CHECK(left_only == right_only);                       // side-exclusive channels come in pairs
        CHECK(neither == (ch > 8 ? ch - 8 : 0));               // exactly the unplaced ones
        CHECK(both + left_only + right_only + neither == ch);  // no channel counted twice
    }
    // Mono is the one layout where a single channel is legitimately both-sided at full gain, and
    // 2ch the one where neither channel is: pin both so the classification above cannot drift.
    CHECK(audio_stereo_downmix(1, g, kAudioMaxBedChannels) == 0);
    CHECK(g[0].left != 0.0f && g[0].right != 0.0f);
    CHECK(audio_stereo_downmix(2, g, kAudioMaxBedChannels) == 0);
    CHECK(g[0].right == 0.0f && g[1].left == 0.0f);

    // Refusals: an unrepresentable width or an undersized output must place nothing and say so.
    AudioStereoGain guard[kAudioMaxBedChannels];
    for (auto& e : guard) e = AudioStereoGain{7.0f, 7.0f};
    CHECK(audio_stereo_downmix(0, guard, kAudioMaxBedChannels) == 0);
    CHECK(audio_stereo_downmix(kAudioMaxBedChannels + 1, guard, kAudioMaxBedChannels)
          == kAudioMaxBedChannels + 1);
    CHECK(audio_stereo_downmix(8, guard, 4) == 8);
    CHECK(audio_stereo_downmix(8, nullptr, kAudioMaxBedChannels) == 8);
    for (auto& e : guard) CHECK(e.left == 7.0f && e.right == 7.0f);   // untouched by every refusal
}

// The decision rule behind PROSPER_AUDIO_STAMP (hle_audio.cpp). The probe answers the one question
// no amount of reading a guest buffer can: does the guest WRITE the grain prosper reads? Its whole
// output is a four-way verdict, and two of the four are inversions of each other — so the ordering
// of the tests inside the classifier is the entire correctness surface, and a live log looks
// perfectly plausible with them swapped.
static void test_stamped_grain_verdicts() {
    using V = AudioGrainVerdict;

    // The stamp is deliberately non-zero in every sample, so a SURVIVING stamp carries a large
    // non-zero count. A classifier that decided emptiness without consulting identity therefore
    // reports `Overwritten` — "the guest is filling this buffer with signal" — for the case that
    // actually means "prosper is reading memory the guest never touches", inverting the probe's
    // whole answer. This arm is what fails when the identity test is dropped.
    //
    // What it does NOT pin, stated because the comment here first claimed it did: the ORDER of the
    // identity and emptiness tests. Swapping them is behaviour-preserving, since a surviving stamp
    // is never empty, and the mutation arm that swapped them passed. The order that IS pinned is
    // the address check coming first, by the two `false` arms below.
    CHECK(audio_classify_stamped_grain(true, true, 2048) == V::Intact);
    CHECK(audio_classify_stamped_grain(true, false, 0) == V::Cleared);
    CHECK(audio_classify_stamped_grain(true, false, 2048) == V::Overwritten);
    // A stamp read back from a DIFFERENT buffer says nothing about either buffer, so the address
    // check must come first — including in the case that would otherwise look like a clean verdict.
    CHECK(audio_classify_stamped_grain(false, true, 2048) == V::PointerMoved);
    CHECK(audio_classify_stamped_grain(false, false, 0) == V::PointerMoved);

    // The counting rule the verdict is drawn from. It must agree with AudioSignalStats::nonzero,
    // because a disagreement would let the stamp probe and PROSPER_AUDIO_FLOW report different
    // answers about the same buffer.
    float zeros[64]{};
    CHECK(audio_count_nonzero_samples(zeros, sizeof zeros, false) == 0);

    // -0.0f is THE boundary case, and it is not academic: a guest mixer that clears with a negated
    // accumulator leaves negative zeros, whose BYTES are not zero. A byte-wise emptiness test — the
    // obvious implementation — calls that buffer non-empty, and the probe then answers
    // `Overwritten` for precisely the cleared buffer it exists to identify.
    float negzeros[64];
    for (float& v : negzeros) v = -0.0f;
    CHECK(audio_count_nonzero_samples(negzeros, sizeof negzeros, false) == 0);
    bool any_byte_set = false;
    for (const uint8_t* b = (const uint8_t*)negzeros; b < (const uint8_t*)(negzeros + 64); ++b)
        if (*b) any_byte_set = true;
    CHECK(any_byte_set);            // the trap is real: these bytes are NOT zero
    AudioSignalStats negstats;
    for (float v : negzeros) negstats.add(v);
    CHECK(negstats.nonzero == 0);   // and the flow probe agrees with the count above

    float mixed[8] = {0.0f, 0.25f, 0.0f, -0.5f, 0.0f, 0.0f, 1.0f, -0.0f};
    CHECK(audio_count_nonzero_samples(mixed, sizeof mixed, false) == 3);
    int16_t s16[8] = {0, 1, 0, -1, 0, 0, 32767, 0};
    CHECK(audio_count_nonzero_samples(s16, sizeof s16, true) == 3);
    CHECK(audio_count_nonzero_samples(nullptr, 64, false) == 0);
    CHECK(audio_count_nonzero_samples(mixed, 0, false) == 0);
}

// --- END-TO-END bed routing: which GUEST channel reaches which HOST channel, at what gain ------
//
// test_stereo_downmix above pins the fold TABLE against literal values. This measures what the mix
// loop actually DOES with that table, by pushing one unit impulse per source channel through the
// real sceAudioOut2 entrypoints and reading the stereo grain the host sink receives. The two are
// different claims and neither implies the other: a correct table applied through a wrong stride,
// a wrong tap list, or a wrong side produces exactly the output a wrong table would, and no
// assertion on the pure function can see it.
//
// It also reports the result, because that report is the answer to the question #1720 asks — "which
// guest channel landed in which host channel, and with what orientation" — for every width the fold
// accepts, rather than for the two widths (8 and 12) the existing arms happen to exercise.
//
// The oracle for the bulk of the comparison is audio_stereo_downmix() itself, which is DELIBERATE
// and is the point: it tests the application, not the table. So that this cannot degenerate into a
// tautology if both ever drift together, a handful of placements are additionally asserted against
// LITERALS below, chosen at widths the rest of the file never pushes end-to-end.
static void test_bed_routing_matrix() {
    audio_reset();
    CapturingSink sink;
    audio_set_sink(&sink);

    constexpr uint32_t kGrain = 64;
    uint8_t ctx_param[0x40]{};
    CHECK(call("sceAudioOut2ContextResetParam", PTR(ctx_param)) == 0);
    *(uint32_t*)(ctx_param + 0x10) = kGrain;
    uint64_t ctx = 0;
    CHECK(call("sceAudioOut2ContextCreate", PTR(ctx_param), 0, 0, PTR(&ctx)) == 0);

    struct PortParam {
        uint16_t type, pad;
        uint32_t data_format, rate, flags;
        uint64_t user;
        uint32_t reserved[10];
    } pp{};
    static_assert(sizeof(PortParam) == 0x40);
    pp.type = 0;                       // MAIN: the only type folded into the host bed
    pp.rate = 48000;

    uint64_t pcm_ptr = 0;
    struct Attribute { uint32_t id, reserved; uint64_t value, value_size; } attr{
        0, 0, PTR(&pcm_ptr), sizeof(pcm_ptr)
    };

    printf("  [bed-routing] measured guest channel -> host L/R, through the real AudioOut2 mix loop\n");
    for (uint32_t w = 1; w <= kAudioMaxBedChannels; ++w) {
        AudioStereoGain declared[kAudioMaxBedChannels];
        const unsigned  unplaced = audio_stereo_downmix(w, declared, kAudioMaxBedChannels);
        CHECK(unplaced == (w > 8 ? w - 8 : 0));

        pp.data_format = (w << 8);     // f32, w channels
        uint64_t port = 0;
        CHECK(call("sceAudioOut2PortCreate", ctx, PTR(&pp), PTR(&port)) == 0);

        printf("  [bed-routing] width=%-2u", w);
        std::vector<float> grain((size_t)kGrain * w);
        for (uint32_t c = 0; c < w; ++c) {
            // One channel at unit level, every other channel exactly zero. Isolation is what makes
            // the reading a per-channel GAIN rather than a sum that several wrong matrices share.
            std::fill(grain.begin(), grain.end(), 0.0f);
            for (uint32_t f = 0; f < kGrain; ++f) grain[(size_t)f * w + c] = 1.0f;
            pcm_ptr = PTR(grain.data());
            sink.outs.clear();
            CHECK(call("sceAudioOut2PortSetAttributes", port, PTR(&attr), 1) == 0);
            CHECK(call("sceAudioOut2ContextPush", ctx, 1) == 0);
            CHECK(sink.outs.size() == 1);
            if (sink.outs.size() != 1) continue;
            CHECK(sink.outs[0].frames == (int)kGrain);
            const float* out = reinterpret_cast<const float*>(sink.outs[0].pcm.data());
            const float measured_l = out[0], measured_r = out[1];
            // A steady input must give a steady output: a per-frame drift would mean the fold is
            // reading across frame boundaries, which a single-frame check cannot see.
            for (uint32_t f = 1; f < kGrain; ++f) {
                CHECK(out[f * 2 + 0] == measured_l);
                CHECK(out[f * 2 + 1] == measured_r);
            }
            CHECK(std::abs(measured_l - declared[c].left) < 1e-6f);
            CHECK(std::abs(measured_r - declared[c].right) < 1e-6f);
            printf(" | ch%u L%.3f R%.3f%s", c, measured_l, measured_r,
                   (declared[c].left == 0.0f && declared[c].right == 0.0f) ? " UNPLACED" : "");
            // The unplaced tier measured end to end, for EVERY width above 8 rather than only the
            // one Dragon Quest VII happens to use: a channel with no known position must contribute
            // exactly nothing to either side, not merely something small.
            if (c >= 8) {
                CHECK(measured_l == 0.0f);
                CHECK(measured_r == 0.0f);
            }
        }
        printf("\n");
        CHECK(call("sceAudioOut2PortDestroy", port) == 0);
        sink.outs.clear();
    }

    // Literal anchors, so the loop above cannot pass by agreeing with a table that has itself
    // drifted. These are deliberately at widths 3, 5, 6 and 7 — the ones no other end-to-end arm in
    // this file pushes, which is exactly where an application bug could live undetected.
    struct { uint32_t width, channel; float left, right; } anchors[] = {
        {3, 2, 0.70710678f, 0.70710678f},   // 3ch centre: -3 dB into BOTH sides
        {5, 3, 0.70710678f, 0.0f},          // 5ch surround left: left ONLY
        {5, 4, 0.0f, 0.70710678f},          // 5ch surround right: right ONLY
        {6, 3, 0.5f, 0.5f},                 // 5.1 LFE: -6 dB into both
        {7, 6, 0.35355339f, 0.35355339f},   // 6.1 rear centre: -3 dB halved across the pair
    };
    for (auto& a : anchors) {
        pp.data_format = (a.width << 8);
        uint64_t port = 0;
        CHECK(call("sceAudioOut2PortCreate", ctx, PTR(&pp), PTR(&port)) == 0);
        std::vector<float> grain((size_t)kGrain * a.width, 0.0f);
        for (uint32_t f = 0; f < kGrain; ++f) grain[(size_t)f * a.width + a.channel] = 1.0f;
        pcm_ptr = PTR(grain.data());
        sink.outs.clear();
        CHECK(call("sceAudioOut2PortSetAttributes", port, PTR(&attr), 1) == 0);
        CHECK(call("sceAudioOut2ContextPush", ctx, 1) == 0);
        CHECK(sink.outs.size() == 1);
        if (!sink.outs.empty()) {
            const float* out = reinterpret_cast<const float*>(sink.outs[0].pcm.data());
            CHECK(std::abs(out[0] - a.left) < 1e-6f);
            CHECK(std::abs(out[1] - a.right) < 1e-6f);
        }
        CHECK(call("sceAudioOut2PortDestroy", port) == 0);
        sink.outs.clear();
    }

    // The same measurement over an S16 bed. Not symmetry for its own sake: every other end-to-end
    // AudioOut2 arm in this file that uses S16 is STEREO, so before this the multichannel fold had
    // no S16 coverage at all — `sample_at`'s integer branch, the `tmp_s16` sizing and the
    // conversion were only ever exercised through the 2-channel identity fold, where a stride or
    // scale error cannot show. Full-scale-half (16384 -> exactly 0.5) keeps the expected products
    // exact in binary floating point, so a 1e-6 tolerance is measuring the fold and not the codec.
    for (uint32_t w : {(uint32_t)6, (uint32_t)12}) {
        AudioStereoGain declared[kAudioMaxBedChannels];
        audio_stereo_downmix(w, declared, kAudioMaxBedChannels);
        pp.data_format = (w << 8) | 1u;                 // s16, w channels
        uint64_t port = 0;
        CHECK(call("sceAudioOut2PortCreate", ctx, PTR(&pp), PTR(&port)) == 0);
        printf("  [bed-routing] width=%-2u s16", w);
        std::vector<int16_t> grain((size_t)kGrain * w);
        for (uint32_t c = 0; c < w; ++c) {
            std::fill(grain.begin(), grain.end(), (int16_t)0);
            for (uint32_t f = 0; f < kGrain; ++f) grain[(size_t)f * w + c] = 16384;
            pcm_ptr = PTR(grain.data());
            sink.outs.clear();
            CHECK(call("sceAudioOut2PortSetAttributes", port, PTR(&attr), 1) == 0);
            CHECK(call("sceAudioOut2ContextPush", ctx, 1) == 0);
            CHECK(sink.outs.size() == 1);
            if (sink.outs.size() != 1) continue;
            const float* out = reinterpret_cast<const float*>(sink.outs[0].pcm.data());
            CHECK(std::abs(out[0] - declared[c].left * 0.5f) < 1e-6f);
            CHECK(std::abs(out[1] - declared[c].right * 0.5f) < 1e-6f);
            printf(" | ch%u L%.3f R%.3f", c, out[0], out[1]);
        }
        printf("\n");
        CHECK(call("sceAudioOut2PortDestroy", port) == 0);
        sink.outs.clear();
    }

    CHECK(call("sceAudioOut2ContextDestroy", ctx) == 0);
    audio_reset();
}

int main() {
    printf("== test_audio ==\n");
    register_builtin_hle();
    test_signal_stats();
    test_stereo_downmix();
    test_stamped_grain_verdicts();
    test_bed_routing_matrix();

    // --- 1. format decoding (all 8 SceAudioOutParamFormat values + an unknown) ---------------
    struct { uint32_t param; int ch; AudioFmt fmt; } fmts[] = {
        {0, 1, AudioFmt::S16}, {1, 2, AudioFmt::S16}, {2, 8, AudioFmt::S16}, {3, 1, AudioFmt::F32},
        {4, 2, AudioFmt::F32}, {5, 8, AudioFmt::F32}, {6, 8, AudioFmt::S16}, {7, 8, AudioFmt::F32},
        {0x1234, 2, AudioFmt::S16},                     // unknown low byte 0x34 -> default stereo S16
    };
    for (auto& t : fmts) {
        int ch; AudioFmt f; audio_decode_format(t.param, ch, f);
        CHECK(ch == t.ch); CHECK(f == t.fmt);
    }

    // AudioOut2 shares the guest-store primitive with AJM. Inaccessible outputs must report an
    // error instead of faulting the host; these cover both fixed-size zero-fill and u64 stores.
    CHECK((int32_t)call("sceAudioOut2ContextResetParam", 1) == (int32_t)0x80268001);
    CHECK((int32_t)call("sceAudioOut2ContextQueryMemory", 0, 1) == (int32_t)0x80268001);

    // Queue-level queries advance a wall clock and can observe a just-pushed grain before or after
    // it drains. Cover the positive accounting transition itself without that timing dependency.
    uint32_t reserved_grains = 0;
    CHECK(audio2_reserve_queue_slot(reserved_grains, 4));
    CHECK(reserved_grains == 1);
    reserved_grains = 4;
    CHECK(!audio2_reserve_queue_slot(reserved_grains, 4));
    CHECK(reserved_grains == 4);

    // --- 2. open -> handle + backend.open with decoded params --------------------------------
    audio_reset();
    CapturingSink sink; audio_set_sink(&sink);

    // sceAudioOutOpen(userId=1, type=0(MAIN), index=0, len=256, freq=48000, param=1(S16_STEREO))
    int64_t h = call("sceAudioOutOpen", 1, 0, 0, 256, 48000, 1);
    CHECK(h >= 1);
    CHECK(sink.opens.size() == 1);
    if (!sink.opens.empty()) {
        auto& o = sink.opens[0];
        CHECK(o.port == (int)h);
        CHECK(o.info.freq == 48000); CHECK(o.info.channels == 2);
        CHECK(o.info.fmt == AudioFmt::S16); CHECK(o.info.grain == 256);
        CHECK(audio_grain_bytes(o.info) == 256 * 2 * 2);   // 256 frames * 2ch * 2B
    }

    // --- 3. output forwards the exact grain (frames + bytes) ---------------------------------
    std::vector<uint8_t> pcm(256 * 2 * 2);
    for (size_t i = 0; i < pcm.size(); i++) pcm[i] = (uint8_t)(i * 7 + 3);
    int64_t n = call("sceAudioOutOutput", (uint64_t)h, PTR(pcm.data()));
    CHECK(n == 256);                                      // returns frames written
    CHECK(sink.outs.size() == 1);
    if (!sink.outs.empty()) {
        CHECK(sink.outs[0].port == (int)h);
        CHECK(sink.outs[0].frames == 256);
        CHECK(sink.outs[0].pcm == pcm);                  // byte-for-byte forwarded
    }
    // ptr == 0 is a drain request: returns 0, does NOT forward a grain.
    CHECK(call("sceAudioOutOutput", (uint64_t)h, 0) == 0);
    CHECK(sink.outs.size() == 1);

    // --- 4. sparse volume masks use Sony's channel-indexed array, not compacted values -------
    int vols[8] = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000};
    constexpr uint32_t sparse_mask = (1u << 4) | (1u << 7);
    int cached[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    audio_apply_channel_volumes(cached, sparse_mask, vols);
    CHECK(cached[4] == 5000); CHECK(cached[7] == 8000);
    CHECK(cached[0] == -1); CHECK(cached[6] == -1);
    CHECK(audio_peak_channel_volume(sparse_mask, vols) == 8000);
    CHECK(call("sceAudioOutSetVolume", (uint64_t)h, sparse_mask, PTR(vols)) == 0);
    CHECK(sink.vols_.size() == 1);
    if (!sink.vols_.empty()) {
        CHECK(sink.vols_[0].mask == sparse_mask);
        CHECK(sink.vols_[0].vols.size() == 2);
        CHECK(sink.vols_[0].vols[0] == 5000); CHECK(sink.vols_[0].vols[1] == 8000);
    }

    // --- 5. get port state fills the struct --------------------------------------------------
    uint8_t state[0x20]; memset(state, 0xEE, sizeof state);
    CHECK(call("sceAudioOutGetPortState", (uint64_t)h, PTR(state)) == 0);
    // Layout per Kyty Audio.cpp:340: u16 output @0; u8 channel @2; i16 volume @4; u16 reroute @6; u64 flag @8.
    CHECK(*(uint16_t*)(state + 0) == 1);                  // output enabled
    CHECK(*(uint8_t*)(state + 2) == 2);                   // channel (u8, capped at 2 for main port)
    CHECK(*(int16_t*)(state + 4) == 127);                 // volume (Kyty reports 127)
    CHECK(*(uint64_t*)(state + 8) == 0);                  // flag must NOT carry a bogus volume

    // Port routing is independent of the requested PCM channel count. Voice and Personal report
    // a mono headphone route, Aux reports the external route, and PadSpk reports its mono route.
    struct PortRoute { uint32_t type; uint16_t output; uint8_t channel; } routes[] = {
        {1, 0x01, 2}, {2, 0x40, 1}, {3, 0x40, 1}, {4, 0x04, 1}, {127, 0x80, 0},
    };
    for (const auto& route : routes) {
        int64_t route_h = call("sceAudioOutOpen", 1, route.type, 0, 256, 48000, 2 /* S16 8ch */);
        CHECK(route_h >= 1);
        uint8_t route_state[0x20]; memset(route_state, 0xEE, sizeof route_state);
        CHECK(call("sceAudioOutGetPortState", (uint64_t)route_h, PTR(route_state)) == 0);
        CHECK(*(uint16_t*)(route_state + 0) == route.output);
        CHECK(route_state[2] == route.channel);
        CHECK(call("sceAudioOutClose", (uint64_t)route_h) == 0);
    }

    // --- 6. error paths: the real SCE codes (Kyty Errno.h), not a generic -1 -----------------
    CHECK((int32_t)call("sceAudioOutOutput", 999, PTR(pcm.data())) == (int32_t)0x80260003);  // INVALID_PORT
    CHECK((int32_t)call("sceAudioOutSetVolume", 999, 0x1, PTR(vols)) == (int32_t)0x80260003);
    CHECK((int32_t)call("sceAudioOutClose", 999) == (int32_t)0x80260003);
    CHECK(call("sceAudioOutInit", 0) == 0);

    // --- 7. close, then output-after-close fails --------------------------------------------
    CHECK(call("sceAudioOutClose", (uint64_t)h) == 0);
    CHECK(!sink.closes.empty() && sink.closes.back() == (int)h);
    CHECK(call("sceAudioOutOutput", (uint64_t)h, PTR(pcm.data())) < 0);

    // --- 8. exhaustion: 16 ports open, 17th fails; float format decoded on open --------------
    audio_reset(); sink = CapturingSink{}; audio_set_sink(&sink);
    int opened = 0;
    for (int i = 0; i < 16; i++) if (call("sceAudioOutOpen", 1, 0, 0, 512, 44100, 4 /*F32 stereo*/) >= 1) opened++;
    CHECK(opened == 16);
    CHECK((int32_t)call("sceAudioOutOpen", 1, 0, 0, 512, 44100, 4) == (int32_t)0x80260005);  // PORT_FULL
    CHECK(sink.opens.size() == 16);
    if (!sink.opens.empty()) {
        CHECK(sink.opens[0].info.fmt == AudioFmt::F32);
        CHECK(sink.opens[0].info.freq == 44100);
        CHECK(audio_grain_bytes(sink.opens[0].info) == 512 * 2 * 4);
    }

    // --- 9. sceAudioOutOutputs (batch) forwards each valid entry -----------------------------
    audio_reset(); sink = CapturingSink{}; audio_set_sink(&sink);
    int64_t ha = call("sceAudioOutOpen", 1, 0, 0, 128, 48000, 1);
    int64_t hb = call("sceAudioOutOpen", 1, 0, 0, 128, 48000, 1);
    CHECK(ha >= 1 && hb >= 1);
    struct OutParam { int32_t handle; int32_t reserved; uint64_t ptr; } batch[3];
    std::vector<uint8_t> pa(128 * 2 * 2, 0x11), pb(128 * 2 * 2, 0x22);
    batch[0] = { (int32_t)ha, 0, PTR(pa.data()) };
    batch[1] = { 999,          0, PTR(pb.data()) };        // invalid handle -> skipped
    batch[2] = { (int32_t)hb, 0, PTR(pb.data()) };
    int64_t tot = call("sceAudioOutOutputs", PTR(batch), 3);
    CHECK(tot == 128);                                    // ONE grain (the shared slice), not the sum over
                                                          // ports (Kyty/shadPS4 both return a single grain)
    CHECK(sink.outs.size() == 2);                         // ...but both valid entries are still forwarded

    // --- 10. libSceAudioIn: tagged lifecycle, exact silence, errors, exhaustion, pacing -------
    CHECK(call("sceAudioInInit") == 0);
    CHECK((int32_t)call("sceAudioInOpen", 1, 1, 0, 0, 16000, 0) == (int32_t)0x80260102);
    CHECK((int32_t)call("sceAudioInOpen", 1, 1, 0, 2049, 16000, 0) == (int32_t)0x80260102);
    CHECK((int32_t)call("sceAudioInOpen", 1, 1, 0, 320, 44100, 0) == (int32_t)0x80260103);
    CHECK((int32_t)call("sceAudioInOpen", 1, 1, 0, 320, 16000, 1) == (int32_t)0x80260106);

    int64_t hi = call("sceAudioInOpen", 1, 1 /* GENERAL */, 0, 320, 16000, 0 /* S16 mono */);
    CHECK(hi >= 0 && ((uint32_t)hi & 0x7f000000u) == 0x30000000u);
    CHECK((int32_t)call("sceAudioInInput", (uint64_t)hi, 0) == (int32_t)0x80260105);
    CHECK((int32_t)call("sceAudioInInput", (uint64_t)hi, 1) == (int32_t)0x80260105);

    std::vector<uint8_t> mic(320 * 2 + 2, 0xA5);
    auto mic_start = std::chrono::steady_clock::now();
    CHECK(call("sceAudioInInput", (uint64_t)hi, PTR(mic.data() + 1)) == 320);
    auto mic_elapsed = std::chrono::steady_clock::now() - mic_start;
    CHECK(mic_elapsed >= std::chrono::milliseconds(10));  // 320 / 16 kHz = 20 ms; tolerate coarse clocks
    CHECK(mic.front() == 0xA5 && mic.back() == 0xA5);     // exact-size write canaries
    for (size_t i = 1; i + 1 < mic.size(); i++) CHECK(mic[i] == 0);

    CHECK(call("sceAudioInClose", (uint64_t)hi) == 0);
    memset(mic.data() + 1, 0x5A, mic.size() - 2);
    CHECK((int32_t)call("sceAudioInInput", (uint64_t)hi, PTR(mic.data() + 1)) ==
          (int32_t)0x80260109);
    for (size_t i = 1; i + 1 < mic.size(); i++) CHECK(mic[i] == 0x5A); // closed handle leaves output
    CHECK((int32_t)call("sceAudioInClose", (uint64_t)hi) == (int32_t)0x80260109);
    CHECK((int32_t)call("sceAudioInInput", 1, PTR(mic.data() + 1)) == (int32_t)0x80260101);
    CHECK((int32_t)call("sceAudioInClose", 1) == (int32_t)0x80260101);

    int64_t stereo_hi = call("sceAudioInOpen", 1, 0, 0, 64, 48000, 2 /* S16 stereo */);
    std::vector<uint8_t> stereo_mic(64 * 2 * 2 + 2, 0xC7);
    CHECK(call("sceAudioInInput", (uint64_t)stereo_hi, PTR(stereo_mic.data() + 1)) == 64);
    CHECK(stereo_mic.front() == 0xC7 && stereo_mic.back() == 0xC7);
    for (size_t i = 1; i + 1 < stereo_mic.size(); i++) CHECK(stereo_mic[i] == 0);
    CHECK(call("sceAudioInClose", (uint64_t)stereo_hi) == 0);

    int64_t input_handles[7];
    for (int i = 0; i < 7; i++) {
        input_handles[i] = call("sceAudioInOpen", 1, 0, i, 64, 48000, 2 /* S16 stereo */);
        CHECK(input_handles[i] >= 0);
        for (int j = 0; j < i; j++) CHECK(input_handles[i] != input_handles[j]);
    }
    CHECK((int32_t)call("sceAudioInOpen", 1, 0, 7, 64, 48000, 2) == (int32_t)0x80260107);
    audio_reset();
    memset(stereo_mic.data() + 1, 0x3C, stereo_mic.size() - 2);
    CHECK((int32_t)call("sceAudioInInput", (uint64_t)input_handles[0], PTR(stereo_mic.data() + 1)) ==
          (int32_t)0x80260109);
    for (size_t i = 1; i + 1 < stereo_mic.size(); i++) CHECK(stereo_mic[i] == 0x3C);
    for (int i = 0; i < 7; i++) {
        input_handles[i] = call("sceAudioInOpen", 1, 0, i, 64, 48000, 2);
        CHECK(input_handles[i] >= 0);                     // reset released every input slot
    }
    CHECK((int32_t)call("sceAudioInOpen", 1, 0, 7, 64, 48000, 2) == (int32_t)0x80260107);
    audio_reset();

    // --- 11. libSceAudioOut2: a dirty ABI padding word must not corrupt the attribute id --------
    audio_reset(); sink = CapturingSink{}; audio_set_sink(&sink);
    struct A2SpeakerPosition { int16_t azimuth, elevation; };
    struct A2SpeakerInfo {
        uint8_t type, reserved0;
        int16_t reserved1;
        uint32_t available_bits, flags, reserved2;
        A2SpeakerPosition positions[16];
    } a2_speakers;
    static_assert(sizeof(A2SpeakerInfo) == 0x50);
    memset(&a2_speakers, 0xCC, sizeof(a2_speakers));
    CHECK(call("sceAudioOut2GetSpeakerInfo", PTR(&a2_speakers), 1, 0) == 0);
    CHECK(a2_speakers.type == 0 && a2_speakers.available_bits == 0x3);
    CHECK(a2_speakers.flags == 0 && a2_speakers.reserved2 == 0);
    CHECK(a2_speakers.positions[0].azimuth == -30 && a2_speakers.positions[0].elevation == 0);
    CHECK(a2_speakers.positions[1].azimuth == 30 && a2_speakers.positions[1].elevation == 0);
    CHECK(a2_speakers.positions[2].azimuth == 0 && a2_speakers.positions[2].elevation == 0);
    CHECK((int32_t)call("sceAudioOut2GetSpeakerInfo", 1, 1, 0) == (int32_t)0x80268001);

    CHECK(call("sceAudioOut2GetSpeakerArrayMemorySize", 8, 0, 1) == 0xC00);
    uint64_t a2_speaker_array = 0;
    CHECK(call("sceAudioOut2SpeakerArrayCreate", PTR(&a2_speaker_array), 0, 0) == 0);
    CHECK(a2_speaker_array != 0);
    float a2_coefficients[8];
    for (float& value : a2_coefficients) value = -99.0f;
    CHECK(call("sceAudioOut2GetSpeakerArrayCoefficients", a2_speaker_array,
               PTR(a2_coefficients), 8, 0) == 0);
    CHECK(a2_coefficients[0] == 1.0f && a2_coefficients[1] == 1.0f);
    for (size_t i = 2; i < 8; ++i) CHECK(a2_coefficients[i] == 0.0f);
    for (float& value : a2_coefficients) value = -99.0f;
    CHECK(call("sceAudioOut2GetSpeakerArrayAmbisonicsCoefficients", a2_speaker_array,
               0, PTR(a2_coefficients), 8) == 0);
    CHECK(std::abs(a2_coefficients[0] - 0.70710677f) < 1e-6f);
    for (size_t i = 1; i < 8; ++i) CHECK(a2_coefficients[i] == 0.0f);
    CHECK(call("sceAudioOut2SpeakerArrayDestroy", a2_speaker_array) == 0);

    uint8_t a2_context_param[0x40]{};
    CHECK(call("sceAudioOut2ContextResetParam", PTR(a2_context_param)) == 0);
    CHECK(*(uint32_t*)(a2_context_param + 0x00) == 256);   // max_ports default
    CHECK(*(uint32_t*)(a2_context_param + 0x04) == 256);   // max_object_ports default
    CHECK(*(uint32_t*)(a2_context_param + 0x0c) == 4);     // queue_depth default
    CHECK(*(uint32_t*)(a2_context_param + 0x10) == 512);   // num_grains default
    CHECK(*(uint32_t*)(a2_context_param + 0x14) == 1);     // enabled
    *(uint32_t*)(a2_context_param + 0x10) = 64;            // one 64-frame grain
    uint64_t a2_context = 0;
    CHECK(call("sceAudioOut2ContextCreate", PTR(a2_context_param), 0, 0, PTR(&a2_context)) == 0);
    CHECK(a2_context != 0);

    uint32_t a2_queued = 0xCCCCCCCCu, a2_available = 0xCCCCCCCCu;
    CHECK(call("sceAudioOut2ContextGetQueueLevel", a2_context,
               PTR(&a2_queued), PTR(&a2_available)) == 0);
    CHECK(a2_queued == 0 && a2_available == 4);

    struct A2PortParam {
        uint16_t type, pad;
        uint32_t data_format, rate, flags;
        uint64_t user;
        uint32_t reserved[10];
    } a2_port_param{};
    static_assert(sizeof(A2PortParam) == 0x40);
    a2_port_param.type = 0;                               // MAIN speaker output
    a2_port_param.data_format = 0x200;                    // float32, two channels
    a2_port_param.rate = 48000;
    uint64_t a2_port = 0;
    CHECK(call("sceAudioOut2PortCreate", a2_context, PTR(&a2_port_param), PTR(&a2_port)) == 0);
    CHECK(a2_port != 0);

    struct A2PortState {
        uint16_t output;
        uint8_t num_channels, pad1;
        int16_t volume;
        uint16_t reroute_counter;
        uint32_t flags, pad2;
        uint64_t reserved[6];
    } a2_state;
    static_assert(sizeof(A2PortState) == 0x40);
    memset(&a2_state, 0xCC, sizeof a2_state);
    CHECK(call("sceAudioOut2PortGetState", a2_port, PTR(&a2_state)) == 0);
    CHECK(a2_state.output == 1 && a2_state.num_channels == 2 && a2_state.volume == 127);
    CHECK(a2_state.pad1 == 0 && a2_state.reroute_counter == 0);
    CHECK(a2_state.flags == 0 && a2_state.pad2 == 0);
    for (uint64_t value : a2_state.reserved) CHECK(value == 0);
    CHECK((int32_t)call("sceAudioOut2PortGetState", a2_port, 1) == (int32_t)0x80268001);

    std::vector<float> a2_pcm(64 * 2);
    for (size_t i = 0; i < a2_pcm.size(); i++) a2_pcm[i] = (float)((int)(i % 17) - 8) / 16.0f;
    uint64_t a2_pcm_ptr = PTR(a2_pcm.data());
    struct A2Attribute { uint32_t id, reserved; uint64_t value, value_size; } a2_attr{
        0, 0x7f2d1234u, PTR(&a2_pcm_ptr), sizeof(a2_pcm_ptr)
    };
    static_assert(sizeof(A2Attribute) == 0x18);
    CHECK(call("sceAudioOut2PortSetAttributes", a2_port, PTR(&a2_attr), 1) == 0);
    CHECK(call("sceAudioOut2ContextPush", a2_context, 1) == 0);
    a2_queued = a2_available = 0xCCCCCCCCu;
    CHECK(call("sceAudioOut2ContextGetQueueLevel", a2_context,
               PTR(&a2_queued), PTR(&a2_available)) == 0);
    // GetQueueLevel advances the emulated 48 kHz device clock. The 64-frame grain lasts only
    // 1.33 ms, so a slow/suspended runner may legitimately drain it during Push's synchronous
    // sink copy. The sink assertions below prove submission; here assert the point-in-time queue
    // contract without assuming the host schedules this query before the first drain boundary.
    CHECK(a2_queued <= 1);
    CHECK(a2_queued + a2_available == 4);
    CHECK(sink.opens.size() == 1);
    CHECK(sink.outs.size() == 1);
    if (!sink.outs.empty()) {
        CHECK(sink.outs[0].port == 17);
        CHECK(sink.outs[0].frames == 64);
        CHECK(sink.outs[0].pcm.size() == a2_pcm.size() * sizeof(float));
        CHECK(memcmp(sink.outs[0].pcm.data(), a2_pcm.data(), sink.outs[0].pcm.size()) == 0);
    }
    CHECK(call("sceAudioOut2PortDestroy", a2_port) == 0);

    // #3411: MAIN and auxiliary ports can submit from the same scratch buffer. The guest
    // overwrites it between SetAttributes calls, and again before Advance/Push. Each port must
    // keep its own submission, including on the S16 path; delaying the copy until Push loses it.
    for (uint32_t format : {0x200u, 0x201u}) {
        auto main_param = a2_port_param;
        main_param.data_format = format;
        auto aux_param = main_param;
        aux_param.type = 6;
        uint64_t main_port = 0, aux_port = 0, second_main = 0;
        CHECK(call("sceAudioOut2PortCreate", a2_context, PTR(&main_param), PTR(&main_port)) == 0);
        CHECK(call("sceAudioOut2PortCreate", a2_context, PTR(&aux_param), PTR(&aux_port)) == 0);
        CHECK(call("sceAudioOut2PortCreate", a2_context, PTR(&main_param), PTR(&second_main)) == 0);
        std::vector<float> scratch_f32(64 * 2, 0.25f);
        std::vector<int16_t> scratch_s16(64 * 2, 8192);
        uint64_t scratch = format == 0x200 ? PTR(scratch_f32.data()) : PTR(scratch_s16.data());
        A2Attribute scratch_attr{0, 0, PTR(&scratch), sizeof(scratch)};
        CHECK(call("sceAudioOut2PortSetAttributes", main_port, PTR(&scratch_attr), 1) == 0);
        for (size_t i = 0; i < scratch_f32.size(); ++i) {
            scratch_f32[i] = i % 2 ? -0.125f : 0.125f;
            scratch_s16[i] = i % 2 ? -4096 : 4096;
        }
        CHECK(call("sceAudioOut2PortSetAttributes", second_main, PTR(&scratch_attr), 1) == 0);
        std::fill(scratch_f32.begin(), scratch_f32.end(), -0.5f);
        std::fill(scratch_s16.begin(), scratch_s16.end(), -16384);
        CHECK(call("sceAudioOut2PortSetAttributes", aux_port, PTR(&scratch_attr), 1) == 0);
        std::fill(scratch_f32.begin(), scratch_f32.end(), 0.0f);
        std::fill(scratch_s16.begin(), scratch_s16.end(), 0);
        CHECK(call("sceAudioOut2ContextAdvance", a2_context) == 0);
        sink.outs.clear();
        CHECK(call("sceAudioOut2ContextPush", a2_context, 1) == 0);
        CHECK(sink.outs.size() == 1);
        if (!sink.outs.empty()) {
            const float* stereo = reinterpret_cast<const float*>(sink.outs[0].pcm.data());
            for (size_t i = 0; i < 64 * 2; ++i) CHECK(stereo[i] == (i % 2 ? 0.125f : 0.375f));
        }
        // A late producer must not loop either port's old grain or starve the host stream.
        sink.outs.clear();
        CHECK(call("sceAudioOut2ContextAdvance", a2_context) == 0);
        CHECK(call("sceAudioOut2ContextPush", a2_context, 1) == 0);
        CHECK(sink.outs.size() == 1);
        if (!sink.outs.empty())
            CHECK(audio_count_nonzero_samples(sink.outs[0].pcm.data(),
                                              sink.outs[0].pcm.size(), false) == 0);
        // Republishing that same address replaces the saved grain, including explicit silence.
        CHECK(call("sceAudioOut2PortSetAttributes", main_port, PTR(&scratch_attr), 1) == 0);
        sink.outs.clear();
        CHECK(call("sceAudioOut2ContextPush", a2_context, 1) == 0);
        CHECK(sink.outs.size() == 1);
        if (!sink.outs.empty())
            CHECK(audio_count_nonzero_samples(sink.outs[0].pcm.data(),
                                              sink.outs[0].pcm.size(), false) == 0);
        CHECK(call("sceAudioOut2PortDestroy", main_port) == 0);
        CHECK(call("sceAudioOut2PortDestroy", aux_port) == 0);
        CHECK(call("sceAudioOut2PortDestroy", second_main) == 0);
    }

    // AudioOut2 main-bed flag bit 1 reserves 20 dB of digital headroom for platform mastering.
    // Restore that reference level only for marked ports, then saturate at the host PCM boundary;
    // the flag-zero byte-for-byte assertion above is the control that ordinary/video ports keep
    // their original level.
    sink.outs.clear();
    a2_port_param.flags = 2;
    std::vector<float> a2_reference(64 * 2);
    constexpr float reference_pattern[4] = {0.025f, -0.05f, 0.125f, -0.2f};
    for (size_t i = 0; i < a2_reference.size(); ++i)
        a2_reference[i] = reference_pattern[i % 4];
    a2_pcm_ptr = PTR(a2_reference.data());
    CHECK(call("sceAudioOut2PortCreate", a2_context, PTR(&a2_port_param), PTR(&a2_port)) == 0);
    CHECK(call("sceAudioOut2PortSetAttributes", a2_port, PTR(&a2_attr), 1) == 0);
    CHECK(call("sceAudioOut2ContextPush", a2_context, 1) == 0);
    CHECK(sink.outs.size() == 1);
    if (!sink.outs.empty()) {
        const float* mastered = reinterpret_cast<const float*>(sink.outs[0].pcm.data());
        for (size_t i = 0; i < a2_reference.size(); ++i) {
            const float amplified = a2_reference[i] * 10.0f;
            const float expected = std::max(-1.0f, std::min(1.0f, amplified));
            CHECK(std::abs(mastered[i] - expected) < 1e-7f);
        }
    }
    CHECK(call("sceAudioOut2PortDestroy", a2_port) == 0);
    sink.outs.clear();
    a2_port_param.flags = 0;

    // A stereo host must hear content routed exclusively to center/LFE/surround speakers. This
    // 7.1 grain deliberately leaves FL/FR silent; the old front-pair-only path emitted silence.
    a2_port_param.data_format = 0x800;                    // float32, eight channels
    CHECK(call("sceAudioOut2PortCreate", a2_context, PTR(&a2_port_param), PTR(&a2_port)) == 0);
    memset(&a2_state, 0xCC, sizeof a2_state);
    CHECK(call("sceAudioOut2PortGetState", a2_port, PTR(&a2_state)) == 0);
    CHECK(a2_state.output == 1 && a2_state.num_channels == 8 && a2_state.volume == 127);
    std::vector<float> a2_surround(64 * 8, 0.0f);
    for (size_t frame = 0; frame < 64; frame++) {
        float* sample = a2_surround.data() + frame * 8;
        sample[2] = 0.02f; // FC
        sample[3] = 0.01f; // LFE
        sample[4] = 0.03f; // BL
        sample[5] = 0.04f; // BR
        sample[6] = 0.05f; // SL
        sample[7] = 0.06f; // SR
    }
    a2_pcm_ptr = PTR(a2_surround.data());
    CHECK(call("sceAudioOut2PortSetAttributes", a2_port, PTR(&a2_attr), 1) == 0);
    CHECK(call("sceAudioOut2ContextPush", a2_context, 1) == 0);
    CHECK(sink.outs.size() == 1);
    if (!sink.outs.empty()) {
        const float* stereo = reinterpret_cast<const float*>(sink.outs[0].pcm.data());
        constexpr float kMinus3Db = 0.70710678f;
        const float expected_l = 0.02f * kMinus3Db + 0.01f * 0.5f
                               + (0.03f + 0.05f) * kMinus3Db;
        const float expected_r = 0.02f * kMinus3Db + 0.01f * 0.5f
                               + (0.04f + 0.06f) * kMinus3Db;
        for (size_t frame = 0; frame < 64; frame++) {
            CHECK(std::abs(stereo[frame * 2 + 0] - expected_l) < 1e-6f);
            CHECK(std::abs(stereo[frame * 2 + 1] - expected_r) < 1e-6f);
        }
    }
    CHECK(call("sceAudioOut2PortDestroy", a2_port) == 0);
    sink.outs.clear();

    // #1700: a MAIN port wider than 7.1 must reach the host sink. Dragon Quest VII Reimagined
    // declares data_format 0xc00 — 12-channel float — and pushes 2.31 MB/s of real content into it;
    // the old `channels > 8` gate discarded the port, and because it is that context's ONLY port
    // the context never opened a host sink at all. This grain reproduces the measured shape of that
    // title's bed: content in the front pair, and exactly zero in the LFE and height channels.
    a2_port_param.data_format = 0xc00;                    // float32, twelve channels
    CHECK(call("sceAudioOut2PortCreate", a2_context, PTR(&a2_port_param), PTR(&a2_port)) == 0);
    memset(&a2_state, 0xCC, sizeof a2_state);
    CHECK(call("sceAudioOut2PortGetState", a2_port, PTR(&a2_state)) == 0);
    CHECK(a2_state.output == 1 && a2_state.num_channels == 12 && a2_state.volume == 127);
    std::vector<float> a2_wide(64 * 12, 0.0f);
    for (size_t frame = 0; frame < 64; frame++) {
        float* sample = a2_wide.data() + frame * 12;
        sample[0] = 0.30f;   // FL
        sample[1] = 0.10f;   // FR
        sample[2] = 0.08f;   // FC  -> both sides
        sample[6] = 0.04f;   // second surround pair, left
        sample[7] = 0.02f;   // second surround pair, right
        sample[9] = 0.90f;   // height tier: no measured stereo position, must NOT reach the bed
    }
    a2_pcm_ptr = PTR(a2_wide.data());
    CHECK(call("sceAudioOut2PortSetAttributes", a2_port, PTR(&a2_attr), 1) == 0);
    CHECK(call("sceAudioOut2ContextPush", a2_context, 1) == 0);
    CHECK(sink.outs.size() == 1);          // the whole point: not discarded
    if (!sink.outs.empty()) {
        const float* stereo = reinterpret_cast<const float*>(sink.outs[0].pcm.data());
        constexpr float kMinus3Db = 0.70710678f;
        const float expected_l = 0.30f + 0.08f * kMinus3Db + 0.04f * kMinus3Db;
        const float expected_r = 0.10f + 0.08f * kMinus3Db + 0.02f * kMinus3Db;
        for (size_t frame = 0; frame < 64; frame++) {
            CHECK(std::abs(stereo[frame * 2 + 0] - expected_l) < 1e-6f);
            CHECK(std::abs(stereo[frame * 2 + 1] - expected_r) < 1e-6f);
        }
        // The unplaced height channel is loud enough that any accidental contribution — to either
        // side, at any gain above 1e-6 — fails the two assertions above. That is deliberate: the
        // fold must be a stated placement, never an incidental one.
    }
    CHECK(call("sceAudioOut2PortDestroy", a2_port) == 0);
    sink.outs.clear();

    // A channel the fold does not place must never be READ, not merely scaled by zero. `Inf * 0.0f`
    // and `NaN * 0.0f` are both NaN, and the output path clamps NaN to zero — so a mix loop that
    // multiplies every channel by both gains lets one non-finite sample on an unplaced or
    // opposite-side channel silence a side whose own content is perfectly good. An unwritten height
    // tier sitting on recycled guest memory is exactly where such a value comes from.
    const float inf_v = std::numeric_limits<float>::infinity();
    const float nan_v2 = std::nan("");
    a2_port_param.data_format = 0xc00;
    CHECK(call("sceAudioOut2PortCreate", a2_context, PTR(&a2_port_param), PTR(&a2_port)) == 0);
    std::vector<float> a2_poison(64 * 12, 0.0f);
    for (size_t frame = 0; frame < 64; frame++) {
        float* sample = a2_poison.data() + frame * 12;
        sample[0] = 0.25f;      // FL: the value that must survive
        sample[1] = 0.125f;     // FR
        sample[8] = nan_v2;     // unplaced height tier: uninitialized-memory shapes
        sample[9] = inf_v;
        sample[10] = -inf_v;
        sample[11] = nan_v2;
    }
    a2_pcm_ptr = PTR(a2_poison.data());
    CHECK(call("sceAudioOut2PortSetAttributes", a2_port, PTR(&a2_attr), 1) == 0);
    CHECK(call("sceAudioOut2ContextPush", a2_context, 1) == 0);
    CHECK(sink.outs.size() == 1);
    if (!sink.outs.empty()) {
        const float* stereo = reinterpret_cast<const float*>(sink.outs[0].pcm.data());
        for (size_t frame = 0; frame < 64; frame++) {
            CHECK(std::abs(stereo[frame * 2 + 0] - 0.25f) < 1e-6f);
            CHECK(std::abs(stereo[frame * 2 + 1] - 0.125f) < 1e-6f);
        }
    }
    CHECK(call("sceAudioOut2PortDestroy", a2_port) == 0);
    sink.outs.clear();

    // The same rule on the OPPOSITE side of a placed channel: 7.1's ch7 is right-only, so an Inf
    // there must not reach the left side at all. This is the 1..8 path, i.e. behaviour that existed
    // before wide beds and must not have regressed when the fold became a matrix.
    //
    // THIS CASE IS NOT REDUNDANT WITH THE ONE ABOVE — do not delete either as a duplicate. They
    // discriminate different wrong fixes. A fix that skips only channels whose gain pair is {0, 0}
    // passes the 12-channel test (its poison sits in the unplaced tier) and FAILS here, because
    // ch7 is placed — just not on this side. Only a fix that skips a zero gain PER SIDE passes
    // both. Deleting the narrower-looking one is exactly how a discriminating case quietly becomes
    // a vacuous one.
    a2_port_param.data_format = 0x800;
    CHECK(call("sceAudioOut2PortCreate", a2_context, PTR(&a2_port_param), PTR(&a2_port)) == 0);
    std::vector<float> a2_side(64 * 8, 0.0f);
    for (size_t frame = 0; frame < 64; frame++) {
        float* sample = a2_side.data() + frame * 8;
        sample[0] = 0.30f;      // FL
        sample[7] = inf_v;      // second surround pair, RIGHT only
    }
    a2_pcm_ptr = PTR(a2_side.data());
    CHECK(call("sceAudioOut2PortSetAttributes", a2_port, PTR(&a2_attr), 1) == 0);
    CHECK(call("sceAudioOut2ContextPush", a2_context, 1) == 0);
    CHECK(sink.outs.size() == 1);
    if (!sink.outs.empty()) {
        const float* stereo = reinterpret_cast<const float*>(sink.outs[0].pcm.data());
        for (size_t frame = 0; frame < 64; frame++) {
            CHECK(std::abs(stereo[frame * 2 + 0] - 0.30f) < 1e-6f);   // left is untouched
            CHECK(stereo[frame * 2 + 1] == 1.0f);                     // right saturates, as documented
        }
    }
    CHECK(call("sceAudioOut2PortDestroy", a2_port) == 0);
    sink.outs.clear();
    a2_port_param.data_format = 0xc00;

    // What is left unreadable must stay a refusal: a grain prosper cannot size has nothing to fold.
    // Both arms matter. A sample type it cannot decode is the obvious one. A declared width beyond
    // the fold's range is the dangerous one — it must NOT be clamped and read, because reading a
    // 20-channel grain at a 16-channel stride walks the guest's buffer and mixes garbage, which is
    // strictly worse than the silence the old over-8 reject produced.
    // The buffer is sized for the widest declaration below and filled with signal, so a build that
    // wrongly ACCEPTS one of these mixes real, in-bounds data and fails loudly. Sizing it for the
    // declared width matters: if the refusal only held because the accepting build read past the
    // end of a shorter buffer, this test would be proving the allocator's behaviour, not prosper's.
    std::vector<float> a2_wide_src(64 * 20, 0.25f);
    a2_pcm_ptr = PTR(a2_wide_src.data());
    const uint32_t unreadable_formats[] = {
        0xc02,      // twelve channels, unknown sample type 2
        0x1400,     // twenty channels, f32: wider than kAudioMaxBedChannels
    };
    for (uint32_t fmt : unreadable_formats) {
        a2_port_param.data_format = fmt;
        CHECK(call("sceAudioOut2PortCreate", a2_context, PTR(&a2_port_param), PTR(&a2_port)) == 0);
        CHECK(call("sceAudioOut2PortSetAttributes", a2_port, PTR(&a2_attr), 1) == 0);
        CHECK(call("sceAudioOut2ContextPush", a2_context, 1) == 0);
        CHECK(sink.outs.empty());
        CHECK(call("sceAudioOut2PortDestroy", a2_port) == 0);
        sink.outs.clear();
    }
    // The count reported back to the guest is the DECLARED one, not a clamp. This pins prosper's own
    // no-clamp contract — a state query that silently reports 16 for a 20-channel port is the same
    // decoder/consumer disagreement #1700 was — and NOT a claim about what hardware returns for an
    // out-of-range declaration, which no evidence covers (PortCreate validates the field nowhere).
    a2_port_param.data_format = 0x1400;
    CHECK(call("sceAudioOut2PortCreate", a2_context, PTR(&a2_port_param), PTR(&a2_port)) == 0);
    memset(&a2_state, 0xCC, sizeof a2_state);
    CHECK(call("sceAudioOut2PortGetState", a2_port, PTR(&a2_state)) == 0);
    CHECK(a2_state.num_channels == 20);
    CHECK(call("sceAudioOut2PortDestroy", a2_port) == 0);
    sink.outs.clear();


    // Signed-16 AudioOut2 is a first-class port format, not an unsupported codec path. Exercise it
    // on a second simultaneous context: each context must get an independently paced host stream so
    // SDL can mix their grains on the same timeline instead of serializing them. The two-attribute
    // submission shape used by middleware carries channel gains in attribute 1 and the per-grain PCM
    // pointer in attribute 0; the mixer must consume the latter even when it is not first.
    uint64_t a2_context2 = 0;
    CHECK(call("sceAudioOut2ContextCreate", PTR(a2_context_param), 0, 0, PTR(&a2_context2)) == 0);
    CHECK(a2_context2 != 0 && a2_context2 != a2_context);
    a2_port_param.data_format = 0x201;                    // signed-16, two channels
    CHECK(call("sceAudioOut2PortCreate", a2_context2, PTR(&a2_port_param), PTR(&a2_port)) == 0);
    std::vector<int16_t> a2_s16(64 * 2);
    for (size_t i = 0; i < a2_s16.size(); ++i)
        a2_s16[i] = i == 0 ? INT16_MIN : (i == 1 ? INT16_MAX : (int16_t)((int)i * 257 - 16000));
    uint64_t a2_s16_ptr = PTR(a2_s16.data());
    float a2_channel_gains[2] = {1.0f, 1.0f};
    A2Attribute a2_s16_attrs[2] = {
        {1, 0, PTR(a2_channel_gains), sizeof(a2_channel_gains)},
        {0, 0, PTR(&a2_s16_ptr), sizeof(a2_s16_ptr)},
    };
    CHECK(call("sceAudioOut2PortSetAttributes", a2_port, PTR(a2_s16_attrs), 2) == 0);
    CHECK(call("sceAudioOut2ContextPush", a2_context2, 1) == 0);
    CHECK(sink.opens.size() == 2);
    if (sink.opens.size() >= 2) CHECK(sink.opens[1].port == 18);
    CHECK(sink.outs.size() == 1);
    if (!sink.outs.empty()) {
        const float* stereo = reinterpret_cast<const float*>(sink.outs[0].pcm.data());
        CHECK(sink.outs[0].port == 18);
        CHECK(sink.outs[0].pcm.size() == a2_s16.size() * sizeof(float));
        for (size_t i = 0; i < a2_s16.size(); ++i)
            CHECK(std::abs(stereo[i] - (float)a2_s16[i] / 32768.0f) < 1e-7f);
    }
    CHECK(call("sceAudioOut2PortDestroy", a2_port) == 0);
    CHECK(call("sceAudioOut2ContextDestroy", a2_context2) == 0);

    // The half of #1700 that made the title inaudible rather than merely quieter: a context whose
    // ONLY port is a wide bed. `have_pcm` gated the sink open, so discarding that port meant the
    // context never opened a host device at all — no amount of correct folding elsewhere recovers
    // it. Exercise that exact shape on its own fresh context.
    sink.outs.clear();
    uint64_t a2_wide_ctx = 0;
    CHECK(call("sceAudioOut2ContextCreate", PTR(a2_context_param), 0, 0, PTR(&a2_wide_ctx)) == 0);
    CHECK(a2_wide_ctx != 0 && a2_wide_ctx != a2_context);
    const size_t opens_before_wide = sink.opens.size();
    a2_port_param.data_format = 0xc00;
    CHECK(call("sceAudioOut2PortCreate", a2_wide_ctx, PTR(&a2_port_param), PTR(&a2_port)) == 0);
    a2_pcm_ptr = PTR(a2_wide.data());
    CHECK(call("sceAudioOut2PortSetAttributes", a2_port, PTR(&a2_attr), 1) == 0);
    CHECK(call("sceAudioOut2ContextPush", a2_wide_ctx, 1) == 0);
    CHECK(sink.opens.size() == opens_before_wide + 1);      // the sink this context never used to open
    CHECK(sink.outs.size() == 1);
    if (!sink.outs.empty() && sink.opens.size() > opens_before_wide) {
        CHECK(sink.outs[0].port == sink.opens[opens_before_wide].port);
        const float* stereo = reinterpret_cast<const float*>(sink.outs[0].pcm.data());
        CHECK(std::abs(stereo[0]) > 0.1f && std::abs(stereo[1]) > 0.1f);
    }
    CHECK(call("sceAudioOut2PortDestroy", a2_port) == 0);
    CHECK(call("sceAudioOut2ContextDestroy", a2_wide_ctx) == 0);
    sink.outs.clear();
    a2_pcm_ptr = PTR(a2_pcm.data());
    a2_port_param.data_format = 0x200;

    // Object-audio ports share the same context but do not route directly to the speaker sink.
    // GTA V reserves 72 of them; exercise the full SDK maximum so a 16-slot regression cannot
    // turn later guest handles into -1 and crash its object mixer.
    a2_port_param.type = 0x100;
    a2_port_param.data_format = 0x100;                    // float32 mono object stream
    std::vector<uint64_t> a2_object_ports(256);
    for (uint64_t& handle : a2_object_ports) {
        CHECK(call("sceAudioOut2PortCreate", a2_context, PTR(&a2_port_param), PTR(&handle)) == 0);
        CHECK(handle != 0);
    }
    uint64_t overflow_port = 0xCCCCCCCCCCCCCCCCull;
    CHECK((int32_t)call("sceAudioOut2PortCreate", a2_context, PTR(&a2_port_param),
                        PTR(&overflow_port)) == (int32_t)0x80268012);
    CHECK(overflow_port == 0xCCCCCCCCCCCCCCCCull);
    for (uint64_t handle : a2_object_ports)
        CHECK(call("sceAudioOut2PortDestroy", handle) == 0);

    // Context and port handles carry a per-slot generation. Destroying a context invalidates its
    // children, and neither a delayed worker nor a handle from another live context may access a
    // recycled slot. PortRegister/Unregister also enforce the port's owning context.
    a2_port_param.type = 0;
    a2_port_param.data_format = 0x200;
    uint64_t a2_peer_context = 0;
    CHECK(call("sceAudioOut2ContextCreate", PTR(a2_context_param), 0, 0,
               PTR(&a2_peer_context)) == 0);
    uint64_t a2_owned_port = 0;
    CHECK(call("sceAudioOut2PortCreate", a2_context, PTR(&a2_port_param),
               PTR(&a2_owned_port)) == 0);
    CHECK(call("sceAudioOut2PortRegister", a2_context, a2_owned_port) == 0);
    CHECK((int32_t)call("sceAudioOut2PortRegister", a2_peer_context, a2_owned_port) ==
          (int32_t)0x80268001);
    CHECK((int32_t)call("sceAudioOut2PortUnregister", a2_peer_context, a2_owned_port) ==
          (int32_t)0x80268001);
    CHECK(call("sceAudioOut2PortUnregister", a2_context, a2_owned_port) == 0);

    const uint64_t a2_stale_context = a2_context;
    const uint64_t a2_stale_port = a2_owned_port;
    CHECK(call("sceAudioOut2ContextDestroy", a2_stale_context) == 0);
    CHECK((int32_t)call("sceAudioOut2ContextDestroy", a2_stale_context) ==
          (int32_t)0x80268001);
    CHECK((int32_t)call("sceAudioOut2ContextAdvance", a2_stale_context) ==
          (int32_t)0x80268001);
    CHECK((int32_t)call("sceAudioOut2ContextPush", a2_stale_context, 0) ==
          (int32_t)0x80268001);
    CHECK((int32_t)call("sceAudioOut2ContextGetQueueLevel", a2_stale_context, 0, 0) ==
          (int32_t)0x80268001);
    CHECK((int32_t)call("sceAudioOut2ContextSetAttributes", a2_stale_context, 0, 0) ==
          (int32_t)0x80268001);
    CHECK((int32_t)call("sceAudioOut2ContextBedWrite", a2_stale_context, 0, 0) ==
          (int32_t)0x80268001);
    CHECK((int32_t)call("sceAudioOut2PortGetState", a2_stale_port, PTR(&a2_state)) ==
          (int32_t)0x80268001);
    CHECK((int32_t)call("sceAudioOut2PortSetAttributes", a2_stale_port, 0, 0) ==
          (int32_t)0x80268001);
    CHECK((int32_t)call("sceAudioOut2PortDestroy", a2_stale_port) ==
          (int32_t)0x80268001);

    uint64_t a2_recycled_context = 0;
    CHECK(call("sceAudioOut2ContextCreate", PTR(a2_context_param), 0, 0,
               PTR(&a2_recycled_context)) == 0);
    CHECK((a2_recycled_context & 0xffffu) == (a2_stale_context & 0xffffu));
    CHECK(a2_recycled_context != a2_stale_context);
    uint64_t rejected_child = 0xCCCCCCCCCCCCCCCCull;
    CHECK((int32_t)call("sceAudioOut2PortCreate", a2_stale_context, PTR(&a2_port_param),
                        PTR(&rejected_child)) == (int32_t)0x80268001);
    CHECK(rejected_child == 0xCCCCCCCCCCCCCCCCull);

    uint64_t a2_recycled_port = 0;
    CHECK(call("sceAudioOut2PortCreate", a2_recycled_context, PTR(&a2_port_param),
               PTR(&a2_recycled_port)) == 0);
    CHECK((a2_recycled_port & 0xffffu) == (a2_stale_port & 0xffffu));
    CHECK(a2_recycled_port != a2_stale_port);
    CHECK((int32_t)call("sceAudioOut2PortRegister", a2_peer_context, a2_recycled_port) ==
          (int32_t)0x80268001);
    CHECK(call("sceAudioOut2PortRegister", a2_recycled_context, a2_recycled_port) == 0);
    CHECK((int32_t)call("sceAudioOut2PortGetState", a2_stale_port, PTR(&a2_state)) ==
          (int32_t)0x80268001);
    CHECK(call("sceAudioOut2PortGetState", a2_recycled_port, PTR(&a2_state)) == 0);
    CHECK(call("sceAudioOut2PortDestroy", a2_recycled_port) == 0);
    CHECK(call("sceAudioOut2ContextDestroy", a2_recycled_context) == 0);
    CHECK(call("sceAudioOut2ContextDestroy", a2_peer_context) == 0);

    // Host sink slot lifetime is serialized with context teardown. Hold one Push inside output(),
    // fill the other three context slots, and race Destroy: before output is released, Destroy must
    // neither reach sink.close nor free slot 1 for a new-generation context. This fails the old
    // clear-under-state-lock / close-after-unlock ordering deterministically once close is attempted.
    audio_reset();
    BlockingSink blocking_sink;
    audio_set_sink(&blocking_sink);
    uint64_t race_context = 0;
    CHECK(call("sceAudioOut2ContextCreate", PTR(a2_context_param), 0, 0,
               PTR(&race_context)) == 0);
    uint64_t race_port = 0;
    CHECK(call("sceAudioOut2PortCreate", race_context, PTR(&a2_port_param), PTR(&race_port)) == 0);
    a2_pcm_ptr = PTR(a2_pcm.data());
    CHECK(call("sceAudioOut2PortSetAttributes", race_port, PTR(&a2_attr), 1) == 0);
    uint64_t occupied_contexts[3]{};
    for (uint64_t& context : occupied_contexts)
        CHECK(call("sceAudioOut2ContextCreate", PTR(a2_context_param), 0, 0,
                   PTR(&context)) == 0);

    int64_t race_push_result = -1;
    std::thread push_thread([&] {
        race_push_result = call("sceAudioOut2ContextPush", race_context, 1);
    });
    CHECK(blocking_sink.wait_for_output());
    std::atomic<bool> destroy_started{false};
    int64_t race_destroy_result = -1;
    std::thread destroy_thread([&] {
        destroy_started.store(true, std::memory_order_release);
        race_destroy_result = call("sceAudioOut2ContextDestroy", race_context);
    });
    while (!destroy_started.load(std::memory_order_acquire)) std::this_thread::yield();
    CHECK(!blocking_sink.wait_for_close_attempt(std::chrono::milliseconds(250)));
    uint64_t premature_context = 0xCCCCCCCCCCCCCCCCull;
    CHECK((int32_t)call("sceAudioOut2ContextCreate", PTR(a2_context_param), 0, 0,
                        PTR(&premature_context)) == (int32_t)0x80268001);
    CHECK(premature_context == 0xCCCCCCCCCCCCCCCCull);

    blocking_sink.unblock_output();
    push_thread.join();
    destroy_thread.join();
    CHECK(race_push_result == 0);
    CHECK(race_destroy_result == 0);
    CHECK(blocking_sink.completed_in_order());
    uint64_t post_destroy_context = 0;
    CHECK(call("sceAudioOut2ContextCreate", PTR(a2_context_param), 0, 0,
               PTR(&post_destroy_context)) == 0);
    CHECK((post_destroy_context & 0xffffu) == (race_context & 0xffffu));
    CHECK(post_destroy_context != race_context);
    CHECK(call("sceAudioOut2ContextDestroy", post_destroy_context) == 0);
    for (uint64_t context : occupied_contexts)
        CHECK(call("sceAudioOut2ContextDestroy", context) == 0);
    audio_reset();

    // --- 12. libSceNgs2 silent lifecycle: sizes, handles, state, and render output -------------
    struct BufferInfo { uint64_t host_buffer, host_buffer_size, reserved[5], user_data; };
    struct RackOption {
        uint64_t size; char name[16]; uint32_t flags, max_grain, max_voices,
            max_delay, max_matrices, max_ports, reserved[20];
    };
    struct RenderInfo { uint64_t buffer, size; uint32_t waveform_type, channels; };

    BufferInfo sys_info; memset(&sys_info, 0xEE, sizeof sys_info);
    CHECK(call_raw("pgFAiLR5qT4", 0, PTR(&sys_info)) == 0); // SystemQueryBufferSize
    CHECK(sys_info.host_buffer == 0);
    CHECK(sys_info.host_buffer_size == 0x1000);
    CHECK(sys_info.reserved[0] == 0 && sys_info.user_data == 0);
    std::vector<uint8_t> sys_work(sys_info.host_buffer_size, 0);
    sys_info.host_buffer = PTR(sys_work.data());
    uint64_t system = 0xDEADBEEFDEADBEEFull;
    CHECK(call_raw("koBbCMvOKWw", 0, PTR(&sys_info), PTR(&system)) == 0); // SystemCreate
    CHECK(system != 0 && system != 0xDEADBEEFDEADBEEFull);

    BufferInfo fallback_info; memset(&fallback_info, 0xDD, sizeof fallback_info);
    CHECK(call_raw("0eFLVCfWVds", 0x2001, 1, PTR(&fallback_info)) == 0);
    CHECK(fallback_info.host_buffer == 0 && fallback_info.host_buffer_size == 0x2000);

    RackOption rack_opt{}; rack_opt.size = sizeof rack_opt; rack_opt.max_voices = 2;
    memcpy(rack_opt.name, "test", 5);
    BufferInfo rack_info; memset(&rack_info, 0xDD, sizeof rack_info);
    CHECK(call_raw("0eFLVCfWVds", 0x2001, PTR(&rack_opt), PTR(&rack_info)) == 0);
    CHECK(rack_info.host_buffer == 0 && rack_info.host_buffer_size >= 0x1000);
    std::vector<uint8_t> rack_work(rack_info.host_buffer_size, 0);
    rack_info.host_buffer = PTR(rack_work.data());
    uint64_t rack = 0xDEADBEEFDEADBEEFull;
    CHECK(call_raw("cLV4aiT9JpA", system, 0x2001, PTR(&rack_opt), PTR(&rack_info), PTR(&rack)) == 0);
    CHECK(rack != 0 && rack != 0xDEADBEEFDEADBEEFull);

    uint64_t voice = 0xDEADBEEFDEADBEEFull;
    CHECK(call_raw("MwmHz8pAdAo", rack, 1, PTR(&voice)) == 0);
    CHECK(voice != 0 && voice != 0xDEADBEEFDEADBEEFull);
    uint64_t invalid_voice = 0xDEADBEEFDEADBEEFull;
    CHECK((int32_t)call_raw("MwmHz8pAdAo", rack, 2, PTR(&invalid_voice)) == (int32_t)0x804A0302);
    CHECK(invalid_voice == 0xDEADBEEFDEADBEEFull);

    // sceNgs2VoiceGetState fills SceNgs2SamplerVoiceState: state_flags@0x00 (Empty for an un-fed voice),
    // envelope_height@0x04 = 1.0f, and num_decoded_samples@0x10 / decoded_data_size@0x18 = 0 (no
    // playback yet). A streaming guest reads num_decoded_samples from here to pace its feed loop.
    uint8_t voice_state[0x30]; memset(voice_state, 0xCC, sizeof voice_state);
    CHECK(call_raw("-TOuuAQ-buE", voice, PTR(voice_state), sizeof voice_state) == 0);
    CHECK(*(uint32_t*)(voice_state + 0x00) == 0);          // state_flags = Empty (never fed)
    CHECK(*(float*)(voice_state + 0x04) == 1.0f);          // envelope_height = unity
    CHECK(*(uint64_t*)(voice_state + 0x10) == 0);          // num_decoded_samples = 0
    CHECK(*(uint64_t*)(voice_state + 0x18) == 0);          // decoded_data_size  = 0
    for (int i = 0x08; i < 0x10; i++) CHECK(voice_state[i] == 0);   // peak_height + reserved cleared
    for (int i = 0x20; i < 0x30; i++) CHECK(voice_state[i] == 0);   // user_data + waveform_data cleared

    uint8_t ngs_pcm[256]; memset(ngs_pcm, 0xA5, sizeof ngs_pcm);
    RenderInfo render{PTR(ngs_pcm), sizeof ngs_pcm, 0, 2};
    CHECK(call_raw("i0VnXM-C9fc", system, PTR(&render), 1) == 0);
    for (uint8_t b : ngs_pcm) CHECK(b == 0);               // silent backend produces silence
    CHECK((int32_t)call_raw("i0VnXM-C9fc", 0, PTR(&render), 1) == (int32_t)0x804A0230);
    CHECK((int32_t)call_raw("i0VnXM-C9fc", system, 1, 1) == (int32_t)0x804A0053);
    RenderInfo inaccessible_output{1, 16, 0, 2};
    CHECK((int32_t)call_raw("i0VnXM-C9fc", system, PTR(&inaccessible_output), 1) ==
          (int32_t)0x804A0053);

    uint8_t source[0xA8], listener[0xA0], listener_work[0x60];
    memset(source, 0xBB, sizeof source); memset(listener, 0xBB, sizeof listener);
    memset(listener_work, 0xBB, sizeof listener_work);
    CHECK(call_raw("0lbbayqDNoE", PTR(source)) == 0);
    CHECK(call_raw("7Lcfo8SmpsU", PTR(listener)) == 0);
    CHECK(call_raw("1WsleK-MTkE", PTR(listener), PTR(listener_work), 0) == 0);
    for (uint8_t b : source) CHECK(b == 0);
    for (uint8_t b : listener) CHECK(b == 0);
    for (uint8_t b : listener_work) CHECK(b == 0);

    // --- restore the default sink so we don't dangle a stack pointer -------------------------
    audio_reset();

    if (fails) { printf("== FAIL: %d check(s) failed ==\n", fails); return 1; }
    printf("== PASS: sceAudioOut + NGS2 HLE lifecycle/output/error contracts ==\n");
    return 0;
}
