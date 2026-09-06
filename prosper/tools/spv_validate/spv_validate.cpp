// spv_validate — emit a representative module from EVERY SPIR-V-producing entry point in the tree
// and write each as a .spv to argv[1] (a directory), then run spirv-val on it: the render tests only
// prove llvmpipe *accepts* the modules, but llvmpipe is lenient — strict validation catches latent
// invalid SPIR-V (bad decorations, ill-formed control flow, type mismatches) that would break on a
// real driver. Pure emit + file write; no Vulkan.
//
// This gate has two failure modes of its own, and #1711 is what both of them cost. That defect —
// an OpAccessChain in build_compute_compare_uvec4() whose result type disagreed with the type it
// walked — shipped to real devices through frontends/shared/live/live_compute.cpp and was found by a
// Vulkan validation LAYER, not here, because:
//
//   1. the corpus only ever walked recompile_*, so spirv_builder.cpp's hand-assembled modules were
//      never validated even though they are created at runtime exactly like recompiled shaders; and
//   2. a missing spirv-val was reported as "== PASS (recompiled; spirv-val not found) ==" with exit
//      status 0. spirv-val is on neither the GitHub runner image nor a plain Fedora host, and CI
//      never installed it — so the gate that CLAUDE.md and both READMEs describe as strict had, in
//      CI, validated nothing at all.
//
// Both are closed below: check_emitter_coverage() reads every `src/gpu/*.hpp` and fails on any
// declared emitter this run did not actually emit a validated module from, and an absent spirv-val
// is a hard failure rather than a pass.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/recompiler/spirv_builder.hpp"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <set>
#include <system_error>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>

using namespace prosper::gpu;

static int fails = 0;

static std::string read_text(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    std::string out;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    fclose(f);
    return out;
}

// spirv-val's diagnostic IS the value of this gate, so capture it rather than discarding it: a bare
// "REJECTED it" tells the next reader nothing about which instruction is wrong. Redirecting to a
// file (not /dev/null) also keeps the invocation portable — cmd.exe has no /dev/null, so the old
// probe could never find the validator on Windows even when it was installed.
static bool run_spirv_val(const std::string& path, std::string& message) {
    const std::string log = path + ".val.txt";
    // --target-env vulkan1.1: these modules are consumed by a Vulkan instance, and the universal
    // default environment does not apply the Vulkan-specific rules a driver's validation layer
    // would. Measured over the whole corpus before adopting it: identical results, so it only
    // tightens what can pass here.
    const std::string cmd =
        "spirv-val --target-env vulkan1.1 \"" + path + "\" > \"" + log + "\" 2>&1";
    const int rc = system(cmd.c_str());
    message = read_text(log);
    std::remove(log.c_str());
    return rc == 0;
}

// Emitters this run actually produced a module from. Recorded here, at the point of use, rather
// than inferred from the source text: coverage then means "this emitter ran and its output was
// validated", which is the property the gate is for. A textual check could be satisfied by a
// call-shaped string in a comment, a string literal, or a call whose result is thrown away.
static std::set<std::string> exercised_emitters;

static void dump(const std::string& dir, const char* name, const std::vector<uint32_t>& spv,
                 const char* emitter = nullptr) {
    if (emitter) exercised_emitters.insert(emitter);
    if (spv.empty() || spv[0] != 0x07230203u) { printf("  [FAIL] %-26s did not recompile\n", name); fails++; return; }
    std::string path = dir + "/" + name + ".spv";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { printf("  [FAIL] %-26s cannot write %s\n", name, path.c_str()); fails++; return; }
    fwrite(spv.data(), 4, spv.size(), f); fclose(f);
    std::string message;
    if (!run_spirv_val(path, message)) {
        printf("  [FAIL] %-26s spirv-val REJECTED it:\n", name);
        if (message.empty()) message = "(spirv-val produced no output)\n";
        printf("%s", message.c_str());
        if (message.back() != '\n') printf("\n");
        fails++;
        return;
    }
    printf("  [ok]   %-26s (%zu words) valid\n", name, spv.size());
}

// --- Emitter coverage -------------------------------------------------------------------------
// An entry point that emits SPIR-V but has no module here is invisible to this gate, and nothing
// reports the omission: that is exactly how #1711 survived. So read the emitter declarations out of
// the graphics headers and fail on any that this run did not actually emit a module from.
//
// Coverage is a RUNTIME fact (`exercised_emitters`, recorded inside dump()), not a grep of this
// file. A source-text check would accept a call-shaped string in a comment or a string literal, or
// a call whose result is discarded — and "a check a comment can pass" is the shape of defect this
// file exists to stop.
//
// The list of gaps is EMPTY, and that is the intended state. A gap belongs here only with the issue
// that tracks it, never as a silent omission.
// (#1715 — the generated geometry stage's missing OpExecutionMode Invocations — is deliberately NOT
// a gap: measured, that module passes spirv-val under both the universal and the vulkan1.1
// environments. 00715 is a Vulkan pipeline-creation rule that spirv-val does not check, so it is the
// validation layer's to catch, and the emitter is covered here regardless.)
struct KnownGap { const char* emitter; const char* reason; };
static const std::vector<KnownGap> kKnownGaps = {};

// Names the scan finds that are NOT distinct SPIR-V producers. The default is deliberately
// inverted: an unclassified name fails, so a new emitter cannot be quietly skipped — only a
// deliberate entry here can exempt one, and it has to say why.
struct NotAnEmitter { const char* name; const char* why; };
static const NotAnEmitter kNotEmitters[] = {
    // Two SpirvCompute members became visible to this scan when the recompiler's shared internals
    // moved into rdna2_to_spirv_internal.hpp so the emit functions could be split into their own
    // translation units. Neither is a new code path -- both were always reached through the entry
    // points validated below; they were simply inside a .cpp, and this gate reads headers.
    {"build_interpolation_geometry",
     "the body of recompile_interpolation_geometry, which IS validated here: that entry point is "
     "`SpirvCompute builder; return builder.build_interpolation_geometry(layout, capture_position);` "
     "and nothing else, so every word this gate validates for it is emitted by this member"},
    {"finish",
     "SpirvCompute's module-assembly tail, not an entry point. It is the last call of every emitter "
     "validated here, so it is exercised by all of them; a module it broke would fail spirv-val "
     "under whichever emitter produced it"},
    {"safe_execz_branches_for_test", "returns a transformed RDNA2 instruction stream, not SPIR-V"},
    {"structured_execz_branches_for_test", "returns analyzed RDNA2 branch PCs, not SPIR-V"},
    {"mask_test_branches_for_test",  "returns a transformed RDNA2 instruction stream, not SPIR-V"},
    {"cselect_b64_low_only_pcs_for_test", "returns CFG-proven instruction PCs, not SPIR-V"},
    {"recompile_graphics_shader_cached",
     "caching wrapper; ctest shader_recompile_cache asserts its words are byte-identical to the "
     "direct emitter, which is validated here"},
    {"recompile_compute_shader_cached",
     "caching wrapper; ctest shader_recompile_cache asserts its words are byte-identical to the "
     "direct emitter, which is validated here"},
    {"recompile_graphics_shader_cached_shared",
     "caching wrapper returning shared immutable words; ctest shader_recompile_cache asserts "
     "*shared == recompile_vertex(...), i.e. byte-identical to the direct emitter"},
    {"recompile_vertex_chain_cached_shared",
     "caching wrapper over recompile_vertex_chain, which IS validated here. Note the difference "
     "from its siblings: shader_recompile_cache pins its cache identity and reuse, but does NOT "
     "compare its words against the direct emitter, so this entry rests on the wrapper adding no "
     "emission of its own"},
};

// Every declaration of a function returning SPIR-V words, whitespace-tolerant and with no
// name-prefix filter — both of those were false-PASS directions: a differently named emitter, or one
// written with a leading qualifier or an extra space, would simply not be seen. Struct members and
// parameters of the same type are excluded by requiring the '(' of a parameter list.
//
// BOTH spellings, and that is not cosmetic: gpu_execute.hpp's live draw path declares entry points
// as `SharedShaderWords` (an alias for shared_ptr<const vector<uint32_t>>), so keying only on the
// literal vector type left two of them neither covered NOR classifiable — a producer written in the
// idiom the live path already uses could be added with no failure and no mention, which is exactly
// the #1711 shape one level up.
//
// Note this scan has no comment handling, deliberately. The hazard runs the safe way now: a
// declaration-shaped line inside a header comment invents a phantom REQUIRED emitter and fails
// loudly, where the previous, textual coverage check could be silently SATISFIED by a comment.
// The same fail-loud direction covers a parenthesised local in inline header code
// (`std::vector<uint32_t> words(n);` reads as a declaration of an emitter named `words`). None
// exists today; if you write one and this gate goes red, that is why — use `=` or brace init.
static std::vector<std::string> declared_emitters(const std::string& header_text) {
    static const char* const kReturnTypes[] = {"std::vector<uint32_t>", "SharedShaderWords"};
    std::vector<std::string> names;
    for (const char* ret_type : kReturnTypes) {
        const std::string kRet = ret_type;
        size_t at = 0;
        while ((at = header_text.find(kRet, at)) != std::string::npos) {
            size_t i = at + kRet.size();
            while (i < header_text.size() && std::isspace((unsigned char)header_text[i])) ++i;
            std::string name;
            while (i < header_text.size() &&
                   (std::isalnum((unsigned char)header_text[i]) || header_text[i] == '_'))
                name.push_back(header_text[i++]);
            while (i < header_text.size() && std::isspace((unsigned char)header_text[i])) ++i;
            if (!name.empty() && i < header_text.size() && header_text[i] == '(')
                names.push_back(name);
            at += kRet.size();
        }
    }
    return names;
}

static int check_emitter_coverage(const std::string& src_root) {
    // Every header under these roots, not a hardcoded pair: an emitter declared in a NEW header is
    // precisely the #1711 shape one level up, and a fixed list cannot see it. RECURSIVE, and both
    // the GPU code and the frontend that drives it, so neither a new subdirectory nor a producer
    // that grows in the frontend re-creates that blind spot at the level of LOCATION rather than
    // filename. Nothing outside these roots emits SPIR-V today. The command that SHOWS that is
    // `git grep -l 0x07230203 -- prosper/src prosper/frontends`: three files, of which
    // rdna2_to_spirv.cpp and spirv_builder.cpp emit and shader_resources.cpp only PARSES. Grepping
    // the whole tree instead returns 29 files — test fixtures, gpu_replay reading a module back,
    // vendored imgui — and demonstrates nothing. If that changes, add the root here rather than
    // discovering it the way #1711 was discovered.
    static const char* const kSearchRoots[] = {"/src/gpu", "/frontends/shared"};
    std::vector<std::string> headers;
    for (const char* root : kSearchRoots) {
        const std::string dir = src_root + root;
        std::error_code ec;
        // Advanced explicitly with an error_code: the range-for's operator++ is the THROWING
        // overload, so an error arising mid-iteration would terminate instead of being reported.
        std::filesystem::recursive_directory_iterator it(dir, ec), done;
        for (; !ec && it != done; it.increment(ec))
            if (it->path().extension() == ".hpp") headers.push_back(it->path().string());
        if (ec) {
            printf("  [FAIL] emitter coverage: cannot enumerate %s (%s)\n", dir.c_str(),
                   ec.message().c_str());
            return 1;
        }
    }
    if (headers.empty()) {
        printf("  [FAIL] emitter coverage: no headers found under %s\n", src_root.c_str());
        return 1;
    }
    std::sort(headers.begin(), headers.end());

    int problems = 0;
    std::set<std::string> declared;
    for (const std::string& header : headers)
        for (const std::string& name : declared_emitters(read_text(header))) {
            bool excluded = false;
            for (const NotAnEmitter& n : kNotEmitters) excluded = excluded || name == n.name;
            if (!excluded) declared.insert(name);
        }

    for (const std::string& name : declared) {
        const bool exercised = exercised_emitters.count(name) != 0;
        const KnownGap* gap = nullptr;
        for (const KnownGap& g : kKnownGaps)
            if (name == g.emitter) gap = &g;
        if (exercised && gap) {
            printf("  [FAIL] emitter coverage: %s is now validated, so its known-gap entry is "
                   "stale -- delete it\n", name.c_str());
            ++problems;
        } else if (!exercised && !gap) {
            printf("  [FAIL] emitter coverage: %s emits SPIR-V that this gate never validates.\n"
                   "         Add a module for it, passing \"%s\" as dump()'s emitter argument; or\n"
                   "         record it in kKnownGaps with the issue that tracks it; or, if it does\n"
                   "         not emit SPIR-V at all, name it in kNotEmitters with the reason.\n",
                   name.c_str(), name.c_str());
            ++problems;
        } else if (!exercised) {
            printf("  [gap]  %-26s not validated: %s\n", name.c_str(), gap->reason);
        }
    }
    for (const KnownGap& g : kKnownGaps)
        if (!declared.count(g.emitter)) {
            printf("  [FAIL] emitter coverage: known-gap entry %s no longer names a declared "
                   "emitter -- delete it\n", g.emitter);
            ++problems;
        }
    // An emitter exercised below but absent from every header means the scan stopped seeing it.
    // That is the silent-coverage-loss direction, and it must not read as success.
    for (const std::string& name : exercised_emitters)
        if (!declared.count(name)) {
            printf("  [FAIL] emitter coverage: a module was emitted from %s, but the header scan no "
                   "longer finds it declared -- the scan has stopped working\n", name.c_str());
            ++problems;
        }
    if (!problems) {
        // Count, rather than say "all": a kKnownGaps entry takes the [gap] branch without adding a
        // problem, so "all N were exercised" would be false the moment that list is non-empty.
        size_t gaps = 0;
        for (const KnownGap& g : kKnownGaps) gaps += declared.count(g.emitter);
        printf("  [ok]   emitter coverage: %zu of %zu declared SPIR-V emitters were exercised%s\n",
               declared.size() - gaps, declared.size(),
               fails ? " (one or more of which FAILED above)" : ", and every module validated");
    }
    return problems;
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : ".";
    // The source root is required, not optional: the coverage check is the half of this gate that
    // survives the next emitter being added, and a check that silently skips itself is the defect
    // this tool exists to stop.
    if (argc <= 2) {
        printf("== FAIL: usage: spv_validate <output-dir> <source-root> ==\n"
               "  <source-root> is prosper/ -- the emitter-coverage check reads its headers.\n");
        return 1;
    }
    const std::string src_root = argv[2];

    // Ask the directory directly rather than inferring writability from whether the validator probe
    // left anything behind: a spirv-val that exists but exits non-zero while printing nothing would
    // otherwise be diagnosed as an unwritable directory, sending the reader to the wrong place.
    const std::string probe = dir + "/spv_validate-probe.txt";
    if (FILE* w = fopen(probe.c_str(), "wb")) {
        fclose(w);
    } else {
        printf("== FAIL: cannot write into the output directory %s ==\n"
               "  Every module and the spirv-val probe are written there, so this run cannot\n"
               "  proceed. Check <output-dir>; this is NOT a missing spirv-tools install.\n",
               dir.c_str());
        return 1;
    }
    const bool have_val = (system(("spirv-val --version > \"" + probe + "\" 2>&1").c_str()) == 0);
    std::remove(probe.c_str());
    if (!have_val) {
        // Previously this printed "PASS (recompiled; spirv-val not found)" and exited 0, which is
        // how a gate documented as strict validation ran in CI for its whole life without ever
        // validating a module. Install spirv-tools (Fedora/Ubuntu: spirv-tools; macOS:
        // brew install spirv-tools; MSYS2: mingw-w64-ucrt-x86_64-spirv-tools).
        printf("== FAIL: spirv-val is not on PATH ==\n"
               "  This test IS the strict SPIR-V validation gate; without the validator it proves\n"
               "  only that the emitters returned bytes. Install spirv-tools and re-run.\n");
        return 1;
    }


    // Compute ALU (float chain).
    { const uint32_t c[] = {0x06000300u, 0x10000500u, 0xBF810000u};
      dump(dir, "compute_alu", recompile_valu(c, 3, 3, 0), "recompile_valu"); }
    // GTA V's exact literal-bearing V_ALIGNBYTE_B32 packet.  Strict validation guards the
    // masked-shift lowering: SPIR-V shift operands must stay in the defined 0..31 range.
    { const uint32_t c[] = {0xd54f0006u,0x0415fe80u,0x3024240cu,0xbf810000u};
      dump(dir, "compute_alignbyte", recompile_valu(c, std::size(c), 6, 6), "recompile_valu"); }
    // GTA V exec_cs_413d1bf00 pc458: exact V_LDEXP_F32 production packet. This exercises the
    // integer-domain edge lowering under the strict Vulkan SPIR-V validator, not merely the generic
    // recompile_valu entry point above.
    { const uint32_t c[] = {0xd7620000u, 0x0002030du, 0xbf810000u};
      dump(dir, "compute_ldexp", recompile_valu(c, sizeof(c) / sizeof(c[0]), 14, 0)); }
    // Compute + SMEM constant-buffer load (s_buffer_load_dword; routes to binding 2).
    { const uint32_t c[] = {0xf4000000u, 0xfa000004u, 0x7e000200u, 0xbf810000u};
      dump(dir, "compute_smem", recompile_valu(c, sizeof(c)/4, 1, 0)); }
    // Compute + immediate SMEM x16 descriptor bundle. Both eight-dword halves are independently
    // resolved by their consuming MIMG PCs; the load's one immediate offset is not a shared key.
    { const uint32_t c[] = {
          0xf4100300u, 0xfa000000u,
          0xf0800f08u, 0x01630000u,
          0xf0800f08u, 0x01850000u,
          0xbf810000u};
      ShaderResourceTable rt;
      for (uint32_t i = 0; i < 2; ++i) {
          ShaderResource t{};
          t.cls = ResourceClass::Texture;
          t.format = DataFormat::Float32;
          t.num_components = 4;
          t.binding = 4 + i;
          t.fetch_pc = 2 + i * 2;
          t.img_dim = 1;
          t.width = t.height = 2;
          rt.resources.push_back(t);
      }
      dump(dir, "compute_smem_x16_descriptor_bundle",
           recompile_valu(c, sizeof(c)/4, 2, 0, &rt)); }
    // Compute private spill/fill, including a signed short crossing a dword boundary.
    { const uint32_t c[] = {0x7e0002ffu,0x00008001u,0xdc684003u,0x00000000u,0x7e000280u,
                            0xdc2c4003u,0x00000000u,0x7e000b00u,0xBF810000u};
      dump(dir, "compute_private_spill", recompile_valu(c, sizeof(c)/4, 0, 0)); }
    // Fragment: solid green (EXP MRT0).
    { const uint32_t c[] = {0x7E000280u,0x7E0202F2u,0x7E040280u,0x7E0602F2u,0xF800180Fu,0x03020100u,0xBF810000u};
      dump(dir, "fragment_color", recompile_fragment(c, sizeof(c)/4), "recompile_fragment"); }
    // Fragment: Astro's exact wave64 MBCNT + device-global append allocation shape.
    { const uint32_t c[] = {
          0xD7660007u,0x0001007Fu,0xBEFC0380u,0xD8FA0014u,0x06000000u,
          0xD7650000u,0x00020E7Eu,0x4A140106u,0x36001481u,0x7E000D00u,
          0x7E020280u,0x7E040280u,0x7E0602F2u,0xF800180Fu,0x03020100u,0xBF810000u};
      dump(dir, "fragment_gds_append", recompile_fragment(c, sizeof(c)/4)); }
    // Same allocation with a live implicit-LOD sample: sampled alpha reaches EXP and forces WQM.
    { const uint32_t c[] = {
          0x7E0002FFu,0x3E800000u,0x7E0202FFu,0x3E800000u,
          0xF0800F08u,0x00820000u,0xD7660007u,0x0001007Fu,0xBEFC0380u,
          0xD8FA0014u,0x06000000u,0xD7650000u,0x00020E7Eu,0x4A140106u,
          0x36001481u,0x7E000D00u,0x7E020280u,0x7E040280u,
          0xF800180Fu,0x03020100u,0xBF810000u};
      ShaderResourceTable rt; ShaderResource t{}; t.cls=ResourceClass::Texture;
      t.binding=4; t.img_dim=1; t.width=2; t.height=2; t.sgpr_base=8;
      rt.resources.push_back(t);
      dump(dir, "fragment_gds_wqm", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Consume shares the same wave allocator but subtracts the non-helper population.
    { const uint32_t c[] = {
          0xD7660007u,0x0001007Fu,0xBEFC0380u,0xD8F60014u,0x06000000u,
          0xD7650000u,0x00020E7Eu,0x4A140106u,0x36001481u,0x7E000D00u,
          0x7E020280u,0x7E040280u,0x7E0602F2u,0xF800180Fu,0x03020100u,0xBF810000u};
      dump(dir, "fragment_gds_consume", recompile_fragment(c, sizeof(c)/4)); }
    // Fragment private spill/fill (Function-storage declaration in the graphics shell).
    { const uint32_t c[] = {0xdc704010u,0x00000000u,0x7e000280u,0xdc304010u,0x00000000u,
                            0xf800000fu,0x00000000u,0xBF810000u};
      dump(dir, "fragment_private_spill", recompile_fragment(c, sizeof(c)/4)); }
    // Vertex: fullscreen triangle from gl_VertexIndex (EXP POS0).
    { const uint32_t c[] = {0x36020081u,0x2C040081u,0x7E020D01u,0x7E040D02u,0x7E0A02F6u,0x7E0C02F2u,0x10020B01u,
                            0x08020D01u,0x10040B02u,0x08040D02u,0x7E060280u,0x7E0802F2u,0xF80008CFu,0x04030201u,0xBF810000u};
      dump(dir, "vertex_fullscreen", recompile_vertex(c, sizeof(c)/4), "recompile_vertex");
      PixelInputMapping consumed;
      consumed.valid_mask = 0xffffffffu;
      consumed.consumed_known = true;
      consumed.consumed_mask = 1u;
      dump(dir, "vertex_unwritten_consumed_param",
           recompile_vertex(c, sizeof(c)/4, nullptr, &consumed));
      // Geometry-probe capture variant: gl_Position decorated for transform-feedback readback. Must
      // still pass spirv-val (Xfb capability + execution mode + member Offset/XfbBuffer/XfbStride).
      dump(dir, "vertex_xfb_capture", recompile_vertex(c, sizeof(c)/4, nullptr, nullptr, true)); }
    // Vertex private spill/fill (Function-storage declaration in the graphics shell).
    { const uint32_t c[] = {0xdc704010u,0x00000000u,0x7e000280u,0xdc304010u,0x00000000u,
                            0xf80008cfu,0x00000000u,0xBF810000u};
      dump(dir, "vertex_private_spill", recompile_vertex(c, sizeof(c)/4)); }
    // Vertex fetch (buffer_load_format_xy from a V# in user-data s[8:11]).
    { const uint32_t c[] = {0x7e060280u,0x7e0802f2u,0xe0042000u,0x80020100u,0xf80008cfu,0x04030201u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource vb{}; vb.cls=ResourceClass::VertexBuffer; vb.format=DataFormat::Float32;
      vb.num_components=2; vb.binding=3; vb.stride=8; vb.sgpr_base=8; rt.resources.push_back(vb);
      dump(dir, "vertex_fetch", recompile_vertex(c, sizeof(c)/4, &rt)); }
    // A compute fetch through a bounded, non-uniform descriptor array. This representative is kept in
    // the strict corpus because capability-number mistakes can pass the emitter's word-level tests and
    // even some drivers while spirv-val correctly rejects the module.
    { const uint32_t c[] = {0x7e060280u,0x7e0802f2u,0xe0300000u,0x80020100u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource vb{}; vb.cls=ResourceClass::VertexBuffer; vb.format=DataFormat::Uint32;
      vb.num_components=1; vb.binding=3; vb.stride=4; vb.sgpr_base=8;
      vb.table_index_count=4; vb.table_entry_stride=16; vb.table_index_sgpr=6;
      vb.table_selector_mode=BufferTableSelectorMode::UserSgprIndex;
      for (uint32_t index=0; index<vb.table_index_count; ++index) {
          ShaderBufferTableEntry entry;
          entry.gpu_addr=0x200000u+index*0x1000u; entry.size=16; entry.stride=4;
          entry.vsharp={static_cast<uint32_t>(entry.gpu_addr),
                        static_cast<uint32_t>(entry.gpu_addr>>32u)|(4u<<16u),
                        4u,(20u<<12u)|0xfacu};
          vb.table_entries.push_back(entry);
      }
      rt.resources.push_back(vb);
      ComputeShaderConfig config; config.user_sgprs.resize(12);
      config.local_x=config.local_y=config.local_z=1;
      dump(dir, "compute_fetch_descriptor_array",
           recompile_compute(c, std::size(c), &rt, config)); }
    // Fragment: image_sample a texture (T# in s[8:15]).
    { const uint32_t c[] = {0x7e0002ffu,0x3e800000u,0x7e0202ffu,0x3e800000u,0xf0800f08u,0x00820000u,0xf800000fu,0x03020100u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource t{}; t.cls=ResourceClass::Texture; t.format=DataFormat::Float32;
      t.num_components=4; t.binding=4; t.img_dim=1; t.width=2; t.height=2; t.sgpr_base=8; rt.resources.push_back(t);
      dump(dir, "fragment_texture", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Fragment: Astro Bot R32_UINT image_atomic_swap with GLC return.
    { const uint32_t c[] = {0x7e000280u,0x7e020280u,0x7e1202ffu,0x3f800000u,
                            0xf03c2108u,0x00000900u,0x7e000280u,0x7e020309u,
                            0x7e040280u,0x7e0602f2u,0xf800000fu,0x03020100u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource image{}; image.cls=ResourceClass::StorageImage;
      image.format=DataFormat::Uint32; image.num_components=1; image.binding=4;
      image.img_dim=1; image.width=1; image.height=1; image.depth=1; image.sgpr_base=0;
      rt.resources.push_back(image);
      dump(dir, "fragment_image_atomic", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Fragment: image_sample a 3D texture (3 coords) -> mrt0. (shader_028 pattern)
    { const uint32_t c[] = {0x7E0002FFu,0x3E800000u,0x7E0202FFu,0x3E800000u,0x7E0402FFu,0x3E800000u,
                            0xF0800F10u,0x00400000u,0xF800180Fu,0x03020100u,0xBF810000u};
      ShaderResourceTable rt; ShaderResource t{}; t.cls=ResourceClass::Texture; t.format=DataFormat::Float32;
      t.num_components=4; t.binding=4; t.img_dim=2; t.width=2; t.height=2; t.sgpr_base=0; rt.resources.push_back(t);
      dump(dir, "fragment_texture_3d", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Fragment: COMPR export — v_cvt_pkrtz packs f16x2, exp ... done compr unpacks to vec4. (shader_029)
    { const uint32_t c[] = {0x7E0002F0u,0x7E0202F0u,0x5E000300u,0x5E020300u,0xF8001C0Fu,0x00000100u,0xBF810000u};
      dump(dir, "fragment_compr_export", recompile_fragment(c, sizeof(c)/4, nullptr)); }
    // Vertex with PARAM export (interpolated varying out).
    { const uint32_t c[] = {0x7e140d00u,0x36020081u,0x2c040081u,0x7e020d01u,0x7e040d02u,0x100202f6u,0x100404f6u,
                            0x060202f3u,0x060404f3u,0x7e060280u,0x7e0802f2u,0xf80008cfu,0x04030201u,0xf800020fu,0x0403030au,0xbf810000u};
      dump(dir, "vertex_param", recompile_vertex(c, sizeof(c)/4)); }
    // Fragment with v_interp (interpolated varying in).
    { const uint32_t c[] = {0xc8000000u,0xc8010001u,0x7e020280u,0x7e0402f2u,0xf800080fu,0x02010100u,0xbf810000u};
      dump(dir, "fragment_interp", recompile_fragment(c, sizeof(c)/4)); }
    // Compute LDS (ds_write / s_barrier / ds_read).
    { const uint32_t c[] = {0x7e020f00u,0x34040282u,0x34060281u,0x4a060681u,0xd8340000u,0x00000302u,0xbf8a0000u,
                            0x4c0a02bfu,0x340c0a82u,0xd8d80000u,0x07000006u,0x7e000d07u,0xbf810000u};
      dump(dir, "compute_lds", recompile_valu(c, sizeof(c)/4, 1, 0)); }
    // GTA V exec_cs_413ced900 pc69: exact DS_MIN_F32 fields. Strictly validate the core-SPIR-V
    // compare-exchange loop used in place of an unavailable portable float atomic min.
    { const uint32_t c[] = {0xbefe0481u, 0x7e180280u,
                            0xd9340010u, 0x0000040cu,
                            0xdb7c0000u, 0x0000000cu,
                            0xbefe04c1u, 0x7e000280u,
                            0xd8480000u, 0x00000900u,
                            0xd8480004u, 0x00000a00u,
                            0xd8480008u, 0x00000b00u,
                            0xd84c000cu, 0x00000600u,
                            0xd84c0010u, 0x00000700u,
                            0xd84c0014u, 0x00000800u,
                            0xbefe0481u,                         // pc81 exec = lane zero
                            0x7e080280u,                         // pc82 v4 = byte address zero
                            0xdbfc0000u, 0x00000004u,            // pc83 ds_read_b128 v[0:3], v4
                            0xd9d80010u, 0x04000004u,            // pc85 ds_read_b64 v[4:5], v4
                            0xbf8cc17fu, 0xbf810000u};           // wait; end (no guest barrier)
      dump(dir, "compute_ds_min_f32", recompile_valu(c, std::size(c), 2, 0)); }
    // Compute MUBUF store (buffer_store_format_x).
    { const uint32_t c[] = {0x7e040f00u,0x06060100u,0xe0102000u,0x80020302u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource vb{}; vb.cls=ResourceClass::VertexBuffer; vb.format=DataFormat::Float32;
      vb.num_components=1; vb.binding=3; vb.stride=4; vb.sgpr_base=8; rt.resources.push_back(vb);
      dump(dir, "compute_store", recompile_valu(c, sizeof(c)/4, 1, 0, &rt)); }
    // GTA V's cross-workgroup scan publication shape: distinct descriptor variables alias one guest
    // allocation, a GLC store is released by vscnt(0), and a GLC+DLC load polls through the alias.
    // This representative module strictly validates Aliased/Coherent decorations, per-access
    // Volatile operands, and Device-scope UniformMemory release under SPIR-V 1.3 + Vulkan 1.1.
    { const uint32_t c[] = {0xe0744008u,0x80001100u,0xbf8c3f70u,0xbbfd0000u,
                            0xe0706010u,0x80001211u,0xe030e010u,0x80001e1du,
                            0xbf810000u};
      ShaderResourceTable rt;
      ShaderResource publish{}; publish.cls=ResourceClass::ConstantBuffer;
      publish.format=DataFormat::Uint32; publish.binding=4; publish.gpu_addr=0x1000;
      publish.size=180; publish.stride=20; publish.fetch_pc=0; rt.resources.push_back(publish);
      ShaderResource flag=publish; flag.binding=5; flag.fetch_pc=4; rt.resources.push_back(flag);
      ShaderResource poll=publish; poll.binding=6; poll.fetch_pc=6; rt.resources.push_back(poll);
      ComputeShaderConfig cfg; cfg.local_x=64; cfg.wave_size=64;
      dump(dir, "compute_coherent_alias", recompile_compute(c, sizeof(c)/4, &rt, cfg)); }
    // GTA V's exact qword-atomic resource uses same-binding u32/u64 aliased Block variables. Validate
    // both RMW opcodes strictly: driver acceptance alone does not prove the duplicate binding, u64
    // AccessChain, capability, or atomic result type is legal Vulkan SPIR-V.
    { const uint32_t swap[] = {0xe0302000u,0x80000013u,
                               0xe1402000u,0x80000913u,0xbf810000u};
      const uint32_t bit_or[] = {0xe0302000u,0x80000013u,
                                 0xe1686000u,0x80000913u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource atomic{};
      atomic.cls=ResourceClass::ConstantBuffer; atomic.format=DataFormat::Uint32;
      atomic.num_components=1; atomic.binding=2; atomic.gpu_addr=0x2000;
      atomic.size=200; atomic.stride=8; atomic.sgpr_base=0; atomic.fetch_pc=2;
      atomic.atomic_x2_record_count=25; rt.resources.push_back(atomic);
      ComputeShaderConfig cfg; cfg.local_x=1; cfg.storage_buffer_int64_atomics=true;
      dump(dir, "compute_atomic_swap_x2",
           recompile_compute(swap, std::size(swap), &rt, cfg));
      dump(dir, "compute_atomic_or_x2",
           recompile_compute(bit_or, std::size(bit_or), &rt, cfg)); }
    // Astro Bot exact raw buffer_store_dwordx3 packet.
    { const uint32_t c[] = {0x7e140f00u,0x7e060281u,0x7e080282u,0x7e0a0283u,
                            0xe07c2000u,0x8004030au,0xbf810000u};
      ShaderResourceTable rt; ShaderResource dst{}; dst.cls=ResourceClass::ConstantBuffer;
      dst.format=DataFormat::Uint32; dst.num_components=1; dst.binding=3;
      dst.stride=12; dst.sgpr_base=16; rt.resources.push_back(dst);
      dump(dir, "compute_store_x3", recompile_valu(c, sizeof(c)/4, 1, 0, &rt)); }
    // Astro Bot's exact RTIP 1.1 BVH instruction. The software intersection path is intentionally
    // scalar/SSBO SPIR-V so it does not require Vulkan ray-query capabilities.
    { const uint32_t c[] = {0xf1989f07u,0x00040303u,0x43440d3fu,0x46424140u,0x00004847u,
                            0xbf810000u};
      ShaderResourceTable rt; ShaderResource bvh{}; bvh.cls=ResourceClass::ConstantBuffer;
      bvh.format=DataFormat::Uint32; bvh.num_components=1; bvh.binding=4;
      bvh.size=128; bvh.fetch_pc=0; rt.resources.push_back(bvh);
      ComputeShaderConfig cfg; cfg.local_x=1;
      dump(dir, "compute_bvh_intersect", recompile_compute(c, sizeof(c)/4, &rt, cfg),
           "recompile_compute"); }
    // GTA V's exact program 0x205b5e8600 pc1476 packet uses a 64-byte BVH. This separately covers
    // the constant-false 128-byte-node bound that triangle and FP16-box allocations require.
    { constexpr uint32_t pc = 1476u;
      std::vector<uint32_t> c(pc, 0xbf800000u); // s_nop 0
      const uint32_t tail[] = {0xf1989f07u,0x00060202u,0x28292c23u,0x22262725u,0x00002a24u,
                               0xbf810000u};
      c.insert(c.end(), tail, tail + sizeof(tail)/sizeof(tail[0]));
      ShaderResourceTable rt; ShaderResource bvh{}; bvh.cls=ResourceClass::ConstantBuffer;
      bvh.format=DataFormat::Uint32; bvh.num_components=1; bvh.binding=4;
      bvh.size=64; bvh.fetch_pc=pc; rt.resources.push_back(bvh);
      ComputeShaderConfig cfg; cfg.local_x=1;
      dump(dir, "compute_bvh_intersect_64", recompile_compute(c.data(), c.size(), &rt, cfg)); }
    // Compute EXEC-predicated store (v_cmpx + guard execz + store).
    { const uint32_t c[] = {0x7e040f00u,0x06060100u,0x7e0a0284u,0x7da20b02u,0xbf880002u,0xe0102000u,0x80020302u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource vb{}; vb.cls=ResourceClass::VertexBuffer; vb.format=DataFormat::Float32;
      vb.num_components=1; vb.binding=3; vb.stride=4; vb.sgpr_base=8; rt.resources.push_back(vb);
      dump(dir, "compute_pred_store", recompile_valu(c, sizeof(c)/4, 1, 0, &rt)); }

    // Compute STORAGE images (image_load -> image_store, no sampler): 1D, NSA 3D (split-address coords),
    // and 2D_ARRAY (layer coord). One shared rt: src U# in user-data s[0:7]->binding 4, dst s[8:15]->5.
    { ShaderResourceTable rt;
      ShaderResource s{}; s.cls = ResourceClass::StorageImage; s.binding = 4; s.sgpr_base = 0; rt.resources.push_back(s);
      ShaderResource d{}; d.cls = ResourceClass::StorageImage; d.binding = 5; d.sgpr_base = 8; rt.resources.push_back(d);
      const uint32_t c1d[]  = {0x7E080300u,0xF0000F00u,0x00000004u,0xBF8C3F70u,0xF0200F00u,0x00020004u,0xBF810000u};
      dump(dir, "storage_copy_1d",    recompile_valu(c1d,  sizeof(c1d)/4,  1, 0, &rt));
      const uint32_t cnsa[] = {0xF0000F12u,0x0000000Au,0x00000C0Bu,0xBF8C3F70u,0xF0200F12u,0x0002000Au,0x00000C0Bu,0xBF810000u};
      dump(dir, "storage_nsa_3d",     recompile_valu(cnsa, sizeof(cnsa)/4, 1, 0, &rt));
      const uint32_t carr[] = {0xF0000F28u,0x00000004u,0xBF8C3F70u,0xF0200F28u,0x00020004u,0xBF810000u};
      dump(dir, "storage_arrayed_2d", recompile_valu(carr, sizeof(carr)/4, 1, 0, &rt));
      const uint32_t cms[]  = {0xF0000F30u,0x00000000u,0xBF8C3F70u,0x7E000D00u,0xBF810000u};   // image_load 2D_MSAA (x,y,sample)
      dump(dir, "storage_msaa_2d",    recompile_valu(cms,  sizeof(cms)/4,  1, 0, &rt));
      // image_load 2D_MSAA_ARRAY (dim 7): coords (x,y,layer) + sample — needs Arrayed+MS + ImageMSArray cap.
      const uint32_t cmsa[] = {0xF0000F3Au,0x00000000u,0x00030201u,0xBF8C3F70u,0xBF810000u};
      dump(dir, "storage_msaa_array_2d", recompile_valu(cmsa, sizeof(cmsa)/4, 1, 0, &rt)); }
    // GTA V's audited one-level IMAGE_LOAD_MIP / IMAGE_STORE_MIP subset. Validate both sampled
    // image shapes (including the retained array layer) and the NSA storage write strictly.
    { ShaderResourceTable rt;
      ShaderResource t{}; t.cls=ResourceClass::Texture; t.format=DataFormat::Uint32;
      t.num_components=1; t.binding=4; t.fetch_pc=1; t.img_dim=1;
      t.width=4; t.height=4; t.depth=1; t.size=64; t.proven_zero_mip=true;
      rt.resources.push_back(t);
      const uint32_t c[] = {0x7e040207u,0xf0043f08u,0x00050000u,0xbf810000u};
      dump(dir, "compute_load_mip_zero_2d", recompile_valu(c, sizeof(c)/4, 1, 0, &rt));
      rt.resources[0].img_dim=5; rt.resources[0].depth=2; rt.resources[0].size=128;
      const uint32_t ca[] = {0x7e060206u,0xf0043128u,0x00050000u,0xbf810000u};
      dump(dir, "compute_load_mip_zero_array", recompile_valu(ca, sizeof(ca)/4, 1, 0, &rt)); }
    { ShaderResourceTable rt;
      ShaderResource s{}; s.cls=ResourceClass::StorageImage; s.format=DataFormat::Uint32;
      s.num_components=1; s.binding=4; s.fetch_pc=1; s.img_dim=1;
      s.width=4; s.height=4; s.depth=1; s.size=64; s.proven_zero_mip=true;
      rt.resources.push_back(s);
      const uint32_t c[] = {
          0x7e0a0206u,0xf024310au,0x00030004u,0x00000503u,0xbf810000u};
      dump(dir, "compute_store_mip_zero_2d", recompile_valu(c, sizeof(c)/4, 1, 0, &rt));
      rt.resources[0].format=DataFormat::Uint8; rt.resources[0].num_components=4;
      const uint32_t cf[] = {
          0x7e0c0206u,0xf0243f0au,0x00030005u,0x00000604u,0xbf810000u};
      dump(dir, "compute_store_mip_zero_xyzw_2d",
           recompile_valu(cf, sizeof(cf)/4, 1, 0, &rt)); }
    // Compute that SAMPLES a texture and STORES to a storage image (the bloom/downsample shape, shader 006):
    // exercises the sampled-texture path inside a compute shell (needs vec4<float> declared there).
    { ShaderResourceTable rt;
      ShaderResource t{};  t.cls=ResourceClass::Texture;      t.binding=0; t.sgpr_base=0;  t.img_dim=1; rt.resources.push_back(t);
      ShaderResource s{};  s.cls=ResourceClass::StorageImage; s.binding=1; s.sgpr_base=16; rt.resources.push_back(s);
      const uint32_t c[] = {0x7E0002F0u,0x7E0202F0u,0xF09C0F08u,0x00400400u,0xBF8C3F70u,0xF0200108u,0x00040402u,0xBF810000u};
      dump(dir, "compute_sample_store", recompile_valu(c, sizeof(c)/4, 0, 0, &rt)); }
    // Compute image_get_resinfo on a sampled 3D image (DOLL's volume-initializer bounds query).
    { ShaderResourceTable rt;
      ShaderResource t{}; t.cls=ResourceClass::Texture; t.binding=4; t.sgpr_base=0; t.img_dim=2; rt.resources.push_back(t);
      const uint32_t c[] = {0x7E060280u,0xF0380710u,0x00000003u,0xBF810000u};
      dump(dir, "compute_resinfo_3d", recompile_valu(c, sizeof(c)/4, 0, 0, &rt)); }
    // Compute integer image_load from a UINT8x4 3D texture. The sampled image's scalar type and
    // OpImageFetch result must be uint, matching the explicit v_cvt_f32_ubyte* sequence used by
    // UE4's volumetric-lightmap indirection volume (DOLL producer pc 816).
    { ShaderResourceTable rt;
      ShaderResource t{}; t.cls=ResourceClass::Texture; t.format=DataFormat::Uint8;
      t.num_components=4; t.binding=4; t.sgpr_base=0; t.img_dim=2;
      t.width=32; t.height=32; t.depth=32; rt.resources.push_back(t);
      const uint32_t c[] = {0x7E1E0280u,0x7E200280u,0x7E220280u,
                            0xF0000F10u,0x0000000Fu,0xBF8C3F70u,
                            0x7E000D00u,0x7E020D01u,0x7E040D02u,0x7E060D03u,
                            0xBF810000u};
      dump(dir, "compute_uint_load_3d", recompile_valu(c, sizeof(c)/4, 0, 0, &rt)); }
    // Compute mul_hi (high 32 bits via OpUMulExtended -> {lo,hi} struct extract).
    { const uint32_t c[] = {0x7E0202FFu,0x80000000u,0xD56A0002u,0x00020301u,0x7E000D02u,0xBF810000u};
      dump(dir, "compute_mul_hi", recompile_valu(c, sizeof(c)/4, 1, 0)); }
    // Compute: saved VOPC mask combined with VCC by s_nor_b64 (UE4 visibility-mask shape).
    { const uint32_t c[] = {0x7C040CF9u,0x06869880u,0x8DEA6A18u,0xBF810000u};
      dump(dir, "compute_mask_nor", recompile_valu(c, sizeof(c)/4, 0, 0)); }
    // Compute: nested/multi-branch CFG dispatcher (varying VCC exit + inner SCC branch + back-edge).
    // Locks both structured switch-loop formation and the workgroup-scratch hardware-wave vote.
    { const uint32_t c[] = {0xBE800380u,0x7E000280u,0x7E020300u,
                            0xD7610013u,0x00014A7Eu,0xD7610013u,0x0001507Fu,
                            0xD760000Eu,0x00014B13u,0xD760000Fu,0x00015113u,0xBEFE040Eu,
                            0xE00C2000u,0x80020400u,0x7DB900F9u,0x86050007u,
                            0x7D020200u,0xBF860006u,0xBF0A8204u,0x360000FDu,0xBF840001u,
                            0x81008100u,0x81008100u,0xBF82FFF4u,
                            0xBF810000u};
      ShaderResourceTable rt; ShaderResource vb{}; vb.cls=ResourceClass::VertexBuffer;
      vb.binding=3; vb.sgpr_base=8; vb.stride=16; vb.format=DataFormat::Float32;
      vb.num_components=4; rt.resources.push_back(vb);
      dump(dir, "compute_cfg_dispatch", recompile_valu(c, sizeof(c)/4, 0, 0, &rt)); }
    // GTA V Wave64 survivor-mask join: one arm retains scalar EXEC words while the other computes
    // the same physical pair through S_ANDN2_B64. Validate the native subgroup ballots that make
    // the logical result scalar-readable at the exact trailing S_CMP_EQ_U64.
    { const uint32_t c[] = {
          0xBEB8037Eu,0xBEB9037Fu,0x7E400280u,0xBF068008u,0xBF840003u,
          0x7D8A40C1u,0x8AB86A38u,0xBF800000u,0xBF128038u,
          0xBE800380u,0x7E000280u,0x7E020300u,
          0xD7610013u,0x00014A7Eu,0xD7610013u,0x0001507Fu,
          0xD760000Eu,0x00014B13u,0xD760000Fu,0x00015113u,0xBEFE040Eu,
          0xE00C2000u,0x80020400u,0x7DB900F9u,0x86050007u,
          0x7D020200u,0xBF860006u,0xBF0A8204u,0x360000FDu,0xBF840001u,
          0x81008100u,0x81008100u,0xBF82FFF4u,0xBF810000u};
      ShaderResourceTable rt; ShaderResource vb{}; vb.cls=ResourceClass::VertexBuffer;
      vb.binding=3; vb.sgpr_base=8; vb.stride=16; vb.format=DataFormat::Float32;
      vb.num_components=4; rt.resources.push_back(vb);
      ComputeShaderConfig cfg; cfg.local_x=64; cfg.wave_size=64;
      cfg.native_subgroup_size=64;
      dump(dir, "compute_wave64_logical_ballot",
           recompile_compute(c, sizeof(c)/4, &rt, cfg)); }
    // Ordinary e64 integer-pair comparisons (unsigned then signed), separate from mask provenance.
    { const uint32_t c[] = {0xd4e4006au,0x00010000u,0xd4a4006au,0x00010000u,
                            0xbf810000u};
      ComputeShaderConfig cfg; cfg.local_x=128; cfg.wave_size=64;
      dump(dir, "compute_i64_compare",
           recompile_compute(c, sizeof(c)/4, nullptr, cfg)); }
    // Astro Bot Wave64: e64 mask-vs-zero normalization followed by a last-live B64 mask SCC vote.
    // Both reductions use the portable dispatcher's uniform common phase.
    { const uint32_t c[] = {0x7d840000u,0xd4e4006au,0x0001006au,0xbea0047eu,
                            0x87ea6a20u,0x7e000280u,0xbf840003u,0x7d840100u,
                            0x02020100u,0xbf860002u,0xbf060000u,0xbf850001u,
                            0x7e020280u,0xbf810000u};
      ComputeShaderConfig cfg; cfg.local_x=128; cfg.wave_size=64;
      dump(dir, "compute_wave64_mask_vote",
           recompile_compute(c, sizeof(c)/4, nullptr, cfg)); }

    // --- #273 additions (DOLL recompiler frontier) ---
    // Fragment: 3D image_load (integer LUT fetch through the combined sampler; OpImage+OpImageFetch).
    { const uint32_t c[] = {0x7e000280u,0x7e020280u,0x7e040280u,0xf0001f10u,0x00020000u,
                            0xf800000fu,0x03020100u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource t{}; t.cls=ResourceClass::Texture; t.binding=4; t.img_dim=2;
      t.width=16; t.height=16; t.sgpr_base=8; rt.resources.push_back(t);
      dump(dir, "fragment_load_3d", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Fragment: image_sample_lz_o (packed texel offset folded into normalized coords; ImageQuery).
    { const uint32_t c[] = {0x7e0002ffu,0x00000101u,0x7e0202f0u,0x7e0402f0u,0xf0dc0f08u,0x00820000u,
                            0xf800000fu,0x03020100u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource t{}; t.cls=ResourceClass::Texture; t.binding=4; t.img_dim=1;
      t.width=2; t.height=2; t.sgpr_base=8; rt.resources.push_back(t);
      dump(dir, "fragment_sample_lz_o", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Fragment: image_gather4_lz_o (dynamic gather offset; ImageGatherExtended — locks the operand-ID fix).
    { const uint32_t c[] = {0x7e0002ffu,0x00000101u,0x7e0202f0u,0x7e0402f0u,0xf15c0808u,0x00820400u,
                            0xf800000fu,0x07060504u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource t{}; t.cls=ResourceClass::Texture; t.binding=4; t.img_dim=1;
      t.width=2; t.height=2; t.sgpr_base=8; rt.resources.push_back(t);
      dump(dir, "fragment_gather4_lz_o", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Compute: if/else-if/else cascade with common-merge s_branch arms (kernel T17's stream).
    { const uint32_t c[] = {0x7e020f00u,0x7e080501u,0xbf0a8204u,0xbf840002u,0x4a02028au,0xbf820005u,
                            0xbf0a8504u,0xbf840002u,0x4a020294u,0xbf820001u,0x4a02029eu,0x7e000d01u,0xbf810000u};
      dump(dir, "compute_cascade_ifelse", recompile_valu(c, sizeof(c)/4, 1, 0)); }
    // Compute: readfirstlane waterfall + v_movrels (kernel T19's stream).
    { const uint32_t c[] = {0x7e020f00u,0xbe86047eu,0x7e0402ffu,0x40a00000u,0x7e0602ffu,0x40e00000u,
                            0x7e0802ffu,0x41100000u,0x7e080501u,0x7da40204u,0xbefc0304u,0x7e0a8702u,
                            0x8a867e06u,0xbefe0406u,0xbf85fff9u,0xbefe04c1u,0x7e000305u,0xbf810000u};
      dump(dir, "compute_waterfall_movrels", recompile_valu(c, sizeof(c)/4, 1, 0)); }
    // Fragment: divergent execz region (v_cmpx + s_cbranch_execz over a scalar-writing block).
    { const uint32_t c[] = {0x7e020280u,0x7e0002f2u,0x7c2200f0u,0xbf880002u,0xbe8503f2u,0x7e020205u,
                            0x7e040205u,0xf800180fu,0x01010101u,0xbf810000u};
      dump(dir, "fragment_execz_if", recompile_fragment(c, sizeof(c)/4, nullptr)); }
    // A saved lane mask first defined inside a forward if and consumed after its merge. The skipped
    // edge contributes false; without that phi the arm-local definition does not dominate its use.
    { const uint32_t c[] = {0x7E020280u,0x7E0002F2u,0x7C2200F0u,0xBF880001u,0xBE82047Eu,
                            0xBEFE0402u,0x7E0202F2u,0xF800180Fu,0x01010101u,0xBF810000u};
      dump(dir, "fragment_if_new_mask", recompile_fragment(c, sizeof(c)/4, nullptr)); }
    // Fragment: DIVERGENT execz-exit loop with a nested execz if in the body (#273 — the DOLL
    // title post-process accumulation shape; test_recompiled_fragment executes it).
    { const uint32_t c[] = {0x7E020284u,0xBE800380u,0x7E040280u,0x7E060280u,0x7E080282u,0x7E0A02F2u,
                            0xBE82047Eu,0x7DA20200u,0xBF88000Au,0xBE84047Eu,0x7DA20800u,0xBF880002u,
                            0x060404FFu,0x3E800000u,0xBEFE0404u,0x060606FFu,0x3E800000u,0x81008100u,
                            0xBF82FFF4u,0xBEFE0402u,0xF800180Fu,0x05020302u,0xBF810000u};
      dump(dir, "fragment_divergent_loop", recompile_fragment(c, sizeof(c)/4, nullptr)); }
    // Fragment: uniform VCCZ-exit light-accumulation loop (#615, Dead Cells). The compare's VGPR
    // bound is moved from an inline uniform value, making the wave-empty VCC branch structurizable.
    { const uint32_t c[] = {0xBE800380u,0x7E000280u,0x7E020284u,0x7E0602F2u,0x7D020200u,
                            0xBF860004u,0x060000FFu,0x3E800000u,0x81008100u,0xBF82FFFAu,
                            0x7E020300u,0x7E040300u,0xF800080Fu,0x03020100u,0xBF810000u};
      dump(dir, "fragment_uniform_vcc_loop", recompile_fragment(c, sizeof(c)/4, nullptr)); }
    // Fragment: EXECNZ-back-edge loop with a mid-body vccz break (#273 — the scalar-indexed unroll).
    { const uint32_t c[] = {0xBE82047Eu,0xBE800380u,0xBE810383u,0x7E040280u,0x7E0A02F2u,0xBF880009u,
                            0xBF0A0100u,0x8584807Eu,0xBEEA0404u,0xBEFE0404u,0xBF860004u,0x060404FFu,
                            0x3E800000u,0x81008100u,0xBF89FFF6u,0xBEFE0402u,0xF800180Fu,0x05020202u,
                            0xBF810000u};
      dump(dir, "fragment_execnz_loop", recompile_fragment(c, sizeof(c)/4, nullptr)); }
    // Fragment: v_writelane/v_readlane scalar-spill slots (#273).
    { const uint32_t c[] = {0xBE8503FFu,0x3E800000u,0xBE8603F2u,0xD761000Au,0x00010605u,0xD761000Au,
                            0x00010E06u,0xD7600007u,0x0001070Au,0xD7600008u,0x00010F0Au,0x7E000207u,
                            0x7E020208u,0xF800180Fu,0x01000100u,0xBF810000u};
      dump(dir, "fragment_lane_slots", recompile_fragment(c, sizeof(c)/4, nullptr)); }
    // NESTED divergent execz-exit loops (#590/#1067 — DOLL's last post-process kernel shape): an
    // inner table loop entirely inside an outer row loop, execz exits + backward s_branch
    // back-edges, an exec save around the inner loop, post-loop s_barrier (compute). Locks the
    // nested OpLoopMerge emission under strict validation for BOTH shells; the execution twins
    // live in test_rdna2_to_spirv / test_recompiled_fragment.
    { const uint32_t c[] = {0x7E020283u,0x7E080284u,0xBE800380u,0x7E040280u,0x7E060280u,0x7E0A02F2u,
                            0xBE82047Eu,0x7DA20200u,0xBF88000Du,0xBE810380u,0xBE84047Eu,0x7DA20801u,
                            0xBF880004u,0x060606FFu,0x3D800000u,0x81018101u,0xBF82FFFAu,0xBEFE0404u,
                            0x060404FFu,0x3E800000u,0x81008100u,0xBF82FFF1u,0xBEFE0402u,0xBF8A0000u,
                            0xBF810000u};
      dump(dir, "compute_nested_exec_loops", recompile_valu(c, sizeof(c)/4, 1, 3)); }
    { const uint32_t c[] = {0x7E020283u,0x7E080284u,0xBE800380u,0x7E040280u,0x7E060280u,0x7E0A02F2u,
                            0xBE82047Eu,0x7DA20200u,0xBF88000Du,0xBE810380u,0xBE84047Eu,0x7DA20801u,
                            0xBF880004u,0x060606FFu,0x3D800000u,0x81018101u,0xBF82FFFAu,0xBEFE0404u,
                            0x060404FFu,0x3E800000u,0x81008100u,0xBF82FFF1u,0xBEFE0402u,
                            0xF800180Fu,0x05020302u,0xBF810000u};
      dump(dir, "fragment_nested_exec_loops", recompile_fragment(c, sizeof(c)/4, nullptr)); }

    // --- Emitters that are NOT the RDNA2 recompiler ---
    // spirv_builder.cpp hand-assembles compute modules that the live frontend creates at runtime
    // (frontends/shared/live/live_compute.cpp: prepare_compare_pipeline / the scale-bias probe), so they
    // reach vkCreateShaderModule exactly like a recompiled shader. #1711 lived here.
    dump(dir, "builder_scale_bias", build_compute_scale_bias(2.0f, 0.5f),
         "build_compute_scale_bias");
    dump(dir, "builder_compare_uvec4", build_compute_compare_uvec4(),
         "build_compute_compare_uvec4");

    // --- Recompiler entry points beyond the four main stage functions ---
    // Wave32 fragment lowering (s_wqm_b32 through the low-half EXEC/VCC mask path).
    { const uint32_t c[] = {0xbe80037eu,0xbefe0900u,0xbefe097eu,
                            0x7e000280u,0x7e020280u,0x7e040280u,0x7e0602f2u,
                            0xf800000fu,0x03020100u,0xbf810000u};
      dump(dir, "fragment_wave32_wqm", recompile_fragment_wave32_for_test(c, sizeof(c)/4),
           "recompile_fragment_wave32_for_test"); }
    // Terminal NGG output gate: CMPX + EXECZ around the POS export.
    { const uint32_t c[] = {0xBF900009u,0x34040A81u,0x36060AC2u,0x7E000280u,0x7E0202F2u,
                            0x36040482u,0x4A0606C1u,0x4A0404C1u,0x7E060B03u,0x7E040B02u,
                            0x7E280281u,0x7E2A0280u,0x7C3E2B14u,0xBF880002u,
                            0xF80008CFu,0x01000302u,0xBF810000u};
      dump(dir, "vertex_ngg_terminal_gate",
           recompile_vertex_terminal_ngg_gate_for_test(c, sizeof(c)/4),
           "recompile_vertex_terminal_ngg_gate_for_test"); }
    // One-lane NGG projection: a B64 mask scan reduced against the single live guest lane.
    { const uint32_t c[] = {0xBF900009u,0xBEEA04C1u,0xBE80146Au,0x7E000C00u,
                            0x7E020280u,0x7E040280u,0x7E0602F2u,
                            0xF80008CFu,0x03020100u,0xBF810000u};
      dump(dir, "vertex_ngg_one_lane",
           recompile_vertex_ngg_one_lane_for_test(c, sizeof(c)/4),
           "recompile_vertex_ngg_one_lane_for_test"); }
    // Split no-GS NGG program: a separately installed producer plus its terminal wrapper, recompiled
    // as one register-preserving module (the chain path the loader installs for real Astro draws).
    { const uint32_t producer[] = {
          0xD765000Au,0x000100C1u,0xD766000Au,0x000214C1u,
          0x34040A81u,0x36060AC2u,0x7E000280u,0x7E0202F2u,
          0x36040482u,0x4A0606C1u,0x4A0404C1u,0x7E060B03u,0x7E040B02u,
          0x7E0C0280u,0x7E0E0280u,0x7E1002F2u,
          0xD8380100u,0x00080706u,0xD8380302u,0x00030206u,0xD8380504u,0x00010006u,
          0x7E12030Au,0xD8340018u,0x00000906u,0xBF8A0000u,0xBE802006u};
      const uint32_t wrapper[] = {
          0xBF900009u,0xF8000941u,0x00000000u,
          0xD5430000u,0x03FE249Cu,0x00000728u,0xD5430002u,0x03FE249Cu,0x00000730u,
          0xD8DC0100u,0x00000000u,0xD8DC0100u,0x02000002u,0xF80000CFu,0x03020100u,
          0xD5430000u,0x03FE249Cu,0x00000720u,0xD8DC0100u,0x00000000u,
          0xF8000203u,0x00000100u,0x1608249Cu,0xD8D80738u,0x04000004u,
          0xF8000211u,0x00000004u,0xBF810000u};
      ShaderResourceTable rt; rt.vertices_per_instance = 3;
      dump(dir, "vertex_ngg_chain",
           recompile_vertex_chain(producer, sizeof(producer)/4, wrapper, sizeof(wrapper)/4,
                                  &rt, nullptr, false, 7),
           "recompile_vertex_chain"); }
    // Generated interpolation geometry stage: AMD's explicit-parameter form publishes P0/P10/P20
    // plus perspective-center I/J from a synthesised Geometry entry point.
    { const uint32_t ps[] = {0xc80e0000u,0xc8120001u,0xc8160002u,
                             0xd54b0003u,0x04160103u,0xd54b0003u,0x040e0304u,
                             0x7e080280u,0x7e0a0280u,0x7e0c02f2u,
                             0xf800000fu,0x06050403u,0xbf810000u};
      PixelSystemInputMapping perspective_center{1u << 1, 1u << 1};
      const FragmentInterpolationLayout layout =
          fragment_interpolation_layout(ps, sizeof(ps)/4, &perspective_center);
      dump(dir, "geometry_interpolation", recompile_interpolation_geometry(layout),
           "recompile_interpolation_geometry"); }

    fails += check_emitter_coverage(src_root);

    if (fails) { printf("== FAIL: %d emitter(s) failed emission/validation/coverage ==\n", fails); return 1; }
    printf("== PASS (every declared emitter accounted for; all modules pass spirv-val) ==\n");
    return 0;
}
