// nid_census — which of a title's imports fall to the dispatcher's return-0 default?
//
// An import prosper does not register never reaches a handler: `prosper_on_unimpl` logs it once and
// returns 0 (src/hle/dispatch/dispatch.cpp). For a contract whose success code is 0 — i.e. most `SCE_OK`
// functions — that 0 IS "success", so the guest is told an operation completed that never ran, and
// any out-parameter the call was supposed to fill is left holding whatever was already there. That
// is strictly worse than an error return: the failure is silent at the call site and surfaces
// arbitrarily far away. Issue #2081 calls this the FALSE SUCCESS class.
//
// The boot log already reports each such NID ("[prosper] unimplemented: ... -> returning 0"), but
// only for the code paths a particular run happened to execute, and only after booting the title.
// This answers the same question STATICALLY and exhaustively: every import in the module, whether
// or not a run reaches it, across as many titles as you point it at.
//
// It is deliberately built on prosper's OWN sources of truth rather than a source grep:
//   * imports come from `Module::load` — the same parser the loader uses, so the census sees the
//     exact NID set the loader will try to bind;
//   * cross-module exports come from `module_export_nids`, the loader's own definition of what a
//     module contributes to the global export table;
//   * registration comes from `Hle::registered` after `register_builtin_hle()` — the real runtime
//     registry, so a NID registered from a table, a loop, or a raw literal is seen identically.
// A grep over `register_fn(...)` call sites would miss every non-literal registration and would
// report those as false-success candidates. Asking the registry cannot.
//
// The cross-module half is not cosmetic. `linker.cpp` pass 2 resolves an import against the global
// export table FIRST — "Cross-module export beats a stub slot" — so an import that another of the
// title's own modules defines never reaches the dispatcher at all, no matter what the HLE registry
// says. A census that only differenced imports against the registry would report every one of those
// as a false-success candidate. Reachability here therefore means "unregistered AND undefined by
// any module shipped with this title", per title, which is exactly the condition under which
// `prosper_on_unimpl` runs.
//
// Usage:
//   nid_census <app0-dir|module> [more...] [--names <PS5-3.20_Libs-dir>]
//              [--registered] [--tsv] [--lib <substr>] [--self-check]
//
// `<app0-dir>` is scanned recursively for eboot.bin and *.prx/*.sprx. Passing several titles ranks
// each NID by how many of them import it, which is the reachability signal #2081 asks for: a NID
// one title imports is a different priority from one that twelve do.
//
// `--names` points at the PS5 3.20 stub dump, whose loader lines carry `<NID> <-> <funcName>`
// pairs directly. `--self-check` re-derives each pair with prosper's own `nid_hash` and reports
// any disagreement: the name table is the instrument this tool reads the census through, so it
// gets a control of its own rather than being trusted.
#include "host/image/boot_program.hpp"
#include "../common/nid_stub_names.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "../../src/loader/linker.hpp"
#include "../../src/self/module.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace prosper;
namespace fs = std::filesystem;

namespace {

struct Row {
    std::string nid;
    std::string name;                 // "" when the stub dump does not name it
    std::set<std::string> libs;       // import library names as the module declares them
    std::set<std::string> titles;     // which inputs import it
    size_t modules = 0;               // how many modules import it
};

// ---- the PS5 3.20 stub dump: authoritative <NID> <-> <funcName> pairs --------------------------
// Each generated library has one loader line per export:
//     if(sprx_dlsym(__handle, "PI7jIZj4pcE", &__ptr_sceRandomGetRandomNumber)) return;
// so the pair is read off directly. No hashing is required to build the map, which is what makes
// --self-check meaningful: the hash is checked AGAINST the dump rather than used to produce it.
//
// The reading of the dump itself lives in tools/common/nid_stub_names.hpp so that self_dump
// --import-slots names its imports from the identical parse; only the --self-check control and its
// mismatch accounting are this tool's.
struct NameTable {
    std::map<std::string, std::string> by_nid;   // nid -> function name
    std::map<std::string, std::string> lib_of;   // nid -> library file stem
    size_t pairs = 0, mismatches = 0;
};

NameTable load_names(const std::string& dir, bool self_check) {
    NameTable t;
    auto stub = prosper_tools::load_stub_names(
        dir, [&](const std::string& nid, const std::string& name) {
            // The dump states the NID; prosper computes it. They must agree, and a disagreement
            // means one of the two is wrong for that name — report it rather than silently
            // preferring either. This is the control on the name table itself.
            if (!self_check || nid_hash(name) == nid) return;
            t.mismatches++;
            // stderr, not stdout: under --tsv these land above the header and corrupt the
            // stream a consumer parses. The control's result belongs with the diagnostics,
            // and its COUNT is reported in the scope block below in both modes.
            fprintf(stderr, "  [name-mismatch] %s: dump says %s, nid_hash() says %s\n",
                    name.c_str(), nid.c_str(), nid_hash(name).c_str());
        });
    t.by_nid = std::move(stub.by_nid);
    t.lib_of = std::move(stub.lib_of);
    t.pairs = stub.pairs;
    return t;
}

// ---- module discovery -------------------------------------------------------------------------
bool is_module_file(const fs::path& p) {
    if (p.filename() == "eboot.bin") return true;
    const std::string ext = p.extension().string();
    return ext == ".prx" || ext == ".sprx";
}

// The module set is now the LOADER'S link set, not a tree scan (#2199).
//
// This tool's central inference is that an import satisfied by a SIBLING module's export never
// reaches the dispatcher, so it is excluded from the census. That is sound only if the sibling set
// matches what the loader actually links. A tree scan is strictly larger -- it picks up .sprx (never
// auto-linked), everything under sce_module/ (the loader takes exactly two named files), plugin
// directories other than Media/Plugins/ (the only one auto-discovered), and modules whose file the
// loader would drop or refuse. Every one of those made the tool exclude a binding that DOES fall to
// the dispatcher's `return 0` at runtime.
//
// The direction of that error is what made it worth fixing: a FALSE ABSENCE. The row is missing
// rather than wrong, so nothing in the output looks suspicious, in a report whose entire purpose is
// to enumerate what reaches the dispatcher.
//
// A single regular file still means "just this module", which is how --lib and single-module runs
// work; only a dump ROOT goes through the loader's set.
std::vector<fs::path> collect_modules(const std::string& input) {
    std::vector<fs::path> out;
    std::error_code ec;
    const fs::path root(input);
    if (fs::is_regular_file(root, ec)) { out.push_back(root); return out; }
    // verbose=false: boot_link_inputs prints the loader's auto-link and case-correction lines, and
    // --tsv writes machine-readable rows to the same stdout.
    for (const auto& li : prosper::boot_link_inputs(input, /*verbose=*/false))
        out.push_back(fs::path(li.path));
    std::sort(out.begin(), out.end());
    return out;
}

// Report the population before filtering and the registration state of the shown subset.
// A module that failed to parse contributes no imports, so its absence must also be explicit.
// `prefix` is "# " for --tsv (comment lines a consumer skips) and "" for the human report.
void print_scope(const char* prefix, size_t total, size_t modules_read, size_t modules_failed,
                 size_t unregistered, size_t shown, size_t shown_unregistered,
                 size_t satisfied_cross_module,
                 size_t mismatches, const std::string& lib_filter, bool self_check) {
    printf("%sscope: %zu distinct imported NIDs over %zu module(s) read, %zu unreadable\n",
           prefix, total, modules_read, modules_failed);
    printf("%sscope: %zu unregistered before filtering, %zu shown (%zu unregistered)%s%s\n",
           prefix, unregistered, shown, shown_unregistered,
           lib_filter.empty() ? "" : ", --lib filter=", lib_filter.c_str());
    printf("%sscope: %zu binding(s) excluded as satisfied by a sibling module's export\n",
           prefix, satisfied_cross_module);
    if (self_check)
        printf("%sscope: name-table self-check %zu mismatch(es)\n", prefix, mismatches);
    if (modules_failed)
        printf("%sWARNING: %zu module(s) did not parse -- their imports are ABSENT from this "
               "census, so a NID missing below may be unmeasured rather than unimported\n",
               prefix, modules_failed);
    // The distinction an absence in this report is most likely to be misread as. Recorded because it
    // already produced a wrong lead: ArcRunner statically imports all three sceAgc*GetSize gaps from
    // #1756 plus sceKernelWaitCommandBufferCompletion — a ready-made explanation for its fault — and
    // calls NONE of them. The runtime unimplemented-call census over a full faulting run was 12 NIDs,
    // none in libSceAgc/libSceAgcDriver/libkernel (#1226).
    printf("%sNOTE: this is a STATIC import census -- what a title MAY call, not what it did. A NID "
           "listed here may never execute, and a fault is not explained by its presence. For what a "
           "run actually called, use prosper_on_unimpl's first-seen census from a live boot, or "
           "hle_calls (#1980), and bound it to the window the behaviour occurs in.\n", prefix);
    printf("%sNOTE: dump roots use the loader's link set (boot_program.cpp); explicit module "
           "inputs inspect that module alone. Cross-module exclusions use the modules "
           "selected for each input.\n",
           prefix);
}

void usage(const char* argv0) {
    fprintf(stderr,
            "usage: %s <app0-dir|module> [more...] [--names <PS5-3.20_Libs-dir>]\n"
            "         [--registered] [--tsv] [--lib <substr>] [--self-check]\n\n"
            "  --names DIR    resolve NIDs through the PS5 3.20 stub dump\n"
            "  --registered   also list imports that DO have a handler (default: only unregistered)\n"
            "  --tsv          machine-readable output\n"
            "  --lib SUBSTR   only report NIDs whose import library contains SUBSTR\n"
            "  --self-check   verify every dump NID against prosper's nid_hash()\n",
            argv0);
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> inputs;
    std::string names_dir, lib_filter;
    bool show_registered = false, tsv = false, self_check = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--names" && i + 1 < argc) names_dir = argv[++i];
        else if (a == "--lib" && i + 1 < argc) lib_filter = argv[++i];
        else if (a == "--registered") show_registered = true;
        else if (a == "--tsv") tsv = true;
        else if (a == "--self-check") self_check = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else if (!a.empty() && a[0] == '-') { usage(argv[0]); return 2; }
        else inputs.push_back(a);
    }
    if (inputs.empty()) { usage(argv[0]); return 2; }

    NameTable names;
    if (!names_dir.empty()) {
        names = load_names(names_dir, self_check);
        if (!tsv)
            printf("[names] %zu NID<->name pairs from %s%s\n", names.pairs, names_dir.c_str(),
                   self_check ? "" : "  (pass --self-check to verify them)");
        if (self_check && !tsv)
            printf("[names] self-check: %zu mismatch(es) against nid_hash()\n", names.mismatches);
    }

    // The registry is the ground truth for "is there a handler?". Populate it exactly the way a
    // boot does, then query it per NID.
    register_builtin_hle();

    std::map<std::string, Row> rows;
    size_t modules_read = 0, modules_failed = 0, satisfied_cross_module = 0;

    for (const auto& input : inputs) {
        const std::string title = fs::path(input).filename().string();
        const auto mods = collect_modules(input);
        if (mods.empty()) fprintf(stderr, "[warn] no modules under %s\n", input.c_str());

        // A title's modules are linked together, so its own exports are the first resolver. Parse
        // every module once, keep them, and take the union of their exports before classifying any
        // import — an import defined by a sibling module is bound to that definition and never
        // reaches the dispatcher, so it is not a candidate here at all.
        std::vector<Module> loaded;
        std::set<std::string> title_exports;
        for (const auto& mp : mods) {
            std::string err;
            auto m = Module::load(mp.string(), &err);
            if (!m) {
                modules_failed++;
                fprintf(stderr, "[warn] %s: %s\n", mp.string().c_str(), err.c_str());
                continue;
            }
            modules_read++;
            for (const auto& nid : module_export_nids(*m)) title_exports.insert(nid);
            loaded.push_back(std::move(*m));
        }

        for (const auto& m : loaded) {
            for (const auto& im : m.imports) {
                if (title_exports.count(im.nid)) { satisfied_cross_module++; continue; }
                Row& r = rows[im.nid];
                r.nid = im.nid;
                if (!im.lib_name.empty()) r.libs.insert(im.lib_name);
                r.titles.insert(title);
                r.modules++;
            }
        }
    }

    // Classify against the live registry.
    std::vector<Row*> selected;
    size_t total = 0, unregistered = 0, shown_unregistered = 0;
    for (auto& [nid, r] : rows) {
        total++;
        const bool registered = Hle::registered(nid);
        if (!registered) unregistered++;
        if (registered && !show_registered) continue;
        if (auto it = names.by_nid.find(nid); it != names.by_nid.end()) r.name = it->second;
        if (!lib_filter.empty()) {
            bool hit = false;
            for (const auto& l : r.libs) if (l.find(lib_filter) != std::string::npos) hit = true;
            if (auto it = names.lib_of.find(nid);
                it != names.lib_of.end() && it->second.find(lib_filter) != std::string::npos) hit = true;
            if (!hit) continue;
        }
        selected.push_back(&r);
        if (!registered) shown_unregistered++;
    }

    // Most-imported first: the count of distinct titles is the reachability rank.
    std::sort(selected.begin(), selected.end(), [](const Row* a, const Row* b) {
        if (a->titles.size() != b->titles.size()) return a->titles.size() > b->titles.size();
        if (a->name != b->name) return a->name < b->name;
        return a->nid < b->nid;
    });

    if (tsv) {
        printf("nid\tname\tregistered\ttitles\tmodules\tlibs\ttitle_list\n");
        for (const Row* r : selected) {
            std::string libs, tl;
            for (const auto& l : r->libs) { if (!libs.empty()) libs += ","; libs += l; }
            for (const auto& t : r->titles) { if (!tl.empty()) tl += ","; tl += t; }
            printf("%s\t%s\t%d\t%zu\t%zu\t%s\t%s\n", r->nid.c_str(),
                   r->name.empty() ? "?" : r->name.c_str(),
                   Hle::registered(r->nid) ? 1 : 0,
                   r->titles.size(), r->modules, libs.c_str(), tl.c_str());
        }
        print_scope("# ", total, modules_read, modules_failed, unregistered, selected.size(),
                    shown_unregistered, satisfied_cross_module, names.mismatches,
                    lib_filter, self_check);
        return 0;
    }

    printf("%s", show_registered
        ? "\n== imports (registered and unregistered) ==\n"
        : "\n== imports with NO registered handler -> dispatcher returns 0 ==\n");
    printf("%-13s %-52s %10s %5s  %s\n",
           "NID", "name", "registered", "#ttl", "import library");
    for (const Row* r : selected) {
        std::string libs;
        for (const auto& l : r->libs) { if (!libs.empty()) libs += ","; libs += l; }
        printf("%-13s %-52s %10s %5zu  %s\n", r->nid.c_str(),
               r->name.empty() ? "?" : r->name.c_str(),
               Hle::registered(r->nid) ? "yes" : "no", r->titles.size(), libs.c_str());
    }
    printf("\n");
    print_scope("", total, modules_read, modules_failed, unregistered, selected.size(),
                shown_unregistered, satisfied_cross_module, names.mismatches,
                lib_filter, self_check);
    return 0;
}
