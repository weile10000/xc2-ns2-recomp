// SPDX-License-Identifier: GPL-3.0-or-later
#include "xc2_nav_mesh.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nxrt/log.h"

// Use the repo's vendored MIT miniz (inflate only, zero extra build targets).
#define MINIZ_NO_STDIO
#define MINIZ_NO_ARCHIVE_APIS
#define MINIZ_NO_ARCHIVE_WRITING_APIS
#define MINIZ_NO_ZLIB_COMPATIBLE_NAMES
#define MINIZ_NO_DEFLATE_APIS
#include "../../third_party/nxrecomp/tools/setup/third_party/miniz/miniz.c"

namespace xc2::nav {
namespace {

constexpr const char* kTag = "nav.mesh";

inline uint16_t rd_u16(const uint8_t* p) {
    uint16_t v = 0;
    std::memcpy(&v, p, 2);
    return v;
}
inline uint32_t rd_u32(const uint8_t* p) {
    uint32_t v = 0;
    std::memcpy(&v, p, 4);
    return v;
}
inline int32_t rd_i32(const uint8_t* p) {
    int32_t v = 0;
    std::memcpy(&v, p, 4);
    return v;
}
inline uint64_t rd_u64(const uint8_t* p) {
    uint64_t v = 0;
    std::memcpy(&v, p, 8);
    return v;
}
inline float rd_f32(const uint8_t* p) {
    float v = 0.f;
    std::memcpy(&v, p, 4);
    return v;
}

bool xbc1_decompress(const uint8_t* src, size_t src_len, std::vector<uint8_t>& out) {
    if (src_len < 0x30 || std::memcmp(src, "xbc1", 4) != 0) return false;
    const uint32_t unc_sz = rd_u32(src + 0x08);
    const uint32_t comp_sz = rd_u32(src + 0x0C);
    if (unc_sz == 0 || unc_sz > 64u * 1024u * 1024u) return false;
    if (0x30ull + comp_sz > src_len) return false;
    out.resize(unc_sz);
    const size_t got = tinfl_decompress_mem_to_mem(
        out.data(), unc_sz, src + 0x30, comp_sz, TINFL_FLAG_PARSE_ZLIB_HEADER);
    if (got == TINFL_DECOMPRESS_MEM_TO_MEM_FAILED || got != unc_sz) {
        out.clear();
        return false;
    }
    return true;
}

// ---- Minimal bf2.arh ('arh1') trie decoder (read-only, cached once per session) ----
struct ArhEntry {
    uint64_t offset = 0;
    uint32_t comp = 0;
    uint32_t unc = 0;
    uint32_t flags = 0;
};

struct ArhIndex {
    bool loaded = false;
    std::filesystem::path ard_path;
    std::unordered_map<std::string, ArhEntry> by_path;  // lower-case normalized path ("/map/ma05a.winvhe")
};

std::string to_lower_ascii(std::string s) {
    for (char& ch : s) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return s;
}

bool load_arh_index(const std::filesystem::path& romfs_dir, ArhIndex& out) {
    out.loaded = true;
    const auto arh_path = romfs_dir / "bf2.arh";
    out.ard_path = romfs_dir / "bf2.ard";
    std::ifstream f(arh_path, std::ios::binary);
    if (!f) {
        NXLOG_W(kTag, "bf2.arh not found under %s", romfs_dir.string().c_str());
        return false;
    }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.size() < 0x28 || std::memcmp(buf.data(), "arh1", 4) != 0) return false;

    const uint32_t str_off  = rd_u32(buf.data() + 0x0C);
    const uint32_t str_sz   = rd_u32(buf.data() + 0x10);
    const uint32_t trie_off = rd_u32(buf.data() + 0x14);
    const uint32_t trie_sz  = rd_u32(buf.data() + 0x18);
    const uint32_t ent_off  = rd_u32(buf.data() + 0x1C);
    const uint32_t ent_cnt  = rd_u32(buf.data() + 0x20);
    const uint32_t key      = rd_u32(buf.data() + 0x24);

    if (str_off < 0x28 || str_sz < 4 || trie_off < 0x28 || trie_sz < 8 ||
        str_off + str_sz > buf.size() || trie_off + trie_sz > buf.size() ||
        ent_off + static_cast<uint64_t>(ent_cnt) * 24ull > buf.size()) {
        return false;
    }

    constexpr uint32_t kTrieXor = 0x0C0CACACu;
    auto xor4 = [](std::vector<uint8_t>& d, uint32_t k32) {
        const uint8_t kb[4] = {
            static_cast<uint8_t>(k32 & 0xFF),
            static_cast<uint8_t>((k32 >> 8) & 0xFF),
            static_cast<uint8_t>((k32 >> 16) & 0xFF),
            static_cast<uint8_t>((k32 >> 24) & 0xFF),
        };
        for (size_t i = 0; i < d.size(); ++i) d[i] ^= kb[i & 3];
    };

    std::vector<uint8_t> dec_str(buf.begin() + str_off, buf.begin() + str_off + str_sz);
    std::vector<uint8_t> dec_trie(buf.begin() + trie_off, buf.begin() + trie_off + trie_sz);
    xor4(dec_str, key ^ kTrieXor ^ 0xFFFFFFFFu);
    xor4(dec_trie, key ^ kTrieXor);

    const size_t trie_n = dec_trie.size() / 8;
    auto trie_a = [&](size_t i) -> int32_t { return rd_i32(dec_trie.data() + i * 8); };
    auto trie_b = [&](size_t i) -> int32_t { return rd_i32(dec_trie.data() + i * 8 + 4); };
    const uint8_t* ents = buf.data() + ent_off;

    for (size_t node = 1; node < trie_n; ++node) {
        const int32_t a = trie_a(node);
        const int32_t b = trie_b(node);
        if (b >= 0) continue;
        const int32_t par = -b - 1;
        if (par < 0 || static_cast<size_t>(par) >= trie_n) continue;
        const int32_t pa = trie_a(static_cast<size_t>(par));
        if (pa >= 0) continue;
        const int32_t edge = (-pa - 1) ^ static_cast<int32_t>(node);
        if (edge < 0 || edge >= 256) continue;

        std::string tail;
        uint32_t entry_idx = UINT32_MAX;
        if (a > 0) {
            size_t nul = static_cast<size_t>(a) + 1;
            while (nul < dec_str.size() && dec_str[nul] != 0) ++nul;
            if (nul + 5 > dec_str.size()) continue;
            tail.assign(reinterpret_cast<const char*>(dec_str.data() + a + 1), nul - (static_cast<size_t>(a) + 1));
            entry_idx = rd_u32(dec_str.data() + nul + 1);
        } else if (edge == 0 && a < 0) {
            entry_idx = static_cast<uint32_t>(-a - 1);
        } else {
            continue;
        }
        if (entry_idx >= ent_cnt) continue;

        // Walk parent chain back to root (note: .winvhe / .winvda share prefix ".winv" in trie, so check full path after walking)
        std::string rev;
        size_t cur = node;
        bool ok = true;
        for (int steps = 0; cur > 0; ++steps) {
            if (steps > 512) { ok = false; break; }
            const int32_t cb = trie_b(cur);
            if (cb >= 0) { ok = false; break; }
            const int32_t p = -cb - 1;
            if (p < 0 || static_cast<size_t>(p) >= trie_n) { ok = false; break; }
            const int32_t pa2 = trie_a(static_cast<size_t>(p));
            if (pa2 >= 0) { ok = false; break; }
            const int32_t ch = (-pa2 - 1) ^ static_cast<int32_t>(cur);
            if (ch < 0 || ch >= 256) { ok = false; break; }
            if (ch != 0) rev.push_back(static_cast<char>(ch));
            cur = static_cast<size_t>(p);
        }
        if (!ok) continue;
        std::reverse(rev.begin(), rev.end());
        const std::string full = to_lower_ascii(rev + tail);
        const bool is_nav = (full.find(".winvhe") != std::string::npos || full.find(".winvda") != std::string::npos);
        const bool is_lvb = (full.find("_landmark.lvb") != std::string::npos || full.find("_npc.lvb") != std::string::npos);
        const bool is_gmk = (full == "/bdat/common_gmk.bdat");
        // mnu_face.wilay is the BC7 menu face atlas: the Expansion Pass and Switch 2 Edition blades have no
        // portrait in face.wilay, so the voice cards need it too.
        const bool is_face = (full == "/menu/image/face.wilay" || full == "/menu/image/dlc3_btl_face.wilay" ||
                              full == "/menu/image/btl_face.wilay" || full == "/menu/image/mnu_face.wilay");
        if (!is_nav && !is_lvb && !is_gmk && !is_face) continue;

        const uint8_t* ep = ents + static_cast<size_t>(entry_idx) * 24;
        ArhEntry ae;
        ae.offset = rd_u64(ep + 0x00);
        ae.comp   = rd_u32(ep + 0x08);
        ae.unc    = rd_u32(ep + 0x0C);
        ae.flags  = rd_u32(ep + 0x10);
        out.by_path[full] = ae;
    }

    NXLOG_I(kTag, "indexed %zu navmesh archive files from bf2.arh", out.by_path.size());
    return !out.by_path.empty();
}

// ---- Walkable Polygon Graph + Spatial Grid + A* ----
struct PolyNode {
    NavVec3 center;
    uint32_t edge_start = 0;
    uint16_t edge_count = 0;
};

struct PolyEdge {
    uint32_t to = 0;
    float weight = 0.f;
    NavVec3 mid;  // midpoint of shared portal edge
};

struct QuantPt {
    int32_t x, y, z;
    bool operator==(const QuantPt& o) const { return x == o.x && y == o.y && z == o.z; }
    bool operator<(const QuantPt& o) const {
        if (x != o.x) return x < o.x;
        if (y != o.y) return y < o.y;
        return z < o.z;
    }
};

struct EdgeKey {
    QuantPt a, b;
    bool operator==(const EdgeKey& o) const { return a == o.a && b == o.b; }
};

struct EdgeKeyHash {
    size_t operator()(const EdgeKey& k) const noexcept {
        uint64_t h = 14695981039346656037ull;
        auto mix = [&](int32_t v) {
            h ^= static_cast<uint64_t>(static_cast<uint32_t>(v));
            h *= 1099511628211ull;
        };
        mix(k.a.x); mix(k.a.y); mix(k.a.z);
        mix(k.b.x); mix(k.b.y); mix(k.b.z);
        return static_cast<size_t>(h);
    }
};

struct CellKey {
    int32_t cx, cz;
    bool operator==(const CellKey& o) const { return cx == o.cx && cz == o.cz; }
};

struct CellKeyHash {
    size_t operator()(const CellKey& c) const noexcept {
        return static_cast<size_t>(static_cast<uint64_t>(static_cast<uint32_t>(c.cx)) * 73856093ull ^
                                   static_cast<uint64_t>(static_cast<uint32_t>(c.cz)) * 19349663ull);
    }
};

struct MapNavMesh {
    std::string map_res;
    std::vector<PolyNode> polys;
    std::vector<PolyEdge> edges;
    std::vector<uint32_t> comp_id;    // connected-component root id per polygon
    std::vector<uint32_t> comp_size;  // size of each polygon's connected component
    std::unordered_map<CellKey, std::vector<uint32_t>, CellKeyHash> grid;  // 8m x 8m XZ spatial hash
    static constexpr float kCellSize = 8.0f;

    int32_t find_nearest(NavVec3 p, uint32_t min_comp_size = 16u, uint32_t prefer_comp = UINT32_MAX) const {
        if (polys.empty()) return -1;
        const int32_t gx = static_cast<int32_t>(std::floor(p.x / kCellSize));
        const int32_t gz = static_cast<int32_t>(std::floor(p.z / kCellSize));

        auto search_ring = [&](uint32_t req_comp, uint32_t req_min_sz, int max_rad, float max_d2) -> int32_t {
            int32_t best = -1;
            float best_d2 = max_d2;
            for (int rad = 0; rad <= max_rad; ++rad) {
                for (int dx = -rad; dx <= rad; ++dx) {
                    for (int dz = -rad; dz <= rad; ++dz) {
                        if (rad > 0 && std::abs(dx) != rad && std::abs(dz) != rad) continue;
                        auto it = grid.find({gx + dx, gz + dz});
                        if (it == grid.end()) continue;
                        for (uint32_t pid : it->second) {
                            if (req_comp != UINT32_MAX && comp_id[pid] != req_comp) continue;
                            if (req_min_sz > 0 && comp_size[pid] < req_min_sz) continue;
                            const NavVec3& c = polys[pid].center;
                            const float ddx = c.x - p.x;
                            const float ddy = (c.y - p.y) * 2.2f;  // bias toward matching floor elevation
                            const float ddz = c.z - p.z;
                            const float d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                            if (d2 < best_d2) {
                                best_d2 = d2;
                                best = static_cast<int32_t>(pid);
                            }
                        }
                    }
                }
                if (best >= 0 && best_d2 < (rad + 1) * (rad + 1) * kCellSize * kCellSize) break;
            }
            return best;
        };

        // 1. If a preferred component is given (e.g. player's island), snap to it when within 24m.
        if (prefer_comp != UINT32_MAX && !comp_id.empty()) {
            const int32_t pref = search_ring(prefer_comp, 0u, 3, 24.0f * 24.0f);
            if (pref >= 0) return pref;
        }
        // 2. Avoid snapping onto tiny isolated 1..15 poly rocks/ledges when walkable terrain is nearby.
        if (min_comp_size > 1u && !comp_size.empty()) {
            const int32_t main_hit = search_ring(UINT32_MAX, min_comp_size, 6, 1e30f);
            if (main_hit >= 0) return main_hit;
        }
        // 3. Standard unrestricted ring search.
        int32_t best = search_ring(UINT32_MAX, 0u, 6, 1e30f);
        if (best >= 0) return best;

        // 4. Fallback linear scan if player/target is far outside NavMesh bounds.
        float best_d2 = 1e30f;
        for (size_t i = 0; i < polys.size(); ++i) {
            const NavVec3& c = polys[i].center;
            const float ddx = c.x - p.x, ddy = (c.y - p.y) * 2.2f, ddz = c.z - p.z;
            const float d2 = ddx * ddx + ddy * ddy + ddz * ddz;
            if (d2 < best_d2) { best_d2 = d2; best = static_cast<int32_t>(i); }
        }
        return best;
    }
};

inline QuantPt quantize_pt(NavVec3 v) {
    return {
        static_cast<int32_t>(std::lround(v.x * 4.0f)),
        static_cast<int32_t>(std::lround(v.y * 4.0f)),
        static_cast<int32_t>(std::lround(v.z * 4.0f)),
    };
}

bool build_map_navmesh(const ArhIndex& arh, const std::string& map_res, MapNavMesh& out) {
    const auto t0 = std::chrono::steady_clock::now();
    out = MapNavMesh{};
    out.map_res = map_res;

    const std::string res_lo = to_lower_ascii(map_res);
    const std::string he_key = "/map/" + res_lo + ".winvhe";
    const std::string da_key = "/map/" + res_lo + ".winvda";
    auto it_he = arh.by_path.find(he_key);
    auto it_da = arh.by_path.find(da_key);
    if (it_he == arh.by_path.end() || it_da == arh.by_path.end()) {
        NXLOG_I(kTag, "no navmesh archive entry for %s", map_res.c_str());
        return false;
    }

    std::ifstream ard(arh.ard_path, std::ios::binary);
    if (!ard) return false;

    // 1. Read & decompress .winvhe
    std::vector<uint8_t> he_raw;
    {
        const ArhEntry& e = it_he->second;
        const size_t read_sz = (e.flags == 2) ? (0x30u + e.comp) : (e.comp ? e.comp : e.unc);
        std::vector<uint8_t> tmp(read_sz);
        ard.seekg(static_cast<std::streamoff>(e.offset), std::ios::beg);
        if (!ard.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(read_sz))) return false;
        if (e.flags == 2) {
            if (!xbc1_decompress(tmp.data(), tmp.size(), he_raw)) return false;
        } else {
            he_raw = std::move(tmp);
        }
    }
    if (he_raw.size() < 0x40 || std::memcmp(he_raw.data(), "NVMS", 4) != 0) return false;

    const uint32_t tile_tbl_off = rd_u32(he_raw.data() + 0x08);
    const uint32_t tile_cnt     = rd_u32(he_raw.data() + 0x0C);
    if (tile_tbl_off + static_cast<uint64_t>(tile_cnt) * 16ull > he_raw.size()) return false;

    // 2. Decompress all NVPT tiles from .winvda and collect polygons + quantized shared edges
    struct EdgeRef {
        uint32_t poly_id;
        NavVec3 mid;
    };
    std::unordered_map<EdgeKey, std::vector<EdgeRef>, EdgeKeyHash> edge_map;
    edge_map.reserve(131072);
    out.polys.reserve(32768);

    const uint64_t da_base = it_da->second.offset;
    uint32_t tiles_ok = 0;
    std::vector<uint8_t> comp_buf;
    std::vector<uint8_t> tile_raw;
    std::vector<NavVec3> verts;

    for (uint32_t ti = 0; ti < tile_cnt; ++ti) {
        const uint8_t* te = he_raw.data() + tile_tbl_off + static_cast<size_t>(ti) * 16;
        const uint32_t off_da = rd_u32(te + 0x04);
        const uint32_t unc_sz = rd_u32(te + 0x08);
        if (off_da == 0 || unc_sz == 0) continue;

        uint8_t thdr[0x30];
        ard.seekg(static_cast<std::streamoff>(da_base + off_da), std::ios::beg);
        if (!ard.read(reinterpret_cast<char*>(thdr), 0x30)) continue;
        if (std::memcmp(thdr, "xbc1", 4) != 0) continue;
        const uint32_t t_comp = rd_u32(thdr + 0x0C);
        if (t_comp == 0 || t_comp > 16u * 1024u * 1024u) continue;
        comp_buf.resize(0x30u + t_comp);
        std::memcpy(comp_buf.data(), thdr, 0x30);
        if (!ard.read(reinterpret_cast<char*>(comp_buf.data() + 0x30), t_comp)) continue;
        if (!xbc1_decompress(comp_buf.data(), comp_buf.size(), tile_raw)) continue;
        if (tile_raw.size() < 0x70 || std::memcmp(tile_raw.data(), "NVPT", 4) != 0) continue;
        ++tiles_ok;

        const uint32_t poly_off = rd_u32(tile_raw.data() + 0x28);
        const uint32_t poly_cnt = rd_u32(tile_raw.data() + 0x2C);
        const uint32_t vert_off = rd_u32(tile_raw.data() + 0x30);
        const uint32_t vert_cnt = rd_u32(tile_raw.data() + 0x34);
        const float ox = rd_f32(tile_raw.data() + 0x5C);
        const float oy = rd_f32(tile_raw.data() + 0x60);
        const float oz = rd_f32(tile_raw.data() + 0x64);
        const float sc = rd_f32(tile_raw.data() + 0x68);
        if (vert_off + static_cast<uint64_t>(vert_cnt) * 6ull > tile_raw.size()) continue;
        if (poly_off + static_cast<uint64_t>(poly_cnt) * 8ull > tile_raw.size()) continue;

        verts.resize(vert_cnt);
        for (uint32_t vi = 0; vi < vert_cnt; ++vi) {
            const uint8_t* vp = tile_raw.data() + vert_off + static_cast<size_t>(vi) * 6;
            verts[vi] = {ox + rd_u16(vp) * sc, oy + rd_u16(vp + 2) * sc, oz + rd_u16(vp + 4) * sc};
        }

        for (uint32_t pi = 0; pi < poly_cnt; ++pi) {
            const uint8_t* pd = tile_raw.data() + poly_off + static_cast<size_t>(pi) * 8;
            const uint8_t vc = pd[0];
            const uint32_t idx_off = rd_u32(pd + 4);
            if (vc < 3 || vc > 32 || idx_off + static_cast<uint64_t>(vc) * 2ull > tile_raw.size()) continue;

            NavVec3 pts[32];
            uint32_t npts = 0;
            float sx = 0.f, sy = 0.f, sz = 0.f;
            for (uint8_t k = 0; k < vc; ++k) {
                const uint16_t vi = rd_u16(tile_raw.data() + idx_off + static_cast<size_t>(k) * 2);
                if (vi >= vert_cnt) { npts = 0; break; }
                pts[npts++] = verts[vi];
                sx += verts[vi].x;
                sy += verts[vi].y;
                sz += verts[vi].z;
            }
            if (npts < 3) continue;

            const uint32_t pid = static_cast<uint32_t>(out.polys.size());
            const float inv = 1.0f / static_cast<float>(npts);
            PolyNode node;
            node.center = {sx * inv, sy * inv, sz * inv};
            out.polys.push_back(node);

            const int32_t gx = static_cast<int32_t>(std::floor(node.center.x / MapNavMesh::kCellSize));
            const int32_t gz = static_cast<int32_t>(std::floor(node.center.z / MapNavMesh::kCellSize));
            out.grid[{gx, gz}].push_back(pid);

            for (uint32_t k = 0; k < npts; ++k) {
                const NavVec3& p0 = pts[k];
                const NavVec3& p1 = pts[(k + 1) % npts];
                QuantPt qa = quantize_pt(p0);
                QuantPt qb = quantize_pt(p1);
                if (qa == qb) continue;
                EdgeKey ek = (qa < qb) ? EdgeKey{qa, qb} : EdgeKey{qb, qa};
                NavVec3 mid{(p0.x + p1.x) * 0.5f, (p0.y + p1.y) * 0.5f, (p0.z + p1.z) * 0.5f};
                edge_map[ek].push_back({pid, mid});
            }
        }
    }

    // 3. Build adjacency list + Union-Find components + boundary-edge proximity stitching across small tile/bridge seams
    const size_t N = out.polys.size();
    std::vector<uint32_t> uf(N);
    for (size_t i = 0; i < N; ++i) uf[i] = static_cast<uint32_t>(i);
    auto uf_find = [&](uint32_t x) -> uint32_t {
        while (uf[x] != x) {
            uf[x] = uf[uf[x]];
            x = uf[x];
        }
        return x;
    };
    auto uf_union = [&](uint32_t a, uint32_t b) {
        const uint32_t ra = uf_find(a);
        const uint32_t rb = uf_find(b);
        if (ra != rb) uf[ra] = rb;
    };

    std::vector<std::vector<PolyEdge>> adj(N);
    size_t shared_edges = 0;
    struct OpenEdge {
        uint32_t pid;
        NavVec3 mid;
    };
    std::unordered_map<CellKey, std::vector<OpenEdge>, CellKeyHash> open_grid;
    constexpr float kStitchCell = 2.5f;

    for (const auto& kv : edge_map) {
        const auto& refs = kv.second;
        if (refs.size() == 1) {
            const NavVec3& m = refs[0].mid;
            const int32_t gx = static_cast<int32_t>(std::floor(m.x / kStitchCell));
            const int32_t gz = static_cast<int32_t>(std::floor(m.z / kStitchCell));
            open_grid[{gx, gz}].push_back({refs[0].poly_id, m});
            continue;
        }
        if (refs.size() < 2 || refs.size() > 6) continue;
        ++shared_edges;
        for (size_t i = 0; i < refs.size(); ++i) {
            for (size_t j = i + 1; j < refs.size(); ++j) {
                const uint32_t a = refs[i].poly_id;
                const uint32_t b = refs[j].poly_id;
                if (a == b) continue;
                const NavVec3& ca = out.polys[a].center;
                const NavVec3& cb = out.polys[b].center;
                const float w = std::sqrt((ca.x - cb.x) * (ca.x - cb.x) +
                                          (ca.y - cb.y) * (ca.y - cb.y) +
                                          (ca.z - cb.z) * (ca.z - cb.z));
                adj[a].push_back({b, w, refs[i].mid});
                adj[b].push_back({a, w, refs[i].mid});
                uf_union(a, b);
            }
        }
    }

    // Stitch open boundary edges from distinct components within 2.5m XZ and 2.8m Y (bridges, stairs, tile T-junctions).
    size_t stitched_edges = 0;
    constexpr float kMaxStitchXZ2 = 2.5f * 2.5f;
    constexpr float kMaxStitchY   = 2.8f;
    for (const auto& kv : open_grid) {
        const CellKey ck = kv.first;
        const auto& cell_edges = kv.second;
        for (int dx = 0; dx <= 1; ++dx) {
            for (int dz = (dx == 0 ? 0 : -1); dz <= 1; ++dz) {
                auto nit = (dx == 0 && dz == 0) ? open_grid.find(ck) : open_grid.find({ck.cx + dx, ck.cz + dz});
                if (nit == open_grid.end()) continue;
                const auto& nbr_edges = nit->second;
                for (const auto& ea : cell_edges) {
                    const uint32_t ra = uf_find(ea.pid);
                    for (const auto& eb : nbr_edges) {
                        if (ea.pid == eb.pid) continue;
                        if (uf_find(eb.pid) == ra) continue;
                        const float ddy = std::fabs(ea.mid.y - eb.mid.y);
                        if (ddy > kMaxStitchY) continue;
                        const float ddx = ea.mid.x - eb.mid.x;
                        const float ddz = ea.mid.z - eb.mid.z;
                        if (ddx * ddx + ddz * ddz > kMaxStitchXZ2) continue;
                        const NavVec3& ca = out.polys[ea.pid].center;
                        const NavVec3& cb = out.polys[eb.pid].center;
                        const float w = std::sqrt((ca.x - cb.x) * (ca.x - cb.x) +
                                                  (ca.y - cb.y) * (ca.y - cb.y) +
                                                  (ca.z - cb.z) * (ca.z - cb.z)) + 2.0f;
                        const NavVec3 portal{(ea.mid.x + eb.mid.x) * 0.5f,
                                             (ea.mid.y + eb.mid.y) * 0.5f,
                                             (ea.mid.z + eb.mid.z) * 0.5f};
                        adj[ea.pid].push_back({eb.pid, w, portal});
                        adj[eb.pid].push_back({ea.pid, w, portal});
                        uf_union(ea.pid, eb.pid);
                        ++stitched_edges;
                        break;  // one bridge per open edge is enough to merge components cleanly
                    }
                }
            }
        }
    }

    // Finalize connected-component IDs & sizes so find_nearest() avoids tiny isolated decorative rocks/ledges.
    out.comp_id.resize(N);
    std::vector<uint32_t> root_cnt(N, 0u);
    for (size_t i = 0; i < N; ++i) {
        const uint32_t r = uf_find(static_cast<uint32_t>(i));
        out.comp_id[i] = r;
        ++root_cnt[r];
    }
    out.comp_size.resize(N);
    size_t num_comps = 0;
    for (size_t i = 0; i < N; ++i) {
        if (root_cnt[i] > 0) ++num_comps;
        out.comp_size[i] = root_cnt[out.comp_id[i]];
    }

    out.edges.reserve((shared_edges + stitched_edges) * 2);
    for (size_t i = 0; i < N; ++i) {
        out.polys[i].edge_start = static_cast<uint32_t>(out.edges.size());
        out.polys[i].edge_count = static_cast<uint16_t>(std::min<size_t>(adj[i].size(), 65535u));
        for (size_t k = 0; k < out.polys[i].edge_count; ++k) {
            out.edges.push_back(adj[i][k]);
        }
    }

    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    NXLOG_I(kTag, "loaded %s: %u tiles, %zu polys, %zu shared + %zu stitched edges (%zu comps) in %.1f ms",
            map_res.c_str(), tiles_ok, N, shared_edges, stitched_edges, num_comps, ms);
    return !out.polys.empty();
}

// Smooth portal polyline (Laplacian straightening + Catmull-Rom uniform resampling) while hugging terrain elevation.
void smooth_path(std::vector<NavVec3>& pts) {
    if (pts.size() <= 2) return;

    // 1. 4 passes of gentle Laplacian smoothing on XZ (lighter on Y to preserve slope contour)
    std::vector<NavVec3> tmp = pts;
    for (int pass = 0; pass < 4; ++pass) {
        for (size_t i = 1; i + 1 < pts.size(); ++i) {
            tmp[i].x = pts[i - 1].x * 0.28f + pts[i].x * 0.44f + pts[i + 1].x * 0.28f;
            tmp[i].y = pts[i - 1].y * 0.15f + pts[i].y * 0.70f + pts[i + 1].y * 0.15f;
            tmp[i].z = pts[i - 1].z * 0.28f + pts[i].z * 0.44f + pts[i + 1].z * 0.28f;
        }
        pts.swap(tmp);
    }

    // 2. Thin out tight micro-zigzags (< 2.5m) into clean control points
    std::vector<NavVec3> ctrl;
    ctrl.reserve(pts.size());
    ctrl.push_back(pts.front());
    for (size_t i = 1; i + 1 < pts.size(); ++i) {
        const NavVec3& prev = ctrl.back();
        const float dx = pts[i].x - prev.x, dy = pts[i].y - prev.y, dz = pts[i].z - prev.z;
        if (dx * dx + dy * dy + dz * dz >= 6.25f) {
            ctrl.push_back(pts[i]);
        }
    }
    ctrl.push_back(pts.back());
    if (ctrl.size() <= 2) {
        pts.swap(ctrl);
        return;
    }

    // 3. Catmull-Rom spline resampling at ~1.25m world spacing for silky curved 3D ribbon
    auto catmull = [](const NavVec3& p0, const NavVec3& p1, const NavVec3& p2, const NavVec3& p3, float t) -> NavVec3 {
        const float t2 = t * t;
        const float t3 = t2 * t;
        auto eval = [&](float v0, float v1, float v2, float v3) {
            return 0.5f * ((2.0f * v1) +
                           (-v0 + v2) * t +
                           (2.0f * v0 - 5.0f * v1 + 4.0f * v2 - v3) * t2 +
                           (-v0 + 3.0f * v1 - 3.0f * v2 + v3) * t3);
        };
        return {eval(p0.x, p1.x, p2.x, p3.x), eval(p0.y, p1.y, p2.y, p3.y), eval(p0.z, p1.z, p2.z, p3.z)};
    };

    std::vector<NavVec3> out;
    out.reserve(ctrl.size() * 3);
    out.push_back(ctrl.front());
    for (size_t i = 0; i + 1 < ctrl.size(); ++i) {
        const NavVec3& p0 = ctrl[i > 0 ? i - 1 : 0];
        const NavVec3& p1 = ctrl[i];
        const NavVec3& p2 = ctrl[i + 1];
        const NavVec3& p3 = ctrl[i + 2 < ctrl.size() ? i + 2 : ctrl.size() - 1];
        const float seg_d = std::sqrt((p2.x - p1.x) * (p2.x - p1.x) +
                                      (p2.y - p1.y) * (p2.y - p1.y) +
                                      (p2.z - p1.z) * (p2.z - p1.z));
        const int steps = std::clamp(static_cast<int>(std::ceil(seg_d / 1.25f)), 1, 16);
        for (int s = 1; s <= steps; ++s) {
            out.push_back(catmull(p0, p1, p2, p3, static_cast<float>(s) / static_cast<float>(steps)));
            if (out.size() >= 640) break;
        }
        if (out.size() >= 640) break;
    }
    pts.swap(out);
}

void emit_resampled_line(NavVec3 a, NavVec3 b, float max_len, std::vector<NavVec3>& out_pts, float& out_len) {
    out_pts.clear();
    const float dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
    const float full_d = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (full_d < 0.5f) {
        out_len = 0.f;
        return;
    }
    const float d = std::min(full_d, max_len);
    const float ratio = d / full_d;
    const int steps = std::clamp(static_cast<int>(std::ceil(d / 1.5f)), 2, 32);
    out_pts.reserve(static_cast<size_t>(steps) + 1);
    for (int i = 0; i <= steps; ++i) {
        const float t = (static_cast<float>(i) / static_cast<float>(steps)) * ratio;
        out_pts.push_back({a.x + dx * t, a.y + dy * t, a.z + dz * t});
    }
    out_len = d;
}

bool solve_astar(const MapNavMesh& mesh, NavVec3 start_p, NavVec3 goal_p,
                 std::vector<NavVec3>& out_pts, float& out_len, bool& out_partial) {
    out_pts.clear();
    out_len = 0.f;
    out_partial = false;
    const int32_t s_id = mesh.find_nearest(start_p, 16u);
    if (s_id < 0) return false;
    const uint32_t s_comp = mesh.comp_id.empty() ? UINT32_MAX : mesh.comp_id[static_cast<size_t>(s_id)];
    const int32_t g_id = mesh.find_nearest(goal_p, 16u, s_comp);
    if (g_id < 0) return false;

    const NavVec3 s_ground{start_p.x, mesh.polys[s_id].center.y, start_p.z};
    const NavVec3 g_ground{goal_p.x, mesh.polys[g_id].center.y, goal_p.z};
    if (s_id == g_id) {
        emit_resampled_line(s_ground, g_ground, 80.0f, out_pts, out_len);
        return out_pts.size() >= 2;
    }

    const size_t N = mesh.polys.size();
    auto heur = [&](uint32_t pid) -> float {
        const NavVec3& c = mesh.polys[pid].center;
        const float dx = c.x - goal_p.x;
        const float dy = c.y - goal_p.y;
        const float dz = c.z - goal_p.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    struct QItem {
        float f;
        float g;
        uint32_t id;
        bool operator>(const QItem& o) const { return f > o.f; }
    };

    std::vector<float> gscore(N, 1e30f);
    std::vector<int32_t> parent(N, -1);
    std::vector<NavVec3> parent_mid(N);
    std::priority_queue<QItem, std::vector<QItem>, std::greater<QItem>> pq;

    const float start_h = heur(static_cast<uint32_t>(s_id));
    gscore[s_id] = 0.f;
    pq.push({start_h, 0.f, static_cast<uint32_t>(s_id)});

    bool found = false;
    int32_t best_id = s_id;
    float best_h = start_h;
    const uint32_t max_expand = std::clamp<uint32_t>(static_cast<uint32_t>(N), 80000u, 160000u);
    uint32_t expanded = 0;
    while (!pq.empty() && expanded < max_expand) {
        const QItem cur = pq.top();
        pq.pop();
        if (cur.g > gscore[cur.id]) continue;
        const float ch = heur(cur.id);
        if (ch < best_h) {
            best_h = ch;
            best_id = static_cast<int32_t>(cur.id);
        }
        if (static_cast<int32_t>(cur.id) == g_id) {
            found = true;
            break;
        }
        ++expanded;
        const PolyNode& pn = mesh.polys[cur.id];
        for (uint16_t k = 0; k < pn.edge_count; ++k) {
            const PolyEdge& e = mesh.edges[pn.edge_start + k];
            const float ng = cur.g + e.weight;
            if (ng < gscore[e.to]) {
                gscore[e.to] = ng;
                parent[e.to] = static_cast<int32_t>(cur.id);
                parent_mid[e.to] = e.mid;
                pq.push({ng + heur(e.to), ng, e.to});
            }
        }
    }

    int32_t end_id = found ? g_id : best_id;
    if (!found) {
        out_partial = true;
        // If player is already standing at the closest edge of their island (e.g. facing a ladder/cliff/vine),
        // emit a short directional ground guide toward the objective so the 3D ribbon never silently disappears.
        if (end_id == s_id || end_id < 0) {
            NavVec3 dir_target = goal_p;
            dir_target.y = s_ground.y + std::clamp(goal_p.y - s_ground.y, -4.0f, 4.0f);
            emit_resampled_line(s_ground, dir_target, 28.0f, out_pts, out_len);
            return out_pts.size() >= 2;
        }
    }

    // Snap start/end Y onto the walkable NavMesh floor so the guide line never floats up to cam_tgt.
    out_pts.push_back(found ? g_ground : mesh.polys[end_id].center);
    int32_t cur = end_id;
    while (cur != s_id && cur >= 0) {
        out_pts.push_back(parent_mid[cur]);
        cur = parent[cur];
    }
    out_pts.push_back(s_ground);
    std::reverse(out_pts.begin(), out_pts.end());
    smooth_path(out_pts);

    float total = 0.f;
    for (size_t i = 1; i < out_pts.size(); ++i) {
        const float dx = out_pts[i].x - out_pts[i - 1].x;
        const float dy = out_pts[i].y - out_pts[i - 1].y;
        const float dz = out_pts[i].z - out_pts[i - 1].z;
        total += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    out_len = total;
    return out_pts.size() >= 2;
}

// ---- Background worker singleton ----
struct WorkerState {
    std::mutex mu;
    std::condition_variable cv;
    std::thread thr;
    bool running = false;
    bool stop_req = false;

    std::filesystem::path romfs_dir;
    std::string req_map;
    NavVec3 req_start{};
    NavVec3 req_goal{};
    uint64_t req_gen = 0;

    // Last submitted coords for throttling
    std::string last_sub_map;
    NavVec3 last_sub_start{};
    NavVec3 last_sub_goal{};
    std::chrono::steady_clock::time_point last_sub_time{};

    // Output snapshot
    PathResult result;

    // Cached LVLB lists & lazy-loaded common_gmk.bdat (for 3-level cross-map skip-travel lookup)
    std::unordered_map<std::string, std::vector<LvbLandmark>> lvb_cache;
    std::unordered_map<std::string, std::vector<OffMapLandmark>> offmap_lm_cache;
    ArhIndex lvb_arh;
    bool gmk_bdat_loaded = false;
    std::vector<uint8_t> gmk_bdat_raw;
    std::unordered_map<std::string, std::vector<uint8_t>> gmk_tables;
};

WorkerState& ws() {
    static WorkerState s;
    return s;
}

void worker_loop() {
    ArhIndex arh;
    MapNavMesh mesh;
    std::string failed_map;
    uint64_t done_gen = 0;

    while (true) {
        std::string map_res;
        NavVec3 start_p{}, goal_p{};
        std::filesystem::path romfs;
        {
            std::unique_lock<std::mutex> lk(ws().mu);
            ws().cv.wait(lk, [&] { return ws().stop_req || ws().req_gen != done_gen; });
            if (ws().stop_req) break;
            done_gen = ws().req_gen;
            map_res = ws().req_map;
            start_p = ws().req_start;
            goal_p = ws().req_goal;
            romfs = ws().romfs_dir;
        }

        if (map_res.empty()) continue;

        if (!arh.loaded) {
            load_arh_index(romfs, arh);
        }

        if (mesh.map_res != map_res && failed_map != map_res) {
            if (!build_map_navmesh(arh, map_res, mesh)) {
                failed_map = map_res;
                mesh = MapNavMesh{};
                std::lock_guard<std::mutex> lk(ws().mu);
                ws().result = PathResult{};
                ws().result.map_res = map_res;
                ws().result.loaded = true;
                ws().result.reachable = false;
                continue;
            }
            failed_map.clear();
        }

        if (mesh.map_res != map_res || mesh.polys.empty()) continue;

        PathResult pr;
        pr.map_res = map_res;
        pr.loaded = true;
        pr.poly_count = mesh.polys.size();
        pr.reachable = solve_astar(mesh, start_p, goal_p, pr.points, pr.path_len, pr.partial);

        static std::string s_last_logged_map;
        static int s_last_logged_state = -1;
        const int cur_state = !pr.reachable ? 0 : (pr.partial ? 1 : 2);
        if (s_last_logged_map != map_res || s_last_logged_state != cur_state) {
            s_last_logged_map = map_res;
            s_last_logged_state = cur_state;
            NXLOG_I(kTag, "path %s: %s (len=%.1fm, pts=%zu)",
                    map_res.c_str(),
                    cur_state == 2 ? "full" : (cur_state == 1 ? "partial-best-effort" : "none"),
                    pr.path_len, pr.points.size());
        }

        std::lock_guard<std::mutex> lk(ws().mu);
        ws().result = std::move(pr);
    }
}

}  // namespace

void mesh_init(const std::string& romfs_dir) {
    std::lock_guard<std::mutex> lk(ws().mu);
    ws().romfs_dir = std::filesystem::path(romfs_dir);
    if (!ws().running) {
        ws().stop_req = false;
        ws().running = true;
        ws().thr = std::thread(worker_loop);
    }
}

void mesh_shutdown() {
    std::thread to_join;
    {
        std::lock_guard<std::mutex> lk(ws().mu);
        if (!ws().running) return;
        ws().stop_req = true;
        to_join = std::move(ws().thr);
        ws().running = false;
    }
    ws().cv.notify_all();
    if (to_join.joinable()) to_join.join();
}

void mesh_update_and_get(const std::string& map_res, NavVec3 player, NavVec3 goal, PathResult& out) {
    const auto now = std::chrono::steady_clock::now();
    bool wake = false;
    {
        std::lock_guard<std::mutex> lk(ws().mu);
        if (ws().running && !map_res.empty()) {
            const float dp = std::sqrt((player.x - ws().last_sub_start.x) * (player.x - ws().last_sub_start.x) +
                                       (player.y - ws().last_sub_start.y) * (player.y - ws().last_sub_start.y) +
                                       (player.z - ws().last_sub_start.z) * (player.z - ws().last_sub_start.z));
            const float dg = std::sqrt((goal.x - ws().last_sub_goal.x) * (goal.x - ws().last_sub_goal.x) +
                                       (goal.y - ws().last_sub_goal.y) * (goal.y - ws().last_sub_goal.y) +
                                       (goal.z - ws().last_sub_goal.z) * (goal.z - ws().last_sub_goal.z));
            const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - ws().last_sub_time).count();
            const bool changed = (map_res != ws().last_sub_map) || (dg > 2.0f) || (dp > 2.5f && dt_ms >= 200);
            if (changed) {
                ws().req_map = map_res;
                ws().req_start = player;
                ws().req_goal = goal;
                ws().last_sub_map = map_res;
                ws().last_sub_start = player;
                ws().last_sub_goal = goal;
                ws().last_sub_time = now;
                ++ws().req_gen;
                wake = true;
            }
        }
        out = ws().result;
    }
    if (wake) ws().cv.notify_one();
}

namespace {

bool read_ard_entry(const std::filesystem::path& ard_path, const ArhEntry& e, std::vector<uint8_t>& out) {
    out.clear();
    std::ifstream ard(ard_path, std::ios::binary);
    if (!ard) return false;
    const size_t read_sz = (e.flags == 2) ? (0x30u + e.comp) : (e.comp ? e.comp : e.unc);
    if (read_sz == 0 || read_sz > 64u * 1024u * 1024u) return false;
    std::vector<uint8_t> tmp(read_sz);
    ard.seekg(static_cast<std::streamoff>(e.offset), std::ios::beg);
    if (!ard.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(read_sz))) return false;
    if (e.flags == 2 && tmp.size() >= 0x30 && std::memcmp(tmp.data(), "xbc1", 4) == 0) {
        return xbc1_decompress(tmp.data(), tmp.size(), out);
    }
    out = std::move(tmp);
    return !out.empty();
}

bool read_lvb_locked(const std::string& key, std::vector<LvbLandmark>& out) {
    out.clear();
    auto cit = ws().lvb_cache.find(key);
    if (cit != ws().lvb_cache.end()) {
        out = cit->second;
        return !out.empty();
    }
    if (!ws().lvb_arh.loaded && !ws().romfs_dir.empty()) {
        load_arh_index(ws().romfs_dir, ws().lvb_arh);
    }
    auto ait = ws().lvb_arh.by_path.find(key);
    if (ait == ws().lvb_arh.by_path.end()) {
        ws().lvb_cache[key] = {};
        return false;
    }
    std::vector<uint8_t> raw;
    if (!read_ard_entry(ws().lvb_arh.ard_path, ait->second, raw) ||
        raw.size() < 0x40 || std::memcmp(raw.data(), "LVLB", 4) != 0) {
        ws().lvb_cache[key] = {};
        return false;
    }

    // Parse XFRM (stride 0x40: Vec3 pos at +0x00, hash_id at +0x30) and STRG (NUL-terminated names) chunks.
    std::vector<LvbLandmark> parsed;
    std::vector<std::string> str_names;
    size_t pos = 0x20;
    while (pos + 0x20 <= raw.size()) {
        const uint32_t csz = rd_u32(raw.data() + pos + 4);
        const uint32_t cnt = rd_u32(raw.data() + pos + 12);
        if (csz < 0x20 || pos + csz > raw.size()) break;
        if (std::memcmp(raw.data() + pos, "XFRM", 4) == 0) {
            const size_t data_off = pos + 0x20;
            parsed.reserve(cnt);
            for (uint32_t i = 0; i < cnt && data_off + static_cast<size_t>(i + 1) * 0x40 <= pos + csz; ++i) {
                const uint8_t* ep = raw.data() + data_off + static_cast<size_t>(i) * 0x40;
                LvbLandmark lm;
                lm.pos = {rd_f32(ep + 0x00), rd_f32(ep + 0x04), rd_f32(ep + 0x08)};
                lm.hash_id = rd_u32(ep + 0x30);
                parsed.push_back(std::move(lm));
            }
        } else if (std::memcmp(raw.data() + pos, "STRG", 4) == 0) {
            size_t sp = pos + 0x20;
            const size_t send = pos + csz;
            while (sp < send) {
                const char* s = reinterpret_cast<const char*>(raw.data() + sp);
                size_t len = 0;
                while (sp + len < send && raw[sp + len] != 0) ++len;
                if (len > 0) str_names.emplace_back(s, len);
                sp += len + 1;
            }
        }
        pos += csz;
    }
    for (size_t i = 0; i < parsed.size() && i < str_names.size(); ++i) {
        parsed[i].name = str_names[i];
    }
    ws().lvb_cache[key] = parsed;
    out = std::move(parsed);
    return !out.empty();
}

void decrypt_bdat_v2(std::vector<uint8_t>& d) {
    if (d.size() < 0x20 || std::memcmp(d.data(), "BDAT", 4) != 0) return;
    const uint16_t flags = rd_u16(d.data() + 0x04);
    if ((flags & 2u) == 0u) return;
    const uint16_t name_off = rd_u16(d.data() + 0x06);
    const uint16_t b_off    = rd_u16(d.data() + 0x0A);
    const uint32_t str_off  = rd_u32(d.data() + 0x18);
    const uint32_t str_sz   = rd_u32(d.data() + 0x1C);
    const uint8_t k1_init   = static_cast<uint8_t>(~d[0x17]);
    const uint8_t k2_init   = static_cast<uint8_t>(~d[0x16]);

    auto unscr = [&](size_t start, size_t end) {
        if (start >= d.size() || end > d.size() || start >= end) return;
        uint8_t p1 = k1_init, p2 = k2_init;
        for (size_t i = start; i < end; i += 2) {
            const uint8_t c1 = d[i];
            d[i] = static_cast<uint8_t>(c1 ^ p1);
            p1 = static_cast<uint8_t>(p1 + c1);
            if (i + 1 < end) {
                const uint8_t c2 = d[i + 1];
                d[i + 1] = static_cast<uint8_t>(c2 ^ p2);
                p2 = static_cast<uint8_t>(p2 + c2);
            }
        }
    };
    unscr(name_off, b_off);
    unscr(str_off, static_cast<size_t>(str_off) + str_sz);
    // Clear encrypted bit in our private host copy.
    const uint16_t clear_flags = static_cast<uint16_t>(flags & ~2u);
    std::memcpy(d.data() + 0x04, &clear_flags, 2);
}

const std::vector<uint8_t>* get_gmk_table_locked(const std::string& target_name) {
    auto it = ws().gmk_tables.find(target_name);
    if (it != ws().gmk_tables.end()) {
        return it->second.empty() ? nullptr : &it->second;
    }
    if (!ws().gmk_bdat_loaded) {
        ws().gmk_bdat_loaded = true;
        if (!ws().lvb_arh.loaded && !ws().romfs_dir.empty()) {
            load_arh_index(ws().romfs_dir, ws().lvb_arh);
        }
        auto ait = ws().lvb_arh.by_path.find("/bdat/common_gmk.bdat");
        if (ait != ws().lvb_arh.by_path.end()) {
            read_ard_entry(ws().lvb_arh.ard_path, ait->second, ws().gmk_bdat_raw);
        }
    }
    const auto& raw = ws().gmk_bdat_raw;
    if (raw.size() < 16) return nullptr;
    const uint32_t tcnt = rd_u32(raw.data());
    if (tcnt == 0 || 8ull + static_cast<uint64_t>(tcnt) * 4ull > raw.size()) return nullptr;

    // Scan tables once and match decrypted table name (tables are small, ~258 total).
    for (uint32_t i = 0; i < tcnt; ++i) {
        const uint32_t off = rd_u32(raw.data() + 8 + i * 4);
        const uint32_t end = (i + 1 < tcnt) ? rd_u32(raw.data() + 8 + (i + 1) * 4) : static_cast<uint32_t>(raw.size());
        if (off >= raw.size() || end <= off || end > raw.size() || end - off < 0x20) continue;

        // Fast check: decrypt only the table name header first if encrypted, or decrypt whole table.
        std::vector<uint8_t> tbl(raw.begin() + off, raw.begin() + end);
        decrypt_bdat_v2(tbl);
        const uint16_t name_off = rd_u16(tbl.data() + 0x06);
        if (name_off >= tbl.size()) continue;
        size_t nul = name_off;
        while (nul < tbl.size() && tbl[nul] != 0) ++nul;
        if (nul - name_off == target_name.size() &&
            std::memcmp(tbl.data() + name_off, target_name.data(), target_name.size()) == 0) {
            auto [ins, _] = ws().gmk_tables.emplace(target_name, std::move(tbl));
            return &ins->second;
        }
    }
    ws().gmk_tables[target_name] = {};
    return nullptr;
}

bool host_bdat_find_col(const std::vector<uint8_t>& d, const char* col_name, uint16_t& out_voff, uint8_t& out_typ) {
    if (d.size() < 0x20 || std::memcmp(d.data(), "BDAT", 4) != 0) return false;
    const uint16_t b_off = rd_u16(d.data() + 0x0A);
    const uint16_t b_cnt = rd_u16(d.data() + 0x0C);
    if (b_cnt == 0 || b_cnt > 256 || b_off + b_cnt * 2u > d.size()) return false;
    for (uint16_t b = 0; b < b_cnt; ++b) {
        uint16_t e = rd_u16(d.data() + b_off + b * 2u);
        for (int guard = 0; guard < 64 && e != 0 && static_cast<size_t>(e) + 6 <= d.size(); ++guard) {
            const uint16_t desc = rd_u16(d.data() + e);
            const uint16_t nxt  = rd_u16(d.data() + e + 2);
            const uint16_t noff = rd_u16(d.data() + e + 4);
            if (noff < d.size() && static_cast<size_t>(desc) + 4 <= d.size()) {
                size_t nul = noff;
                while (nul < d.size() && d[nul] != 0) ++nul;
                if (std::strlen(col_name) == nul - noff &&
                    std::memcmp(d.data() + noff, col_name, nul - noff) == 0) {
                    out_typ  = d[desc + 1];
                    out_voff = rd_u16(d.data() + desc + 2);
                    return true;
                }
            }
            e = nxt;
        }
    }
    return false;
}

bool host_bdat_read_u32(const std::vector<uint8_t>& d, uint16_t r_idx, uint16_t voff, uint8_t typ, uint32_t& out) {
    const uint16_t row_sz  = rd_u16(d.data() + 0x08);
    const uint16_t row_off = rd_u16(d.data() + 0x0E);
    const size_t off = static_cast<size_t>(row_off) + static_cast<size_t>(r_idx) * row_sz + voff;
    if (typ == 1 || typ == 4) {
        if (off + 1 > d.size()) return false;
        out = d[off];
        return true;
    }
    if (typ == 2 || typ == 5) {
        if (off + 2 > d.size()) return false;
        out = rd_u16(d.data() + off);
        return true;
    }
    if (off + 4 > d.size()) return false;
    out = rd_u32(d.data() + off);
    return true;
}

bool host_bdat_read_str(const std::vector<uint8_t>& d, uint16_t r_idx, uint16_t voff, std::string& out) {
    out.clear();
    const uint16_t row_sz  = rd_u16(d.data() + 0x08);
    const uint16_t row_off = rd_u16(d.data() + 0x0E);
    const size_t off = static_cast<size_t>(row_off) + static_cast<size_t>(r_idx) * row_sz + voff;
    if (off + 4 > d.size()) return false;
    const uint32_t so = rd_u32(d.data() + off);
    if (so == 0 || so >= d.size()) return false;
    size_t nul = so;
    while (nul < d.size() && d[nul] != 0) ++nul;
    out.assign(reinterpret_cast<const char*>(d.data() + so), nul - so);
    return !out.empty();
}

}  // namespace

bool mesh_get_map_landmarks(const std::string& map_res, std::vector<LvbLandmark>& out) {
    out.clear();
    if (map_res.empty()) return false;
    const std::string key = "/gmk/" + to_lower_ascii(map_res) + "_landmark.lvb";
    std::lock_guard<std::mutex> lk(ws().mu);
    return read_lvb_locked(key, out);
}

bool mesh_get_offmap_landmarks(const std::string& map_res, std::vector<OffMapLandmark>& out) {
    out.clear();
    if (map_res.empty()) return false;
    const std::string res_lo = to_lower_ascii(map_res);

    std::lock_guard<std::mutex> lk(ws().mu);
    auto cit = ws().offmap_lm_cache.find(res_lo);
    if (cit != ws().offmap_lm_cache.end()) {
        out = cit->second;
        return !out.empty();
    }

    std::vector<LvbLandmark> lvbs;
    if (!read_lvb_locked("/gmk/" + res_lo + "_landmark.lvb", lvbs)) {
        ws().offmap_lm_cache[res_lo] = {};
        return false;
    }
    std::unordered_map<std::string, NavVec3> pos_by_name;
    for (const auto& lv : lvbs) {
        if (!lv.name.empty()) pos_by_name[lv.name] = lv.pos;
    }

    const auto* tbl = get_gmk_table_locked(res_lo + "_FLD_LandmarkPop");
    if (!tbl || tbl->size() < 0x20) {
        ws().offmap_lm_cache[res_lo] = {};
        return false;
    }

    uint16_t v_name = 0, v_cat = 0, v_msg = 0, v_mg = 0, v_mi = 0;
    uint8_t t_name = 0, t_cat = 0, t_msg = 0, t_mg = 0, t_mi = 0;
    const bool has_name = host_bdat_find_col(*tbl, "name", v_name, t_name);
    const bool has_cat  = host_bdat_find_col(*tbl, "category", v_cat, t_cat);
    const bool has_msg  = host_bdat_find_col(*tbl, "MSGID", v_msg, t_msg);
    const bool has_mg   = host_bdat_find_col(*tbl, "menuGroup", v_mg, t_mg);
    const bool has_mi   = host_bdat_find_col(*tbl, "menuMapImage", v_mi, t_mi);
    if (!has_name || !has_msg) {
        ws().offmap_lm_cache[res_lo] = {};
        return false;
    }

    const uint16_t row_cnt = rd_u16(tbl->data() + 0x10);
    std::vector<OffMapLandmark> parsed;
    parsed.reserve(row_cnt);
    for (uint16_t r = 0; r < row_cnt; ++r) {
        uint32_t cat = 0, msg = 0, mg = 0, mi = 0;
        if (has_cat && (!host_bdat_read_u32(*tbl, r, v_cat, t_cat, cat) || cat > 1u)) continue;
        if (!host_bdat_read_u32(*tbl, r, v_msg, t_msg, msg) || msg == 0) continue;
        std::string nm;
        if (!host_bdat_read_str(*tbl, r, v_name, nm)) continue;
        auto pit = pos_by_name.find(nm);
        if (pit == pos_by_name.end()) continue;
        if (has_mg) host_bdat_read_u32(*tbl, r, v_mg, t_mg, mg);
        if (has_mi) host_bdat_read_u32(*tbl, r, v_mi, t_mi, mi);
        OffMapLandmark om;
        om.pos = pit->second;
        om.msg_id = msg;
        om.menu_group = mg;
        om.menu_map_image = mi;
        om.category = static_cast<uint8_t>(cat);
        parsed.push_back(om);
    }

    ws().offmap_lm_cache[res_lo] = parsed;
    out = std::move(parsed);
    return !out.empty();
}

bool mesh_find_offmap_npc_pos(const std::string& map_res, uint32_t npc_id, NavVec3& out_pos) {
    if (map_res.empty() || npc_id == 0) return false;
    const std::string res_lo = to_lower_ascii(map_res);

    std::lock_guard<std::mutex> lk(ws().mu);
    const auto* tbl = get_gmk_table_locked(res_lo + "_FLD_NpcPop");
    if (!tbl || tbl->size() < 0x20) return false;

    uint16_t v_name = 0, v_nid = 0;
    uint8_t t_name = 0, t_nid = 0;
    if (!host_bdat_find_col(*tbl, "name", v_name, t_name) ||
        !host_bdat_find_col(*tbl, "NpcID", v_nid, t_nid)) {
        return false;
    }

    std::string target_lvb_name;
    const uint16_t row_cnt = rd_u16(tbl->data() + 0x10);
    for (uint16_t r = 0; r < row_cnt; ++r) {
        uint32_t nid = 0;
        if (host_bdat_read_u32(*tbl, r, v_nid, t_nid, nid) && nid == npc_id) {
            if (host_bdat_read_str(*tbl, r, v_name, target_lvb_name)) break;
        }
    }
    if (target_lvb_name.empty()) return false;

    std::vector<LvbLandmark> npcs;
    if (!read_lvb_locked("/gmk/" + res_lo + "_npc.lvb", npcs)) return false;
    for (const auto& n : npcs) {
        if (n.name == target_lvb_name) {
            out_pos = n.pos;
            return true;
        }
    }
    return false;
}

bool read_archive_file(const std::string& lower_path, std::vector<uint8_t>& out) {
    out.clear();
    if (lower_path.empty()) return false;
    std::lock_guard<std::mutex> lk(ws().mu);
    if (!ws().lvb_arh.loaded && !ws().romfs_dir.empty()) {
        load_arh_index(ws().romfs_dir, ws().lvb_arh);
    }
    auto ait = ws().lvb_arh.by_path.find(lower_path);
    if (ait == ws().lvb_arh.by_path.end()) return false;
    return read_ard_entry(ws().lvb_arh.ard_path, ait->second, out);
}

}  // namespace xc2::nav
