// SPDX-License-Identifier: GPL-3.0-or-later
#include "xc2_voice_renderer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "nxrt/log.h"
#include "nxrt/play.h"
#include "../xc2_nav_mesh.h"

namespace xc2::voice {
namespace {

constexpr const char* kTag = "voice.ui";

// ---- Runtime XC2 Character Portrait Atlas (loaded lazily from player's own bf2.arh) ----
// Zero copyrighted assets in repo: decodes Tegra Block Linear BC3 directly from /menu/image/face.wilay
// (base game Drivers + Blades) and /menu/image/dlc3_btl_face.wilay (Torna DLC 9 playable heroes).
struct PortraitUV {
    uint64_t tex_id = 0;
    float u0 = 0.f, v0 = 0.f, u1 = 0.f, v1 = 0.f;
};

// speaker_id (the VoiceID column of common.bdat) -> /menu/image/face.wilay cell.
// The atlas is 8x26 cells of 256x256: 0..15 are the story party in the atlas' own order, 16..63 the
// blades, with the three Expansion Pass blades at 34..36. A speaker that is not listed falls back to
// the Core Crystal monogram, which is the correct look for the generic blades (61..80).
// Adding a blade is one line here and nothing else.
struct FaceCell {
    uint16_t speaker_id;
    uint8_t cell;
};

constexpr FaceCell kFaceCells[] = {
    // Drivers
    {1, 8},    // 莱克斯 Rex
    {2, 9},    // 尼娅 Nia
    {3, 11},   // 虎 Tora
    {4, 13},   // 凡达姆 Vandham
    {5, 12},   // 梅勒芙 Morag
    {6, 10},   // 齐格 Zeke
    {7, 14},   // 真 Jin
    {8, 15},   // 灭 Malos
    // Story blades (cells 0..7 and the four companion blades at 24..27)
    {11, 0},   // 焰 Pyra
    {12, 1},   // 光 Mythra
    {13, 2},   // 灵魂 Pneuma
    {14, 4},   // 白虎 Dromarch
    {15, 5},   // 花JS Poppi α
    {16, 6},   // 花JK Poppi QT
    {17, 7},   // 花JD Poppi QTπ
    {18, 26},  // 朱雀 Roc
    {19, 25},  // 迦具土 Brighid
    {20, 24},  // 彩歌 Pandoria
    {21, 3},   // 尼娅 (异刃)
    // Torna organisation blades (cells 19, 28..31)
    {22, 28},  // 斩铁
    {23, 17},  // 义经
    {24, 19},  // 弁庆
    {25, 38},  // 佐田彦
    {26, 31},  // 神威
    {27, 30},  // 罗候
    {28, 29},  // 大锤
    // Rare blades
    {31, 27},  // 绵津见 Aegaeon
    {32, 16},  // 红莲 Godfrey
    {35, 21},  // 雄黄 Wulfric
    {34, 18},  // 婆娑罗
    {36, 39},  // 宫比罗
    {37, 22},  // 韦驮天 Boreas
    {38, 20},  // 萤
    {39, 23},  // 重音 Kasandra
    {40, 40},  // 水分 Praxis
    {41, 41},  // 星织 Theory
    {42, 42},  // 时叶
    {43, 43},  // 石榴
    {44, 44},  // 蓟
    {45, 45},  // 七冰
    {46, 46},  // 纽兹
    {47, 47},  // 宇迦
    {48, 32},  // 凛音
    {49, 33},  // 结
    {50, 34},  // 式 Adenine
    {51, 35},  // 雷鼓
    {52, 36},  // 八重切 Zenobia
    {53, 37},  // 伊吹
};

// speaker_id -> /menu/image/mnu_face.wilay cell (BC7, 8x12 of 192x192). The Expansion Pass blades and the
// Switch 2 Edition additions have no cell in face.wilay at all, so this is the only portrait they have.
constexpr FaceCell kMenuFaceCells[] = {
    {54, 48},   // KOS-MOS Re:
    {501, 49},  // 花巴斯塔 Poppibuster
    {502, 51},  // 修尔克 Shulk
    {503, 52},  // 菲奥伦 Fiora
    {504, 53},  // 神八 Corvin
    {505, 54},  // 火花 Crossette
    {506, 55},  // 艾尔玛 Elma
    {231, 60},  // MOMO 第一形态
    {232, 63},  // MOMO 第二形态
};

int face_cell_for(uint16_t speaker_id) {
    for (const FaceCell& e : kFaceCells)
        if (e.speaker_id == speaker_id) return static_cast<int>(e.cell);
    return -1;
}

int menu_face_cell_for(uint16_t speaker_id) {
    for (const FaceCell& e : kMenuFaceCells)
        if (e.speaker_id == speaker_id) return static_cast<int>(e.cell);
    return -1;
}

struct PortraitAtlas {
    bool tried = false;
    uint64_t face_tex = 0;        // /menu/image/face.wilay (2048x6656, 8x26 cells of 256x256)
    uint32_t face_cols = 8;
    uint32_t face_rows = 26;
    uint64_t btl_face_tex = 0;    // /menu/image/btl_face.wilay (2048x2048, 8x8 cells of 256x256; cell 8 = Master Driver Rex)
    uint32_t btl_face_cols = 8;
    uint32_t btl_face_rows = 8;
    uint64_t torna_tex = 0;       // /menu/image/dlc3_btl_face.wilay (2048x512, 8x2 cells of 256x256)
    uint32_t torna_cols = 8;
    uint32_t torna_rows = 2;
    uint64_t mnu_face_tex = 0;    // /menu/image/mnu_face.wilay (1536x2304, BC7, 8x12 cells of 192x192)
    uint32_t mnu_face_cols = 8;   // the Expansion Pass and Switch 2 Edition blades only have a portrait here
    uint32_t mnu_face_rows = 12;

    static uint32_t rd32(const uint8_t* p) {
        uint32_t v = 0;
        std::memcpy(&v, p, 4);
        return v;
    }

    static bool untile_and_upload_wilay(const char* path, uint32_t cell, uint64_t& out_tex, uint32_t& out_cols,
                                       uint32_t& out_rows) {
        std::vector<uint8_t> raw;
        if (!xc2::nav::read_archive_file(path, raw) || raw.size() < 0x1040) return false;
        // Locate trailing LBIM footer (40 bytes at end of file)
        if (std::memcmp(raw.data() + raw.size() - 4, "LBIM", 4) != 0) return false;
        const uint8_t* lb = raw.data() + raw.size() - 0x28;
        const uint32_t img_sz = rd32(lb + 0x00);
        const uint32_t w      = rd32(lb + 0x08);
        const uint32_t h      = rd32(lb + 0x0C);
        const uint32_t fmt    = rd32(lb + 0x18);
        // 68 = RGBA_DXT5 (BC3), 77 = BPTC_UNORM (BC7): the menu face atlas is the only BC7 one of the four.
        if ((fmt != 68u && fmt != 77u) || w < cell || h < cell || (w & 3u) || (h & 3u)) return false;
        if (0x1000ull + img_sz > raw.size()) return false;

        const uint32_t bw = w / 4u;
        const uint32_t bh = h / 4u;
        const size_t lin_bytes = static_cast<size_t>(bw) * static_cast<size_t>(bh) * 16ull;
        if (img_sz < lin_bytes) return false;

        const uint32_t gobs_x = (bw * 16u + 63u) / 64u;
        uint32_t bh_gobs = 16u;
        while (bh_gobs > 1u && (bh_gobs / 2u) * 8u >= bh) bh_gobs /= 2u;
        const uint32_t gob_h = 8u * bh_gobs;
        const size_t block_bytes = 512ull * bh_gobs;

        const uint8_t* src = raw.data() + 0x1000;
        std::vector<uint8_t> lin(lin_bytes, 0);
        for (uint32_t y = 0; y < bh; ++y) {
            const uint32_t by = y / gob_h;
            const uint32_t gy = (y % gob_h) / 8u;
            const uint32_t ly = y & 7u;
            for (uint32_t bx = 0; bx < bw; ++bx) {
                const uint32_t x_byte = bx * 16u;
                const uint32_t gx = x_byte / 64u;
                const uint32_t lx = x_byte & 63u;
                const uint32_t gob_off =
                    ((lx & 0x20u) >> 5) * 256u +
                    ((ly & 0x06u) >> 1) * 64u +
                    ((lx & 0x10u) >> 4) * 32u +
                    (ly & 1u) * 16u +
                    (lx & 0x0Fu);
                const size_t src_off = (static_cast<size_t>(by) * gobs_x + gx) * block_bytes +
                                       static_cast<size_t>(gy) * 512ull + gob_off;
                if (src_off + 16ull <= img_sz) {
                    std::memcpy(lin.data() + (static_cast<size_t>(y) * bw + bx) * 16ull, src + src_off, 16);
                }
            }
        }

        out_tex = fmt == 77u ? nxrt::play::create_hud_texture_bc7(w, h, lin.data(), lin.size())
                             : nxrt::play::create_hud_texture_bc3(w, h, lin.data(), lin.size());
        if (!out_tex) return false;
        out_cols = w / cell;
        out_rows = h / cell;
        NXLOG_I(kTag, "loaded portrait atlas %s (%ux%u, %ux%u cells)", path, w, h, out_cols, out_rows);
        return true;
    }

    void ensure_loaded() {
        if (tried) return;
        tried = true;
        untile_and_upload_wilay("/menu/image/face.wilay", 256u, face_tex, face_cols, face_rows);
        untile_and_upload_wilay("/menu/image/btl_face.wilay", 256u, btl_face_tex, btl_face_cols, btl_face_rows);
        untile_and_upload_wilay("/menu/image/dlc3_btl_face.wilay", 256u, torna_tex, torna_cols, torna_rows);
        untile_and_upload_wilay("/menu/image/mnu_face.wilay", 192u, mnu_face_tex, mnu_face_cols, mnu_face_rows);
    }

    bool lookup(uint16_t speaker_id, bool rex_master_driver, PortraitUV& out) {
        ensure_loaded();
        // Auto-switch Rex (VoiceID=1) to Master Driver outfit portrait (btl_face.wilay cell 8)
        // once main scenario reaches >= 7060 (Ch.7 climax awakening / Ch.8+); zero spoilers before Ch.8.
        if (speaker_id == 1 && rex_master_driver && btl_face_tex && btl_face_cols > 0 && btl_face_rows > 0) {
            constexpr uint32_t kMasterRexCell = 8;
            const uint32_t col = kMasterRexCell % btl_face_cols;
            const uint32_t row = kMasterRexCell / btl_face_cols;
            const float pad_u = 0.03f / static_cast<float>(btl_face_cols);
            const float pad_v = 0.03f / static_cast<float>(btl_face_rows);
            out.tex_id = btl_face_tex;
            out.u0 = static_cast<float>(col) / static_cast<float>(btl_face_cols) + pad_u;
            out.v0 = static_cast<float>(row) / static_cast<float>(btl_face_rows) + pad_v;
            out.u1 = static_cast<float>(col + 1) / static_cast<float>(btl_face_cols) - pad_u;
            out.v1 = static_cast<float>(row + 1) / static_cast<float>(btl_face_rows) - pad_v;
            return true;
        }
        // Torna: The Golden Country (701..709) -> dlc3_btl_face.wilay (8x2 grid)
        if (speaker_id >= 701 && speaker_id <= 709 && torna_tex && torna_cols > 0 && torna_rows > 0) {
            const uint32_t idx = static_cast<uint32_t>(speaker_id - 701);
            const uint32_t col = idx % torna_cols;
            const uint32_t row = idx / torna_cols;
            // Slight 3% inset avoids bleeding from adjacent cells under linear filtering
            const float pad_u = 0.03f / static_cast<float>(torna_cols);
            const float pad_v = 0.03f / static_cast<float>(torna_rows);
            out.tex_id = torna_tex;
            out.u0 = static_cast<float>(col) / static_cast<float>(torna_cols) + pad_u;
            out.v0 = static_cast<float>(row) / static_cast<float>(torna_rows) + pad_v;
            out.u1 = static_cast<float>(col + 1) / static_cast<float>(torna_cols) - pad_u;
            out.v1 = static_cast<float>(row + 1) / static_cast<float>(torna_rows) - pad_v;
            return true;
        }
        // Expansion Pass / Switch 2 Edition blades -> mnu_face.wilay (8x12 grid of 192x192, BC7).
        // The artwork is a diamond inside each cell, so only its inscribed square is sampled (0.145 inset).
        if (const int mc = menu_face_cell_for(speaker_id);
            mc >= 0 && mnu_face_tex && mnu_face_cols > 0 && mnu_face_rows > 0) {
            const uint32_t idx = static_cast<uint32_t>(mc);
            const uint32_t col = idx % mnu_face_cols;
            const uint32_t row = idx / mnu_face_cols;
            if (row < mnu_face_rows) {
                constexpr float kInset = 0.145f;
                const float iu = kInset / static_cast<float>(mnu_face_cols);
                const float iv = kInset / static_cast<float>(mnu_face_rows);
                out.tex_id = mnu_face_tex;
                out.u0 = static_cast<float>(col) / static_cast<float>(mnu_face_cols) + iu;
                out.v0 = static_cast<float>(row) / static_cast<float>(mnu_face_rows) + iv;
                out.u1 = static_cast<float>(col + 1) / static_cast<float>(mnu_face_cols) - iu;
                out.v1 = static_cast<float>(row + 1) / static_cast<float>(mnu_face_rows) - iv;
                return true;
            }
        }
        // Base game Drivers & Story/Rare Blades -> face.wilay (8x26 grid)
        if (!face_tex || face_cols == 0 || face_rows == 0) return false;
        int cell = face_cell_for(speaker_id);
        if (cell < 0) return false;
        const uint32_t col = static_cast<uint32_t>(cell) % face_cols;
        const uint32_t row = static_cast<uint32_t>(cell) / face_cols;
        if (row >= face_rows) return false;
        const float pad_u = 0.03f / static_cast<float>(face_cols);
        const float pad_v = 0.03f / static_cast<float>(face_rows);
        out.tex_id = face_tex;
        out.u0 = static_cast<float>(col) / static_cast<float>(face_cols) + pad_u;
        out.v0 = static_cast<float>(row) / static_cast<float>(face_rows) + pad_v;
        out.u1 = static_cast<float>(col + 1) / static_cast<float>(face_cols) - pad_u;
        out.v1 = static_cast<float>(row + 1) / static_cast<float>(face_rows) - pad_v;
        return true;
    }
};

PortraitAtlas& portraits() {
    static PortraitAtlas s;
    return s;
}

const char* tag_label(SemanticTag t, VoiceUILang lang) {
    if (lang == VoiceUILang::En) {
        switch (t) {
            case SemanticTag::Break:        return "BREAK";
            case SemanticTag::Topple:       return "TOPPLE";
            case SemanticTag::Launch:       return "LAUNCH";
            case SemanticTag::Smash:        return "SMASH";
            case SemanticTag::SpecialReady: return "SPECIAL";
            case SemanticTag::AggroWarning: return "AGGRO";
            case SemanticTag::LowHPHelp:    return "LOW HP";
            case SemanticTag::HealPotion:   return "POTION";
            default:                        return nullptr;
        }
    }
    if (lang == VoiceUILang::Ja) {
        switch (t) {
            case SemanticTag::Break:        return "ブレイク";
            case SemanticTag::Topple:       return "ダウン";
            case SemanticTag::Launch:       return "ライジング";
            case SemanticTag::Smash:        return "スマッシュ";
            case SemanticTag::SpecialReady: return "必殺技";
            case SemanticTag::AggroWarning: return "ヘイト警告";
            case SemanticTag::LowHPHelp:    return "ピンチ";
            case SemanticTag::HealPotion:   return "回復ポット";
            default:                        return nullptr;
        }
    }
    switch (t) {
        case SemanticTag::Break:        return "破防";
        case SemanticTag::Topple:       return "倒地";
        case SemanticTag::Launch:       return "浮空";
        case SemanticTag::Smash:        return "猛击";
        case SemanticTag::SpecialReady: return "必杀就绪";
        case SemanticTag::AggroWarning: return "仇恨警告";
        case SemanticTag::LowHPHelp:    return "危机";
        case SemanticTag::HealPotion:   return "恢复瓶";
        default:                        return nullptr;
    }
}

uint32_t tag_badge_color(SemanticTag t) {
    switch (t) {
        case SemanticTag::Break:        return 0xEC4899FFu;  // Magenta-pink
        case SemanticTag::Topple:       return 0x22C55EFFu;  // Emerald
        case SemanticTag::Launch:       return 0xF59E0BFFu;  // Amber
        case SemanticTag::Smash:        return 0xEF4444FFu;  // Crimson
        case SemanticTag::SpecialReady: return 0xF5C542FFu;  // Affinity Gold
        case SemanticTag::AggroWarning:
        case SemanticTag::LowHPHelp:    return 0xEF4444FFu;  // Warning Red
        case SemanticTag::HealPotion:   return 0x10B981FFu;  // Mint Green
        default:                        return 0x56C8FAFFu;  // Sky Blue
    }
}

// Draw an authentic XC2 circular portrait medallion using the player's own in-game face atlas,
// falling back to a stylized Core-Crystal diamond monogram for speakers without a portrait cell.
void draw_portrait_medallion(const nxrt::gfx::HudCanvas& c, float cx, float cy, float radius,
                             uint16_t speaker_id, bool rex_master_driver,
                             uint32_t fill_rgba, uint32_t ring_rgba,
                             const std::string& glyph, float alpha) {
    PortraitUV uv;
    if (portraits().lookup(speaker_id, rex_master_driver, uv)) {
        // Outer dark drop shadow + obsidian backing plate
        c.circle_filled(cx, cy, radius + 3.0f, with_alpha(0x040A12D8u, alpha), 32);
        c.circle_filled(cx, cy, radius, with_alpha(0x0A1624F0u, alpha), 32);
        // Real in-game portrait cropped cleanly into the circular frame
        const float img_r = radius - 1.5f;
        c.image_rounded(uv.tex_id, cx - img_r, cy - img_r, cx + img_r, cy + img_r,
                        uv.u0, uv.v0, uv.u1, uv.v1, img_r, with_alpha(0xFFFFFFFFu, alpha));
        // Double-layer XC2 Aether ring (crisp colored outer rim + subtle white inner bevel)
        c.circle(cx, cy, radius + 0.8f, with_alpha(ring_rgba, alpha * 0.95f), std::max(1.8f, radius * 0.085f), 32);
        c.circle(cx, cy, radius - 1.4f, with_alpha(0xE6F7FF58u, alpha), 1.0f, 32);
        // Tiny Core-Crystal diamond tick at top of ring (signature XC2 motif)
        const float dr = std::max(2.6f, radius * 0.14f);
        const float dy = cy - radius - 0.8f;
        c.quad_filled(cx, dy - dr, cx + dr, dy, cx, dy + dr, cx - dr, dy, with_alpha(ring_rgba, alpha));
        return;
    }

    // Fallback: Core-Crystal 45° diamond monogram for generic/unmapped Blades
    const float half_diag = radius * 1.02f;
    const float outer[8] = {
        cx,             cy - half_diag,
        cx + half_diag, cy,
        cx,             cy + half_diag,
        cx - half_diag, cy,
    };
    const float inner_r = half_diag * 0.80f;
    const float inner[8] = {
        cx,           cy - inner_r,
        cx + inner_r, cy,
        cx,           cy + inner_r,
        cx - inner_r, cy,
    };
    c.convex_poly_filled(outer, 4, with_alpha(0x07111EE8u, alpha));
    c.convex_poly_filled(inner, 4, with_alpha(fill_rgba, alpha * 0.88f));
    c.triangle_filled(cx, cy - inner_r, cx + inner_r * 0.55f, cy - inner_r * 0.15f,
                      cx - inner_r * 0.85f, cy, with_alpha(0xFFFFFF44u, alpha));
    c.polyline(outer, 4, with_alpha(ring_rgba, alpha), true, std::max(1.5f, half_diag * 0.085f));
    c.polyline(inner, 4, with_alpha(0xFFFFFF48u, alpha), true, 1.0f);

    if (!glyph.empty()) {
        const float font_sz = std::max(13.0f, half_diag * 0.96f);
        float tw = 0.f, th = 0.f;
        c.measure_text(font_sz, glyph.c_str(), &tw, &th);
        c.text_outlined(cx - tw * 0.5f, cy - th * 0.5f, font_sz,
                        with_alpha(0xFFFFFFFFu, alpha),
                        with_alpha(0x000000D8u, alpha),
                        glyph.c_str());
    }
}

}  // namespace

VoiceRenderer::SafeRect VoiceRenderer::compute_safe_rect(uint32_t vp_w, uint32_t vp_h) {
    SafeRect sr;
    const float fw = static_cast<float>(vp_w);
    const float fh = static_cast<float>(vp_h);
    sr.h = fh;
    sr.w = std::min(fw, fh * (16.0f / 9.0f));
    sr.x = (fw - sr.w) * 0.5f;
    sr.y = 0.0f;
    sr.scale = std::max(0.55f, sr.h / 1080.0f);
    return sr;
}

void VoiceRenderer::draw_diamond_avatar(const nxrt::gfx::HudCanvas& c, float cx, float cy,
                                        float half_diag, uint32_t fill_rgba, uint32_t border_rgba,
                                        const std::string& glyph, float alpha) {
    draw_portrait_medallion(c, cx, cy, half_diag, 0, false, fill_rgba, border_rgba, glyph, alpha);
}

void VoiceRenderer::render_combat_strip(const nxrt::gfx::HudCanvas& c, const SafeRect& sr,
                                        const DirectorRenderState& st) const {
    if (st.combat_cards.empty()) return;

    // Larger, high-legibility left-side XC2 tactical comms cards anchored below party HP HUD
    const float base_x   = sr.x + sr.w * 0.022f;
    const float base_y   = sr.y + sr.h * 0.39f;
    const float min_w    = sr.w * 0.285f;
    const float max_w    = sr.w * 0.425f;
    const float row_h    = sr.h * 0.068f;  // ~73px @ 1080p (up from 52px)
    const float gap      = sr.h * 0.014f;
    const float chamfer  = 11.0f * sr.scale;
    const float border_w = std::max(1.5f, 1.8f * sr.scale);

    float cur_y = base_y;
    for (const auto& card : st.combat_cards) {
        const float a = card.alpha * st.global_alpha;
        if (a <= 0.005f) continue;

        const bool merged = (card.partner_speaker_id != 0 && !card.text_line2.empty());
        const float card_h = merged ? (row_h * 1.44f) : row_h;
        const float y0 = cur_y + card.y_offset * (row_h + gap);
        const float y1 = y0 + card_h;

        const float name_sz = std::max(14.0f, 17.5f * sr.scale);
        const float body_sz = std::max(16.0f, 21.0f * sr.scale);

        // Dynamically size card width to fit text cleanly without truncation
        float l1_w = 0.f, l1_h = 0.f, l2_w = 0.f, l2_h = 0.f;
        c.measure_text(body_sz, card.text_line1.c_str(), &l1_w, &l1_h);
        if (merged) c.measure_text(body_sz * 0.94f, card.text_line2.c_str(), &l2_w, &l2_h);
        const float text_left_pad = (merged ? 92.0f : 78.0f) * sr.scale;
        const float card_w = std::clamp(text_left_pad + std::max(l1_w, l2_w) + 56.0f * sr.scale, min_w, max_w);

        const float x0 = base_x + 12.0f * sr.scale;  // room for overlapping portrait medallion
        const float x1 = x0 + card_w;

        // 45-degree chamfered XC2 obsidian-navy glass plate
        const float poly[12] = {
            x0 + chamfer, y0,
            x1,           y0,
            x1,           y1 - chamfer,
            x1 - chamfer, y1,
            x0,           y1,
            x0,           y0 + chamfer,
        };
        // Smooth two-zone horizontal gradient: rich dark obsidian on left, airy fade-out on right
        const float x_mid = x0 + card_w * 0.68f;
        c.convex_poly_hgrad(poly, 6, x0, x1, with_alpha(0x06101CEBu, a), with_alpha(0x06101C00u, a));
        // Subtle inner top specular sheen
        c.rect_filled_hgrad(x0 + chamfer, y0 + border_w, x_mid, y0 + border_w + 1.5f * sr.scale,
                            with_alpha(0xBFE3F728u, a), with_alpha(0xBFE3F700u, a));

        // Dynamic Aether border color (Launcher Sky Blue default, Gold Affinity, or Warning Red)
        uint32_t stroke_col = 0x56C8FAEEu;
        if (card.semantic_tag == SemanticTag::AggroWarning ||
            card.semantic_tag == SemanticTag::LowHPHelp) {
            stroke_col = 0xEF4444F5u;
        } else if (card.is_gold_affinity) {
            stroke_col = 0xF5C542F5u;
        }

        // Crisp left-side glowing accent bar + top-left chamfered rail
        const float accent_edge[8] = {
            x0 + card_w * 0.62f, y0,
            x0 + chamfer,        y0,
            x0,                  y0 + chamfer,
            x0,                  y1,
        };
        c.polyline(accent_edge, 4, with_alpha(stroke_col, a * 0.92f), false, border_w);
        // Bottom fading Aether hairline
        c.rect_filled_hgrad(x0, y1 - border_w, x1 - chamfer, y1,
                            with_alpha(stroke_col, a * 0.78f),
                            with_alpha(stroke_col, 0.f));
        // Subtle inner vertical speaker-color accent strip right behind portrait
        c.rect_filled(x0 + 2.0f * sr.scale, y0 + chamfer + 2.0f * sr.scale,
                      x0 + 5.5f * sr.scale, y1 - 4.0f * sr.scale,
                      with_alpha(card.primary_color_rgba, a * 0.85f), 1.5f * sr.scale);

        // Oversized circular in-game character portrait slightly overlapping the left/top edge (XC2 party HUD style)
        const float av_r  = 29.0f * sr.scale;  // 58px diameter @ 1080p
        const float av_cx = x0 + 26.0f * sr.scale;
        const float av_cy = (y0 + y1) * 0.5f - (merged ? 6.0f * sr.scale : 2.0f * sr.scale);
        draw_portrait_medallion(c, av_cx, av_cy, av_r, card.primary_speaker_id,
                                st.rex_master_driver,
                                card.primary_color_rgba, stroke_col, card.primary_glyph, a);

        if (merged) {
            const float sub_r = 18.0f * sr.scale;
            draw_portrait_medallion(c, av_cx + av_r * 0.76f, av_cy + av_r * 0.64f, sub_r,
                                    card.partner_speaker_id, st.rex_master_driver,
                                    card.partner_color_rgba,
                                    0xF5C542F2u, card.partner_glyph, a);
        }

        // Speaker header + semantic badge + crisp dialogue body
        const float tx = x0 + text_left_pad;
        const float header_y = y0 + 8.0f * sr.scale;

        std::string header = card.primary_name;
        if (merged && !card.partner_name.empty()) {
            header += "  &  " + card.partner_name;
        }
        c.text_outlined(tx, header_y, name_sz,
                        with_alpha(card.primary_color_rgba, a),
                        with_alpha(0x000000E0u, a),
                        header.c_str());

        if (const char* tl = tag_label(card.semantic_tag, st.ui_lang)) {
            float nw = 0.f, nh = 0.f;
            c.measure_text(name_sz, header.c_str(), &nw, &nh);
            const float px = tx + nw + 10.0f * sr.scale;
            const float pill_sz = std::max(12.0f, 13.5f * sr.scale);
            float pw = 0.f, ph = 0.f;
            c.measure_text(pill_sz, tl, &pw, &ph);
            const float pad_x = 7.0f * sr.scale;
            const float pad_y = 2.0f * sr.scale;
            const uint32_t bc = tag_badge_color(card.semantic_tag);
            const float bx0 = px, by0 = header_y + (nh - ph) * 0.5f - pad_y;
            const float bx1 = px + pw + pad_x * 2.f, by1 = by0 + ph + pad_y * 2.f;
            const float bcut = 4.0f * sr.scale;
            c.quad_filled(bx0 + bcut, by0, bx1, by0, bx1 - bcut, by1, bx0, by1, with_alpha(bc, a * 0.34f));
            c.line(bx0 + bcut, by0, bx1, by0, with_alpha(bc, a * 0.92f), 1.2f * sr.scale);
            c.text_outlined(bx0 + pad_x, by0 + pad_y, pill_sz, with_alpha(0xFFFFFFFFu, a), with_alpha(0x000000C8u, a), tl);
        }

        // Subtle divider hairline between speaker name and dialogue text
        const float sep_y = header_y + name_sz + 3.0f * sr.scale;
        c.rect_filled_hgrad(tx, sep_y, x1 - 24.0f * sr.scale, sep_y + 1.0f,
                            with_alpha(stroke_col, a * 0.38f), with_alpha(stroke_col, 0.f));

        // Line 1 text
        const float l1_y = sep_y + 6.0f * sr.scale;
        c.text_outlined(tx, l1_y, body_sz,
                        with_alpha(0xF8FAFCFFu, a),
                        with_alpha(0x000000E8u, a),
                        card.text_line1.c_str());

        // Line 2 text (Driver + Blade merged response)
        if (merged) {
            const float l2_y = l1_y + 27.0f * sr.scale;
            c.text_outlined(tx, l2_y, body_sz * 0.94f,
                            with_alpha(0xFDE68AFFu, a),
                            with_alpha(0x000000E8u, a),
                            card.text_line2.c_str());
        }

        if (!card.exiting) {
            cur_y += card_h + gap;
        }
    }
}

void VoiceRenderer::render_exploration_bar(const nxrt::gfx::HudCanvas& c, const SafeRect& sr,
                                           const DirectorRenderState& st) const {
    if (st.exploration_lines.empty()) return;

    const auto& cur_line = st.exploration_lines.back();
    const float cur_a = cur_line.alpha * st.global_alpha;
    if (cur_a <= 0.005f) return;

    const float cx = sr.x + sr.w * 0.5f;
    // Native XC2 bottom dialogue band (lifts slightly if bottom nav hint is active)
    const float box_h = 84.0f * sr.scale;
    const float y0 = sr.y + sr.h * (st.nav_bottom_busy ? 0.74f : 0.80f);
    const float y1 = y0 + box_h;

    const float name_sz = std::max(15.5f, 19.5f * sr.scale);
    const float body_sz = std::max(18.0f, 23.5f * sr.scale);

    // Measure dialogue width so the XC2 dialogue plate hugs the text naturally
    float bw = 0.f, bh = 0.f, nw = 0.f, nh = 0.f;
    c.measure_text(body_sz, cur_line.text_line1.c_str(), &bw, &bh);
    c.measure_text(name_sz, cur_line.primary_name.c_str(), &nw, &nh);
    const float portrait_space = 96.0f * sr.scale;
    const float content_w = std::max(bw, nw + 40.0f * sr.scale);
    const float plate_w = std::clamp(portrait_space + content_w + 54.0f * sr.scale,
                                     sr.w * 0.38f, sr.w * 0.60f);
    const float x0 = cx - plate_w * 0.5f;
    const float x1 = cx + plate_w * 0.5f;
    const float wing_w = 48.0f * sr.scale;
    const float chamfer = 12.0f * sr.scale;

    // 1. Soft feathered side wings + central 45° chamfered XC2 obsidian-navy dialogue plate
    c.rect_filled_hgrad(x0 - wing_w, y0, x0 + chamfer, y1,
                        with_alpha(0x06101C00u, cur_a), with_alpha(0x06101CD8u, cur_a));
    c.rect_filled_hgrad(x1 - chamfer, y0, x1 + wing_w, y1,
                        with_alpha(0x06101CD8u, cur_a), with_alpha(0x06101C00u, cur_a));
    const float plate[12] = {
        x0 + chamfer, y0,
        x1,           y0,
        x1,           y1 - chamfer,
        x1 - chamfer, y1,
        x0,           y1,
        x0,           y0 + chamfer,
    };
    c.convex_poly_filled(plate, 6, with_alpha(0x071220E2u, cur_a));

    // 2. Twin XC2 Aether horizontal rails (sky-blue + warm pale gold shimmer fading at both wings)
    const float rail_h = std::max(1.4f, 1.8f * sr.scale);
    const uint32_t top_col = 0x56C8FAEEu;   // Sky Blue Aether
    const uint32_t bot_col = 0xE8D28CCCu;   // XC2 Pale Gold
    c.rect_filled_hgrad(x0 - wing_w * 0.8f, y0, cx, y0 + rail_h,
                        with_alpha(top_col, 0.f), with_alpha(top_col, cur_a));
    c.rect_filled_hgrad(cx, y0, x1 + wing_w * 0.8f, y0 + rail_h,
                        with_alpha(top_col, cur_a), with_alpha(top_col, 0.f));
    c.rect_filled_hgrad(x0 - wing_w * 0.7f, y1 - rail_h, cx, y1,
                        with_alpha(bot_col, 0.f), with_alpha(bot_col, cur_a * 0.85f));
    c.rect_filled_hgrad(cx, y1 - rail_h, x1 + wing_w * 0.7f, y1,
                        with_alpha(bot_col, cur_a * 0.85f), with_alpha(bot_col, 0.f));

    // Tiny Core-Crystal diamond finials at the right edge of the dialogue plate
    const float fx = x1 - 14.0f * sr.scale;
    const float fy = (y0 + y1) * 0.5f;
    const float fr = 4.2f * sr.scale;
    c.quad_filled(fx, fy - fr, fx + fr, fy, fx, fy + fr, fx - fr, fy, with_alpha(0x56C8FAA8u, cur_a));

    // 3. Real in-game character portrait medallion on the left (breaking slightly above top rail!)
    const float av_r  = 36.0f * sr.scale;  // 72px diameter @ 1080p
    const float av_cx = x0 + 44.0f * sr.scale;
    const float av_cy = y0 + box_h * 0.44f;
    draw_portrait_medallion(c, av_cx, av_cy, av_r, cur_line.primary_speaker_id,
                            st.rex_master_driver,
                            cur_line.primary_color_rgba,
                            cur_line.primary_color_rgba, cur_line.primary_glyph, cur_a);

    // 4. Speaker nameplate + subtle speaker-color underline + crisp dialogue body
    const float tx = x0 + portrait_space;
    const float name_y = y0 + 11.0f * sr.scale;
    c.text_outlined(tx, name_y, name_sz,
                    with_alpha(cur_line.primary_color_rgba, cur_a),
                    with_alpha(0x000000E4u, cur_a),
                    cur_line.primary_name.c_str());
    const float ul_y = name_y + nh + 3.0f * sr.scale;
    c.rect_filled_hgrad(tx, ul_y, tx + std::max(nw + 36.0f * sr.scale, 130.0f * sr.scale), ul_y + 1.4f * sr.scale,
                        with_alpha(cur_line.primary_color_rgba, cur_a * 0.85f),
                        with_alpha(cur_line.primary_color_rgba, 0.f));

    const float body_y = ul_y + 8.0f * sr.scale;
    c.text_outlined(tx, body_y, body_sz,
                    with_alpha(0xF8FAFCFFu, cur_a),
                    with_alpha(0x000000ECu, cur_a),
                    cur_line.text_line1.c_str());

    // 5. Previous banter line (two-character post-battle exchange) floating cleanly above the top rail
    if (st.exploration_lines.size() >= 2) {
        const auto& prev = st.exploration_lines[0];
        const float pa = prev.alpha * st.global_alpha * 0.68f;
        if (pa > 0.01f) {
            const float prev_sz = body_sz * 0.78f;
            const std::string p_name = prev.primary_name + (st.ui_lang == VoiceUILang::En ? ": " : "：");
            float pnw = 0.f, pnh = 0.f, pbw = 0.f, pbh = 0.f;
            c.measure_text(prev_sz, p_name.c_str(), &pnw, &pnh);
            c.measure_text(prev_sz, prev.text_line1.c_str(), &pbw, &pbh);
            const float total_pw = pnw + pbw + 24.0f * sr.scale;
            const float px0 = cx - total_pw * 0.5f;
            const float py0 = y0 - pbh - 14.0f * sr.scale;
            const float pcut = 6.0f * sr.scale;
            c.quad_filled(px0 + pcut, py0 - 3.f * sr.scale,
                          px0 + total_pw, py0 - 3.f * sr.scale,
                          px0 + total_pw - pcut, py0 + pbh + 4.f * sr.scale,
                          px0, py0 + pbh + 4.f * sr.scale,
                          with_alpha(0x06101CB0u, pa));
            c.text_outlined(px0 + 12.0f * sr.scale, py0, prev_sz,
                            with_alpha(prev.primary_color_rgba, pa),
                            with_alpha(0x000000D0u, pa), p_name.c_str());
            c.text_outlined(px0 + 12.0f * sr.scale + pnw, py0, prev_sz,
                            with_alpha(0xE2E8F0FFu, pa),
                            with_alpha(0x000000D0u, pa), prev.text_line1.c_str());
        }
    }
}

void VoiceRenderer::render(const nxrt::gfx::HudCanvas& c, const DirectorRenderState& st) const {
    if (c.width == 0 || c.height == 0 || !c.draw_list) return;
    if (st.mode == UIVisibilityMode::Hidden || st.global_alpha <= 0.005f) return;

    const SafeRect sr = compute_safe_rect(c.width, c.height);
    // Render whichever track has active items so cards/banter never vanish across the 0.2s Combat<->Exploration transition.
    if (!st.combat_cards.empty()) {
        render_combat_strip(c, sr, st);
    }
    if (!st.exploration_lines.empty()) {
        render_exploration_bar(c, sr, st);
    }
}

void VoiceRenderer::render_voice_log(const nxrt::gfx::HudCanvas& c,
                                     const std::deque<VoiceLogEntry>& history,
                                     VoiceUILang ui_lang) const {
    if (c.width == 0 || c.height == 0 || !c.draw_list) return;
    const SafeRect sr = compute_safe_rect(c.width, c.height);

    const float panel_w = sr.w * 0.30f;
    const float panel_h = sr.h * 0.56f;
    const float x1 = sr.x + sr.w * 0.978f;
    const float x0 = x1 - panel_w;
    const float y0 = sr.y + sr.h * 0.16f;
    const float y1 = y0 + panel_h;
    const float chamfer = 10.0f * sr.scale;

    const float poly[12] = {
        x0 + chamfer, y0,
        x1,           y0,
        x1,           y1 - chamfer,
        x1 - chamfer, y1,
        x0,           y1,
        x0,           y0 + chamfer,
    };
    c.convex_poly_filled(poly, 6, 0x071220E8u);
    c.polyline(poly, 6, 0x56C8FAC8u, true, std::max(1.4f, 1.6f * sr.scale));

    // Header
    const float hdr_sz = std::max(15.0f, 18.0f * sr.scale);
    const float pad_x = 16.0f * sr.scale;
    const float hdr_y = y0 + 12.0f * sr.scale;
    char hdr[96];
    if (ui_lang == VoiceUILang::En) {
        std::snprintf(hdr, sizeof(hdr), "Voice Log (%zu/30)  [Ctrl+F5 Close]", history.size());
    } else if (ui_lang == VoiceUILang::Ja) {
        std::snprintf(hdr, sizeof(hdr), "ボイス履歴 (%zu/30)  [Ctrl+F5 閉じる]", history.size());
    } else {
        std::snprintf(hdr, sizeof(hdr), "语音记录 Voice Log (%zu/30)  [Ctrl+F5 关闭]", history.size());
    }
    c.text_outlined(x0 + pad_x, hdr_y, hdr_sz, 0xF5C542FFu, 0x000000E0u, hdr);

    const float div_y = hdr_y + 26.0f * sr.scale;
    c.rect_filled_hgrad(x0 + pad_x, div_y, x1 - pad_x, div_y + std::max(1.2f, 1.4f * sr.scale),
                        0x56C8FAB8u, 0x56C8FA10u);

    const float row_sz = std::max(13.0f, 16.0f * sr.scale);
    const float row_h = 26.0f * sr.scale;
    float cur_y = div_y + 10.0f * sr.scale;

    if (history.empty()) {
        const char* empty_msg = (ui_lang == VoiceUILang::En)
                                    ? "No party voice lines recorded yet."
                                    : (ui_lang == VoiceUILang::Ja ? "記録されたパーティボイスはまだありません。"
                                                                  : "暂无已记录的队友语音。");
        c.text(x0 + pad_x, cur_y, row_sz, 0x94A3B8CCu, empty_msg);
        return;
    }

    const char* colon = (ui_lang == VoiceUILang::En) ? ": " : "：";
    const size_t max_rows = static_cast<size_t>(std::max(1.0f, (y1 - cur_y - 12.0f * sr.scale) / row_h));
    const size_t start = (history.size() > max_rows) ? (history.size() - max_rows) : 0;
    for (size_t i = start; i < history.size(); ++i) {
        const auto& e = history[i];
        const std::string sp = e.speaker_name + colon;
        float sw = 0.f, sh = 0.f;
        c.measure_text(row_sz, sp.c_str(), &sw, &sh);
        c.text_outlined(x0 + pad_x, cur_y, row_sz, e.color_rgba, 0x000000D0u, sp.c_str());
        c.text_outlined(x0 + pad_x + sw + 2.0f * sr.scale, cur_y, row_sz,
                        0xF1F5F9FFu, 0x000000D0u, e.text.c_str());
        cur_y += row_h;
    }
}

}  // namespace xc2::voice
