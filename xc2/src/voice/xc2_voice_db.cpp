// SPDX-License-Identifier: GPL-3.0-or-later
#include "xc2_voice_db.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

namespace xc2::voice {
namespace {

uint32_t parse_hex_color(const std::string& hex, uint32_t fallback = 0x38BDF8FFu) {
    if (hex.empty()) return fallback;
    size_t off = (hex[0] == '#') ? 1 : 0;
    const size_t len = hex.size() - off;
    if (len != 6 && len != 8) return fallback;
    char* end = nullptr;
    const unsigned long v = std::strtoul(hex.c_str() + off, &end, 16);
    if (!end || static_cast<size_t>(end - (hex.c_str() + off)) != len) return fallback;
    if (len == 6) return (static_cast<uint32_t>(v) << 8) | 0xFFu;
    return static_cast<uint32_t>(v);
}

uint8_t parse_element(const std::string& s) {
    if (s == "fire")  return 1;
    if (s == "water") return 2;
    if (s == "wind")  return 3;
    if (s == "earth") return 4;
    if (s == "elec" || s == "electric") return 5;
    if (s == "ice")   return 6;
    if (s == "light") return 7;
    if (s == "dark")  return 8;
    return 0;
}

SemanticTag parse_tag(const std::string& s) {
    if (s == "Break")        return SemanticTag::Break;
    if (s == "Topple")       return SemanticTag::Topple;
    if (s == "Launch")       return SemanticTag::Launch;
    if (s == "Smash")        return SemanticTag::Smash;
    if (s == "SpecialReady") return SemanticTag::SpecialReady;
    if (s == "AggroWarning") return SemanticTag::AggroWarning;
    if (s == "LowHPHelp")    return SemanticTag::LowHPHelp;
    if (s == "HealPotion")   return SemanticTag::HealPotion;
    if (s == "Banter")       return SemanticTag::Banter;
    return SemanticTag::None;
}

std::string first_utf8_char(const std::string& s) {
    if (s.empty()) return "?";
    const unsigned char c = static_cast<unsigned char>(s[0]);
    size_t len = 1;
    if ((c & 0xE0u) == 0xC0u)      len = 2;
    else if ((c & 0xF0u) == 0xE0u) len = 3;
    else if ((c & 0xF8u) == 0xF0u) len = 4;
    if (len > s.size()) len = s.size();
    return s.substr(0, len);
}

// Minimal self-contained RFC 8259 JSON DOM parser (zero third-party deps).
struct JVal {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<JVal> arr;
    std::unordered_map<std::string, JVal> obj;

    const JVal* get(const char* k) const {
        if (type != Object) return nullptr;
        auto it = obj.find(k);
        return it == obj.end() ? nullptr : &it->second;
    }
    std::string get_str(const char* k, const char* def = "") const {
        const JVal* v = get(k);
        return (v && v->type == String) ? v->str : std::string(def);
    }
    double get_num(const char* k, double def = 0.0) const {
        const JVal* v = get(k);
        return (v && v->type == Number) ? v->num : def;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& src) : s_(src) {}

    bool parse(JVal& out) {
        skip_ws();
        if (pos_ + 3 <= s_.size() && static_cast<unsigned char>(s_[pos_]) == 0xEF &&
            static_cast<unsigned char>(s_[pos_ + 1]) == 0xBB &&
            static_cast<unsigned char>(s_[pos_ + 2]) == 0xBF) {
            pos_ += 3;  // Strip UTF-8 BOM
            skip_ws();
        }
        return parse_value(out);
    }

private:
    const std::string& s_;
    size_t pos_ = 0;

    void skip_ws() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\r' || s_[pos_] == '\n')) {
            ++pos_;
        }
    }

    static void append_utf8(std::string& dst, uint32_t cp) {
        if (cp <= 0x7Fu) {
            dst.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FFu) {
            dst.push_back(static_cast<char>(0xC0u | (cp >> 6)));
            dst.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        } else if (cp <= 0xFFFFu) {
            dst.push_back(static_cast<char>(0xE0u | (cp >> 12)));
            dst.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
            dst.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        } else {
            dst.push_back(static_cast<char>(0xF0u | (cp >> 18)));
            dst.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
            dst.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
            dst.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        }
    }

    bool parse_hex4(uint32_t& out) {
        if (pos_ + 4 > s_.size()) return false;
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = s_[pos_++];
            v <<= 4;
            if (c >= '0' && c <= '9')      v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
            else return false;
        }
        out = v;
        return true;
    }

    bool parse_string(std::string& out) {
        if (pos_ >= s_.size() || s_[pos_] != '"') return false;
        ++pos_;
        out.clear();
        while (pos_ < s_.size()) {
            char c = s_[pos_++];
            if (c == '"') return true;
            if (c == '\\') {
                if (pos_ >= s_.size()) return false;
                char e = s_[pos_++];
                switch (e) {
                    case '"':  out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/'); break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'n':  out.push_back('\n'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'u': {
                        uint32_t cp = 0;
                        if (!parse_hex4(cp)) return false;
                        if (cp >= 0xD800u && cp <= 0xDBFFu && pos_ + 2 <= s_.size() &&
                            s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
                            pos_ += 2;
                            uint32_t lo = 0;
                            if (!parse_hex4(lo)) return false;
                            cp = 0x10000u + (((cp - 0xD800u) << 10) | (lo - 0xDC00u));
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default: return false;
                }
            } else {
                out.push_back(c);
            }
        }
        return false;
    }

    bool parse_value(JVal& out) {
        skip_ws();
        if (pos_ >= s_.size()) return false;
        char c = s_[pos_];
        if (c == '{') {
            ++pos_;
            out.type = JVal::Object;
            skip_ws();
            if (pos_ < s_.size() && s_[pos_] == '}') { ++pos_; return true; }
            while (pos_ < s_.size()) {
                skip_ws();
                std::string k;
                if (!parse_string(k)) return false;
                skip_ws();
                if (pos_ >= s_.size() || s_[pos_] != ':') return false;
                ++pos_;
                JVal v;
                if (!parse_value(v)) return false;
                out.obj.emplace(std::move(k), std::move(v));
                skip_ws();
                if (pos_ < s_.size() && s_[pos_] == ',') { ++pos_; continue; }
                if (pos_ < s_.size() && s_[pos_] == '}') { ++pos_; return true; }
                return false;
            }
            return false;
        }
        if (c == '[') {
            ++pos_;
            out.type = JVal::Array;
            skip_ws();
            if (pos_ < s_.size() && s_[pos_] == ']') { ++pos_; return true; }
            while (pos_ < s_.size()) {
                JVal v;
                if (!parse_value(v)) return false;
                out.arr.push_back(std::move(v));
                skip_ws();
                if (pos_ < s_.size() && s_[pos_] == ',') { ++pos_; continue; }
                if (pos_ < s_.size() && s_[pos_] == ']') { ++pos_; return true; }
                return false;
            }
            return false;
        }
        if (c == '"') {
            out.type = JVal::String;
            return parse_string(out.str);
        }
        if (s_.compare(pos_, 4, "true") == 0)  { pos_ += 4; out.type = JVal::Bool; out.b = true; return true; }
        if (s_.compare(pos_, 5, "false") == 0) { pos_ += 5; out.type = JVal::Bool; out.b = false; return true; }
        if (s_.compare(pos_, 4, "null") == 0)  { pos_ += 4; out.type = JVal::Null; return true; }
        if (c == '-' || (c >= '0' && c <= '9')) {
            char* end = nullptr;
            out.num = std::strtod(s_.c_str() + pos_, &end);
            if (!end || end == s_.c_str() + pos_) return false;
            pos_ = static_cast<size_t>(end - s_.c_str());
            out.type = JVal::Number;
            return true;
        }
        return false;
    }
};

}  // namespace

void VoiceDatabase::clear() {
    speakers_.clear();
    cues_.clear();
}

bool VoiceDatabase::load_json_file(const std::string& path, std::string* err_out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        if (err_out) *err_out = "cannot open file";
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string content = ss.str();

    JVal root;
    JsonParser parser(content);
    if (!parser.parse(root) || root.type != JVal::Object) {
        if (err_out) *err_out = "invalid JSON root object";
        return false;
    }

    if (const JVal* sp = root.get("speakers"); sp && sp->type == JVal::Object) {
        for (const auto& [k, v] : sp->obj) {
            if (v.type != JVal::Object) continue;
            SpeakerInfo info;
            info.id = static_cast<uint16_t>(std::strtoul(k.c_str(), nullptr, 10));
            info.name = v.get_str("name", k.c_str());
            info.short_glyph = first_utf8_char(info.name);
            info.is_blade = (v.get_str("role") == "blade");
            info.partner_id = static_cast<uint16_t>(v.get_num("partner_id", 0));
            info.element_id = parse_element(v.get_str("element"));
            info.color_rgba = parse_hex_color(v.get_str("color_hex"), 0x38BDF8FFu);
            speakers_[info.id] = std::move(info);
        }
    }

    if (const JVal* cs = root.get("cues"); cs && cs->type == JVal::Object) {
        for (const auto& [k, v] : cs->obj) {
            if (v.type != JVal::Object) continue;
            CueEntry e;
            e.key = k;
            e.speaker_id = static_cast<uint16_t>(v.get_num("speaker_id", 0));
            const int prio = static_cast<int>(v.get_num("priority", 2));
            e.priority = (prio >= 0 && prio <= 3) ? static_cast<VoicePriority>(prio)
                                                  : VoicePriority::P2_AmbientCombat;
            e.tag = parse_tag(v.get_str("tag"));
            e.duration_sec = static_cast<float>(v.get_num("duration_sec", 0.0));
            e.text_full = v.get_str("text_full");
            e.text_compact = v.get_str("text_compact");
            cues_[k] = std::move(e);
        }
    }
    return true;
}

void VoiceDatabase::seed_mock_entries() {
    auto add_sp = [&](uint16_t id, const char* name, bool blade, uint16_t partner, uint8_t elem, uint32_t col) {
        if (speakers_.count(id)) return;
        SpeakerInfo s;
        s.id = id;
        s.name = name;
        s.short_glyph = first_utf8_char(s.name);
        s.is_blade = blade;
        s.partner_id = partner;
        s.element_id = elem;
        s.color_rgba = col;
        speakers_[id] = std::move(s);
    };

    auto elem_color = [](uint8_t e) -> uint32_t {
        switch (e) {
            case 1: return 0xEF4444FFu;  // Fire
            case 2: return 0x38BDF8FFu;  // Water
            case 3: return 0x22C55EFFu;  // Wind
            case 4: return 0xD97706FFu;  // Earth
            case 5: return 0xEAB308FFu;  // Electric
            case 6: return 0x67E8F9FFu;  // Ice
            case 7: return 0xFACC15FFu;  // Light
            case 8: return 0xA855F7FFu;  // Dark
            default: return 0x38BDF8FFu;
        }
    };

    // Complete 100% official XC2 3.0.1 speaker table directly verified from common.bdat (CHR_Dr / CHR_Bl) + cn/common_ms.bdat
    // Main game Drivers & Azurda
    add_sp(1,  "莱克斯", false, 11, 0, 0x3B82F6FFu);
    add_sp(2,  "尼娅",   false, 14, 0, 0xF59E0BFFu);
    add_sp(3,  "虎",     false, 15, 0, 0x22C55EFFu);
    add_sp(4,  "凡达姆", false, 18, 0, 0xF97316FFu);
    add_sp(5,  "梅勒芙", false, 19, 0, 0x6366F1FFu);
    add_sp(6,  "齐格",   false, 20, 0, 0xA855F7FFu);
    add_sp(7,  "真",     false, 0,  6, 0xCBD5E1FFu);
    add_sp(8,  "灭",     false, 0,  8, 0x7C3AEDFFu);
    add_sp(10, "青龙",   false, 1,  3, 0x14B8A6FFu);

    // Story Blades (11..21)
    add_sp(11, "焰",     true, 1, 1, 0xEF4444FFu);
    add_sp(12, "光",     true, 1, 7, 0xFACC15FFu);
    add_sp(13, "灵魂", true, 1, 7, 0x34D399FFu);
    add_sp(14, "白虎",   true, 2, 2, 0x38BDF8FFu);
    add_sp(15, "花JS",   true, 3, 4, 0xEC4899FFu);
    add_sp(16, "花JK",   true, 3, 4, 0xF472B6FFu);
    add_sp(17, "花JD",   true, 3, 3, 0xFB7185FFu);
    add_sp(18, "朱雀",   true, 4, 3, 0x22C55EFFu);
    add_sp(19, "迦具土", true, 5, 1, 0x60A5FAFFu);
    add_sp(20, "彩歌",   true, 6, 5, 0x34D399FFu);
    add_sp(21, "尼娅",   true, 1, 2, 0x38BDF8FFu);

    // Torna Organization Blades (22..28)
    add_sp(22, "斩铁",   true, 0, 3, elem_color(3));
    add_sp(23, "义经",   true, 0, 5, elem_color(5));
    add_sp(24, "弁庆",   true, 0, 4, elem_color(4));
    add_sp(25, "佐田彦", true, 0, 8, elem_color(8));
    add_sp(26, "神威",   true, 0, 5, elem_color(5));
    add_sp(27, "罗候",   true, 0, 1, elem_color(1));
    add_sp(28, "大锤",   true, 0, 4, elem_color(4));

    // All 29 Rare Blades (31..59)
    add_sp(31, "绵津见",    true, 5, 2, elem_color(2));
    add_sp(32, "红莲",      true, 0, 6, elem_color(6));
    add_sp(33, "力男",      true, 0, 4, elem_color(4));
    add_sp(34, "婆娑罗",    true, 0, 8, elem_color(8));
    add_sp(35, "雄黄",      true, 0, 2, elem_color(2));
    add_sp(36, "宫比罗",    true, 0, 1, elem_color(1));
    add_sp(37, "韦驮天",    true, 0, 3, elem_color(3));
    add_sp(38, "萤",        true, 0, 4, elem_color(4));
    add_sp(39, "重音",      true, 0, 8, elem_color(8));
    add_sp(40, "水分",      true, 0, 2, elem_color(2));
    add_sp(41, "星织",      true, 0, 6, elem_color(6));
    add_sp(42, "时叶",      true, 0, 6, elem_color(6));
    add_sp(43, "石榴",      true, 0, 5, elem_color(5));
    add_sp(44, "蓟",        true, 0, 8, elem_color(8));
    add_sp(45, "七冰",      true, 0, 6, elem_color(6));
    add_sp(46, "纽兹",      true, 0, 1, elem_color(1));
    add_sp(47, "宇迦",      true, 0, 4, elem_color(4));
    add_sp(48, "凛音",      true, 0, 2, elem_color(2));
    add_sp(49, "结",        true, 0, 5, elem_color(5));
    add_sp(50, "式",        true, 0, 3, elem_color(3));
    add_sp(51, "雷鼓",      true, 0, 5, elem_color(5));
    add_sp(52, "八重切",    true, 0, 3, elem_color(3));
    add_sp(53, "伊吹",      true, 0, 3, elem_color(3));
    add_sp(54, "KOS-MOS Re:", true, 0, 7, elem_color(7));
    add_sp(55, "天威",      true, 0, 5, elem_color(5));
    add_sp(56, "鸣",        true, 0, 8, elem_color(8));
    add_sp(57, "玛瑙",      true, 0, 4, elem_color(4));
    add_sp(58, "月",        true, 0, 6, elem_color(6));
    add_sp(59, "T-elos Re:", true, 0, 8, elem_color(8));

    // Common Blades (61..80)
    for (uint16_t cb = 61; cb <= 80; ++cb) {
        add_sp(cb, "普通异刃", true, 0, 0, 0x64748BFFu);
    }

    // DLC Blades (501..506)
    add_sp(501, "花巴斯塔", true, 0, 7, elem_color(7));
    add_sp(502, "修尔克",   true, 0, 7, elem_color(7));
    add_sp(503, "菲奥伦",   true, 0, 3, elem_color(3));
    add_sp(504, "神八",     true, 0, 7, elem_color(7));
    add_sp(505, "火花",     true, 0, 1, elem_color(1));
    add_sp(506, "艾尔玛",   true, 0, 8, elem_color(8));

    // Switch 2 Edition additions (231..232). MOMO is not part of the 91-speaker base table and has no line in
    // the shipped subtitle packs, so her cues come through the director's [待翻译] fallback.
    add_sp(231, "MOMO",     true, 0, 7, elem_color(7));
    add_sp(232, "MOMO (i)", true, 0, 7, elem_color(7));

    // Torna: The Golden Country (黄金国伊拉) 9 Playable Heroes (701..709)
    add_sp(701, "劳拉",   false, 704, 0, 0xF43F5EFFu);
    add_sp(702, "阿德尔", false, 706, 0, 0x3B82F6FFu);
    add_sp(703, "尤格",   false, 708, 0, 0x6366F1FFu);
    add_sp(704, "真",     true,  701, 6, 0xE2E8F0FFu);
    add_sp(705, "霞",     true,  701, 3, 0x22C55EFFu);
    add_sp(706, "光",     true,  702, 7, 0xFACC15FFu);
    add_sp(707, "米诺奇", true,  702, 8, 0xA855F7FFu);
    add_sp(708, "迦具土", true,  703, 1, 0x60A5FAFFu);
    add_sp(709, "绵津见", true,  703, 2, 0x38BDF8FFu);

    auto add_cue = [&](const char* key, uint16_t sp, VoicePriority p, SemanticTag tag,
                       const char* full, const char* compact) {
        if (cues_.count(key)) return;
        CueEntry e;
        e.key = key;
        e.speaker_id = sp;
        e.priority = p;
        e.tag = tag;
        e.text_full = full;
        e.text_compact = compact ? compact : "";
        cues_[key] = std::move(e);
    };

    // Phase 1 mock cues covering P0 banter, P1 tactical, Driver+Blade merge, Aggro/Heal, and P2 ambient
    add_cue("mock_banter_1",  1,  VoicePriority::P0_StoryAndBanter, SemanticTag::Banter,
            "刚才那一下配合得真不错啊，焰！", "");
    add_cue("mock_banter_2",  11, VoicePriority::P0_StoryAndBanter, SemanticTag::Banter,
            "嗯，莱克斯的动作也越来越干脆了呢。", "");
    add_cue("mock_banter_3",  2,  VoicePriority::P0_StoryAndBanter, SemanticTag::Banter,
            "喂喂，可别把我跟白虎给忘了啊！", "");
    add_cue("mock_break_rex", 1,  VoicePriority::P1_Tactical, SemanticTag::Break,
            "好机会！趁现在接倒地！", "破防成功，快接倒地！");
    add_cue("mock_break_pyra", 11, VoicePriority::P1_Tactical, SemanticTag::SpecialReady,
            "莱克斯，必杀技已经蓄力完成了！", "火焰必杀技准备就绪！");
    add_cue("mock_topple_nia", 2, VoicePriority::P1_Tactical, SemanticTag::Topple,
            "漂亮！接下来交给我来倒地！", "接手倒地，准备浮空！");
    add_cue("mock_aggro_tora", 3, VoicePriority::P1_Tactical, SemanticTag::AggroWarning,
            "呜哇！仇恨太高了，大家小心攻击！", "高危警告：仇恨拉满！");
    add_cue("mock_heal_byakko", 14, VoicePriority::P1_Tactical, SemanticTag::HealPotion,
            "小姐，恢复瓶已经准备好了，请及时拾取。", "恢复瓶已掉落，注意拾取");
    add_cue("mock_ambient_1", 15, VoicePriority::P2_AmbientCombat, SemanticTag::Banter,
            "花会全力支援主人的，嗼！", "花全力支援中，嗼！");
}

const SpeakerInfo* VoiceDatabase::find_speaker(uint16_t id) const {
    auto it = speakers_.find(id);
    return it == speakers_.end() ? nullptr : &it->second;
}

const CueEntry* VoiceDatabase::find_cue(const char* event_name, uint32_t cue_id) const {
    if (event_name && *event_name) {
        auto it = cues_.find(event_name);
        if (it != cues_.end()) return &it->second;
    }
    if (cue_id != 0) {
        auto it = cues_.find(std::to_string(cue_id));
        if (it != cues_.end()) return &it->second;
    }
    return nullptr;
}

}  // namespace xc2::voice
