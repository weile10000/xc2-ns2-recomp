// SPDX-License-Identifier: GPL-3.0-or-later
// SPIR-V rewrites (declarations in nvn_spirv_patch.h).
#include "nvn_spirv_patch.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <string>

namespace nxrt::nvn::be::gpu {
namespace {

enum : uint32_t {
    OpEntryPoint = 15, OpTypeVoid = 19, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23, OpTypePointer = 32, OpTypeForwardPointer = 39,
    OpFunction = 54, OpVariable = 59, OpDecorate = 71,
    DecorationLocation = 30, StorageOutput = 3,
};

struct Inst {
    size_t at;
    uint32_t op, len;
};

bool is_type_or_constant(uint32_t op) { return (op >= OpTypeVoid && op <= OpTypeForwardPointer) || (op >= 41 && op <= 52); }

}  // namespace

bool spirv_add_outputs(const std::vector<uint32_t>& in, const std::vector<SpvOutputSlot>& slots, std::vector<uint32_t>& out) {
    if (slots.empty() || in.size() < 5) return false;
    std::vector<Inst> insts;
    for (size_t i = 5; i < in.size();) {
        const uint32_t len = in[i] >> 16;
        if (!len || i + len > in.size()) return false;
        insts.push_back({i, in[i] & 0xffff, len});
        i += len;
    }

    uint32_t bound = in[3];
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> ints;    // (width, signed) -> id
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> vecs;    // (component type, count) -> id
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> ptrs;    // (storage class, pointee) -> id
    std::map<uint32_t, uint32_t> output_location;              // variable id -> Location
    std::map<uint32_t, uint32_t> output_vars;                  // output variable id -> pointee type id
    uint32_t f32 = 0;
    size_t first_type = 0, first_function = 0, entry_at = 0;
    for (const Inst& n : insts) {
        const uint32_t* w = &in[n.at];
        if (!first_type && is_type_or_constant(n.op)) first_type = n.at;
        if (!first_function && n.op == OpFunction) first_function = n.at;
        if (n.op == OpEntryPoint && !entry_at) entry_at = n.at;
        if (n.op == OpTypeInt && n.len == 4) ints[{w[2], w[3]}] = w[1];
        else if (n.op == OpTypeFloat && n.len >= 3 && w[2] == 32) f32 = w[1];
        else if (n.op == OpTypeVector && n.len == 4) vecs[{w[2], w[3]}] = w[1];
        else if (n.op == OpTypePointer && n.len == 4) ptrs[{w[2], w[3]}] = w[1];
        else if (n.op == OpDecorate && n.len >= 4 && w[2] == DecorationLocation) output_location[w[1]] = w[3];
    }
    if (!first_type || !first_function || !entry_at || !f32) return false;
    for (const Inst& n : insts)
        if (n.op == OpVariable && n.len >= 4 && in[n.at + 3] == StorageOutput) output_vars[in[n.at + 2]] = in[n.at + 1];
    for (const SpvOutputSlot& s : slots)
        for (const auto& kv : output_location)
            if (output_vars.count(kv.first) && kv.second == s.location) return false;   // already declared (as a single variable)

    std::vector<uint32_t> decos, decls, new_vars;
    auto scalar_id = [&](SpvScalar t) -> uint32_t {
        if (t == SpvScalar::Float) return f32;
        const std::pair<uint32_t, uint32_t> k = {32, t == SpvScalar::Sint ? 1u : 0u};
        auto it = ints.find(k);
        if (it != ints.end()) return it->second;
        const uint32_t id = bound++;
        decls.insert(decls.end(), {(4u << 16) | OpTypeInt, id, 32u, k.second});
        ints[k] = id;
        return id;
    };
    auto vec4_id = [&](uint32_t scalar) -> uint32_t {
        const std::pair<uint32_t, uint32_t> k = {scalar, 4};
        auto it = vecs.find(k);
        if (it != vecs.end()) return it->second;
        const uint32_t id = bound++;
        decls.insert(decls.end(), {(4u << 16) | OpTypeVector, id, scalar, 4u});
        vecs[k] = id;
        return id;
    };
    auto out_ptr_id = [&](uint32_t pointee) -> uint32_t {
        const std::pair<uint32_t, uint32_t> k = {StorageOutput, pointee};
        auto it = ptrs.find(k);
        if (it != ptrs.end()) return it->second;
        const uint32_t id = bound++;
        decls.insert(decls.end(), {(4u << 16) | OpTypePointer, id, StorageOutput, pointee});
        ptrs[k] = id;
        return id;
    };
    for (const SpvOutputSlot& s : slots) {
        const uint32_t ptr = out_ptr_id(vec4_id(scalar_id(s.type)));
        const uint32_t var = bound++;
        decls.insert(decls.end(), {(4u << 16) | OpVariable, ptr, var, StorageOutput});
        decos.insert(decos.end(), {(4u << 16) | OpDecorate, var, DecorationLocation, s.location});
        new_vars.push_back(var);
    }

    out.clear();
    out.reserve(in.size() + decos.size() + decls.size() + new_vars.size() + 4);
    out.insert(out.end(), in.begin(), in.begin() + 5);
    out[3] = bound;
    const Inst* entry = nullptr;
    for (const Inst& n : insts)
        if (n.at == entry_at) entry = &n;
    for (const Inst& n : insts) {
        if (n.at == first_type) out.insert(out.end(), decos.begin(), decos.end());
        if (n.at == first_function) out.insert(out.end(), decls.begin(), decls.end());
        if (&n == entry) {
            // the interface list is the tail of OpEntryPoint
            out.push_back(((n.len + static_cast<uint32_t>(new_vars.size())) << 16) | OpEntryPoint);
            out.insert(out.end(), in.begin() + static_cast<std::ptrdiff_t>(n.at) + 1, in.begin() + static_cast<std::ptrdiff_t>(n.at + n.len));
            out.insert(out.end(), new_vars.begin(), new_vars.end());
            continue;
        }
        out.insert(out.end(), in.begin() + static_cast<std::ptrdiff_t>(n.at), in.begin() + static_cast<std::ptrdiff_t>(n.at + n.len));
    }
    return true;
}

}  // namespace nxrt::nvn::be::gpu

namespace nxrt::nvn::be::gpu {

bool spirv_dual_source_output(const std::vector<uint32_t>& in, std::vector<uint32_t>& out) {
    if (in.size() < 5) return false;
    std::vector<Inst> insts;
    for (size_t i = 5; i < in.size();) {
        const uint32_t len = in[i] >> 16;
        if (!len || i + len > in.size()) return false;
        insts.push_back({i, in[i] & 0xffff, len});
        i += len;
    }
    std::map<uint32_t, bool> is_output;
    for (const Inst& n : insts)
        if (n.op == OpVariable && n.len >= 4 && in[n.at + 3] == StorageOutput) is_output[in[n.at + 2]] = true;
    std::vector<uint32_t> result;
    result.reserve(in.size() + 4);
    result.insert(result.end(), in.begin(), in.begin() + 5);
    bool changed = false;
    for (const Inst& n : insts) {
        const uint32_t* w = &in[n.at];
        if (n.op == OpDecorate && n.len == 4 && w[2] == DecorationLocation && w[3] == 1 && is_output.count(w[1])) {
            result.insert(result.end(), {(4u << 16) | OpDecorate, w[1], DecorationLocation, 0u});
            result.insert(result.end(), {(4u << 16) | OpDecorate, w[1], 32u /* Index */, 1u});
            changed = true;
            continue;
        }
        result.insert(result.end(), in.begin() + static_cast<std::ptrdiff_t>(n.at), in.begin() + static_cast<std::ptrdiff_t>(n.at + n.len));
    }
    if (!changed) return false;
    out = std::move(result);
    return true;
}

}  // namespace nxrt::nvn::be::gpu

namespace nxrt::nvn::be::gpu {
namespace {
enum : uint32_t { OpCapability = 17, OpExecutionMode = 16, OpConstant = 43, OpStore = 62, OpLoad = 61, OpCopyObject = 83, OpAccessChain = 65, OpInBoundsAccessChain = 66,
                  DecOffset = 35, DecXfbBuffer = 36, DecXfbStride = 37, CapTransformFeedback = 53, ExecXfb = 11 };
}  // namespace

bool spirv_add_xfb(const std::vector<uint32_t>& in, std::vector<uint32_t>& out, std::vector<SpvXfbOutput>* layout, uint32_t* stride) {
    if (in.size() < 5) return false;
    std::vector<Inst> insts;
    for (size_t i = 5; i < in.size();) {
        const uint32_t len = in[i] >> 16;
        if (!len || i + len > in.size()) return false;
        insts.push_back({i, in[i] & 0xffff, len});
        i += len;
    }
    uint32_t bound = in[3];
    uint32_t f32 = 0, entry_id = 0;
    size_t first_type = 0, after_entry = 0, first_cap_end = 0;
    std::map<uint32_t, uint32_t> location;                            // variable id -> Location
    std::map<uint32_t, uint32_t> output_ptr_type;                     // output variable id -> pointer type id
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> vecs, ptrs;     // (component, count) / (storage, pointee) -> id
    std::map<uint32_t, uint32_t> constants;                           // OpConstant id -> low word
    std::map<uint32_t, uint32_t> max_component;                       // variable -> highest constant component index stored (+1)
    std::map<uint32_t, bool> whole;                                   // variable used as a whole (store, load, copy)
    for (const Inst& n : insts) {
        const uint32_t* w = &in[n.at];
        if (!first_type && is_type_or_constant(n.op)) first_type = n.at;
        if (n.op == OpCapability) first_cap_end = n.at + n.len;
        if (n.op == OpEntryPoint && !entry_id) entry_id = w[2];
        if (n.op == OpEntryPoint || n.op == OpExecutionMode) after_entry = n.at + n.len;
        if (n.op == OpTypeFloat && n.len >= 3 && w[2] == 32) f32 = w[1];
        else if (n.op == OpTypeVector && n.len == 4) vecs[{w[2], w[3]}] = w[1];
        else if (n.op == OpTypePointer && n.len == 4) ptrs[{w[2], w[3]}] = w[1];
        else if (n.op == OpConstant && n.len >= 4) constants[w[2]] = w[3];
        else if (n.op == OpDecorate && n.len >= 4 && w[2] == DecorationLocation) location[w[1]] = w[3];
        else if (n.op == OpVariable && n.len >= 4 && w[3] == StorageOutput) output_ptr_type[w[2]] = w[1];
    }
    if (!first_type || !first_cap_end || !after_entry || !f32 || !entry_id) return false;
    auto is_captured = [&](uint32_t v) { return location.count(v) && output_ptr_type.count(v); };
    for (const Inst& n : insts) {
        const uint32_t* w = &in[n.at];
        if ((n.op == OpAccessChain || n.op == OpInBoundsAccessChain) && n.len >= 5) {
            const uint32_t base = w[3];
            if (!is_captured(base)) continue;
            auto c = constants.find(w[4]);
            if (c == constants.end()) whole[base] = true;
            else max_component[base] = std::max(max_component[base], c->second + 1);
        } else if (n.op == OpStore && n.len >= 3) {
            if (is_captured(w[1])) whole[w[1]] = true;
        } else if ((n.op == OpLoad || n.op == OpCopyObject) && n.len >= 4) {
            if (is_captured(w[3])) whole[w[3]] = true;
        }
    }
    // the captured variables in Location order, each narrowed to the components that are stored
    std::map<uint32_t, uint32_t> by_location;   // Location -> variable id
    for (const auto& kv : output_ptr_type)
        if (location.count(kv.first)) by_location[location[kv.first]] = kv.first;
    if (by_location.empty()) return false;
    std::vector<uint32_t> decos, decls;
    std::map<uint32_t, uint32_t> new_ptr;       // variable id -> replacement pointer type
    uint32_t offset = 0;
    std::vector<SpvXfbOutput> lay;
    std::vector<std::pair<uint32_t, uint32_t>> decorate;   // variable, byte offset
    for (const auto& kv : by_location) {
        const uint32_t var = kv.second;
        uint32_t ncomp = (whole.count(var) || !max_component.count(var)) ? 4u : std::min(4u, max_component[var]);
        if (ncomp < 2) ncomp = 4;                // a lone scalar component keeps its vec4 (a scalar cannot be indexed)
        if (ncomp != 4) {
            const std::pair<uint32_t, uint32_t> vk = {f32, ncomp};
            uint32_t vid;
            auto vit = vecs.find(vk);
            if (vit != vecs.end()) {
                vid = vit->second;
            } else {
                vid = bound++;
                decls.insert(decls.end(), {(4u << 16) | OpTypeVector, vid, f32, ncomp});
                vecs[vk] = vid;
            }
            const std::pair<uint32_t, uint32_t> pk = {StorageOutput, vid};
            uint32_t pid;
            auto pit = ptrs.find(pk);
            if (pit != ptrs.end()) {
                pid = pit->second;
            } else {
                pid = bound++;
                decls.insert(decls.end(), {(4u << 16) | OpTypePointer, pid, StorageOutput, vid});
                ptrs[pk] = pid;
            }
            new_ptr[var] = pid;
        }
        decorate.push_back({var, offset});
        lay.push_back({kv.first, ncomp, offset});
        offset += 4 * ncomp;
    }
    for (const auto& d : decorate) {
        decos.insert(decos.end(), {(4u << 16) | OpDecorate, d.first, DecXfbBuffer, 0u});
        decos.insert(decos.end(), {(4u << 16) | OpDecorate, d.first, DecXfbStride, offset});
        decos.insert(decos.end(), {(4u << 16) | OpDecorate, d.first, DecOffset, d.second});
    }
    std::vector<uint32_t> res;
    res.reserve(in.size() + decos.size() + decls.size() + 8);
    res.insert(res.end(), in.begin(), in.begin() + 5);
    res[3] = bound;
    bool decls_done = false;
    for (const Inst& n : insts) {
        if (n.at == first_type) res.insert(res.end(), decos.begin(), decos.end());
        // the new vector and pointer types go in front of the first variable that uses one (every type precedes its users)
        if (!decls_done && ((n.op == OpVariable && n.len >= 4 && in[n.at + 3] == StorageOutput && new_ptr.count(in[n.at + 2])) || n.op == OpFunction)) {
            res.insert(res.end(), decls.begin(), decls.end());
            decls_done = true;
        }
        std::vector<uint32_t> copy(in.begin() + static_cast<std::ptrdiff_t>(n.at), in.begin() + static_cast<std::ptrdiff_t>(n.at + n.len));
        if (n.op == OpVariable && copy.size() >= 4 && copy[3] == StorageOutput) {
            auto it = new_ptr.find(copy[2]);
            if (it != new_ptr.end()) copy[1] = it->second;
        }
        res.insert(res.end(), copy.begin(), copy.end());
        if (n.at + n.len == first_cap_end) res.insert(res.end(), {(2u << 16) | OpCapability, CapTransformFeedback});
        if (n.at + n.len == after_entry) res.insert(res.end(), {(3u << 16) | OpExecutionMode, entry_id, ExecXfb});
    }
    if (!decls_done) return false;
    out = std::move(res);
    if (layout) *layout = lay;
    if (stride) *stride = offset;
    return true;
}

bool spirv_outputs_to_private(const std::vector<uint32_t>& in, std::vector<uint32_t>& out, uint32_t keep_mask) {
    if (in.size() < 5) return false;
    enum : uint32_t {
        OpVariableOp = 59, OpLoadOp = 61, OpStoreOp = 62, OpAccessChainOp = 65, OpInBoundsAccessChainOp = 66,
        OpDecorateOp = 71, OpEntryPointOp = 15, StoragePrivate = 6, DecIndex = 32, DecComponent = 31, DecBuiltIn = 11,
    };
    std::vector<Inst> insts;
    for (size_t i = 5; i < in.size();) {
        const uint32_t len = in[i] >> 16;
        if (!len || i + len > in.size()) return false;
        insts.push_back({i, in[i] & 0xffff, len});
        i += len;
    }
    uint32_t bound = in[3];
    size_t first_function = 0;
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> ptr_info;    // ptr_id -> (storage, pointee)
    std::map<uint32_t, size_t> ptr_at;
    std::map<uint32_t, uint32_t> priv_ptr;                          // pointee -> existing StoragePrivate ptr
    std::map<uint32_t, uint32_t> out_var, loc_of;
    std::set<uint32_t> builtin;
    for (const Inst& n : insts) {
        const uint32_t* w = &in[n.at];
        if (!first_function && n.op == OpFunction) first_function = n.at;
        else if (n.op == OpTypePointer && n.len == 4) {
            ptr_info[w[1]] = {w[2], w[3]};
            ptr_at[w[1]] = n.at;
            if (w[2] == StoragePrivate && !priv_ptr.count(w[3])) priv_ptr[w[3]] = w[1];
        } else if (n.op == OpVariableOp && n.len >= 4 && !first_function && w[3] == StorageOutput) out_var[w[2]] = w[1];
        else if (n.op == OpDecorateOp && n.len >= 4 && w[2] == DecorationLocation) loc_of[w[1]] = w[3];
        else if (n.op == OpDecorateOp && n.len >= 4 && w[2] == DecBuiltIn) builtin.insert(w[1]);
    }
    if (out_var.empty()) return false;
    std::set<uint32_t> drop_vars;
    for (const auto& kv : out_var) {
        if (builtin.count(kv.first) || !loc_of.count(kv.first)) continue;
        const uint32_t loc = loc_of[kv.first];
        if (loc < 32 && ((keep_mask >> loc) & 1u)) continue;
        drop_vars.insert(kv.first);
    }
    if (drop_vars.empty()) return false;

    std::map<uint32_t, uint32_t> drop_chains;                       // access chain result -> ptr_id
    std::set<uint32_t> needed_ptrs;
    for (uint32_t v : drop_vars) needed_ptrs.insert(out_var[v]);
    if (first_function) {
        for (const Inst& n : insts) {
            if (n.at < first_function) continue;
            const uint32_t* w = &in[n.at];
            if ((n.op == OpAccessChainOp || n.op == OpInBoundsAccessChainOp) && n.len >= 4 &&
                (drop_vars.count(w[3]) || drop_chains.count(w[3]))) {
                drop_chains[w[2]] = w[1];
                needed_ptrs.insert(w[1]);
            }
        }
        for (const Inst& n : insts) {
            if (n.at < first_function) continue;
            const uint32_t* w = &in[n.at];
            for (uint32_t k = 1; k < n.len; ++k) {
                const uint32_t id = w[k];
                if (!drop_vars.count(id) && !drop_chains.count(id)) continue;
                if (n.op == OpStoreOp && k == 1) continue;
                if (n.op == OpLoadOp && k == 3) continue;
                if ((n.op == OpAccessChainOp || n.op == OpInBoundsAccessChainOp) && (k == 2 || k == 3)) continue;
                return false;
            }
        }
    }

    std::map<uint32_t, uint32_t> remap_ptr;
    std::set<uint32_t> emit_after_ptr;
    for (uint32_t old_ptr : needed_ptrs) {
        auto pi = ptr_info.find(old_ptr);
        if (pi == ptr_info.end() || pi->second.first != StorageOutput) return false;
        const uint32_t pointee = pi->second.second;
        auto pp = priv_ptr.find(pointee);
        if (pp != priv_ptr.end() && ptr_at[pp->second] < ptr_at[old_ptr]) {
            remap_ptr[old_ptr] = pp->second;
        } else {
            remap_ptr[old_ptr] = bound++;
            emit_after_ptr.insert(old_ptr);
        }
    }

    const bool before_14 = in[1] < 0x10400;
    std::vector<uint32_t> res(in.begin(), in.begin() + 5);
    res.reserve(in.size() + emit_after_ptr.size() * 4);
    for (const Inst& n : insts) {
        std::vector<uint32_t> c(in.begin() + static_cast<std::ptrdiff_t>(n.at), in.begin() + static_cast<std::ptrdiff_t>(n.at + n.len));
        if (n.op == OpVariableOp && n.len >= 4 && drop_vars.count(c[2])) {
            c[1] = remap_ptr[c[1]];
            c[3] = StoragePrivate;
        } else if ((n.op == OpAccessChainOp || n.op == OpInBoundsAccessChainOp) && n.len >= 4 && drop_chains.count(c[2])) {
            c[1] = remap_ptr[c[1]];
        } else if (n.op == OpDecorateOp && n.len >= 4 && drop_vars.count(c[1]) &&
                   (c[2] == DecorationLocation || c[2] == DecIndex || c[2] == DecComponent)) {
            continue;
        } else if (n.op == OpEntryPointOp && before_14) {
            // OpEntryPoint model id name... interface ids: drop the demoted output variables from the interface
            size_t name_words = 0;
            const char* nm = reinterpret_cast<const char*>(&c[3]);
            const size_t max_chars = (c.size() - 3) * 4;
            size_t chars = 0;
            while (chars < max_chars && nm[chars]) ++chars;
            name_words = chars / 4 + 1;
            const size_t iface = 3 + name_words;
            std::vector<uint32_t> d(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(std::min(iface, c.size())));
            for (size_t i = iface; i < c.size(); ++i)
                if (!drop_vars.count(c[i])) d.push_back(c[i]);
            d[0] = (static_cast<uint32_t>(d.size()) << 16) | OpEntryPointOp;
            c = d;
        }
        res.insert(res.end(), c.begin(), c.end());
        if (n.op == OpTypePointer && n.len == 4 && emit_after_ptr.count(c[1])) {
            res.insert(res.end(), {(4u << 16) | OpTypePointer, remap_ptr[c[1]], StoragePrivate, c[3]});
        }
    }
    res[3] = bound;
    out = std::move(res);
    return true;
}

}  // namespace nxrt::nvn::be::gpu

namespace nxrt::nvn::be::gpu {

bool spirv_gs_passthrough(const std::vector<uint32_t>& in, const std::vector<SpvOutputSlot>& slots, std::vector<uint32_t>& out) {
    enum : uint32_t { OpExecutionModeOp = 16, OpConstantOp = 43, OpLoadOp = 61, OpStoreOp = 62, OpAccessChainOp = 65, OpEmitVertexOp = 218,
                      OpTypeArrayOp = 28, StorageInput = 1, ExecInputTriangles = 22 };
    for (const SpvOutputSlot& s : slots)
        if (s.type != SpvScalar::Float) return false;
    std::vector<uint32_t> tmp;
    if (!spirv_add_outputs(in, slots, tmp)) return false;       // the outputs (never written yet) and the interface entries

    std::vector<Inst> insts;
    for (size_t i = 5; i < tmp.size();) {
        const uint32_t len = tmp[i] >> 16;
        if (!len || i + len > tmp.size()) return false;
        insts.push_back({i, tmp[i] & 0xffff, len});
        i += len;
    }
    uint32_t bound = tmp[3];
    uint32_t f32 = 0, int32 = 0, v4 = 0, ptr_in_v4 = 0, first_function = 0, input_triangles = 0;
    std::map<uint32_t, uint32_t> location, var_type, array_elem, array_len_id, int_const;   // var -> Location / pointer type; array type -> element / length id; value -> id
    std::map<uint32_t, uint32_t> ptr_pointee, const_value;
    std::map<uint32_t, uint32_t> input_var, output_var;                                     // Location -> variable id
    for (const Inst& n : insts) {
        const uint32_t* w = &tmp[n.at];
        if (!first_function && n.op == OpFunction) first_function = static_cast<uint32_t>(n.at);
        if (n.op == OpExecutionModeOp && n.len >= 3 && w[2] == ExecInputTriangles) input_triangles = 1;
        if (n.op == OpTypeFloat && n.len >= 3 && w[2] == 32) f32 = w[1];
        else if (n.op == OpTypeInt && n.len == 4 && w[2] == 32 && w[3] == 1) int32 = w[1];
        else if (n.op == OpTypeVector && n.len == 4 && w[3] == 4 && f32 && w[2] == f32) v4 = w[1];
        else if (n.op == OpTypePointer && n.len == 4) { ptr_pointee[w[1]] = w[3]; if (w[2] == StorageInput && w[3] == v4) ptr_in_v4 = w[1]; }
        else if (n.op == OpTypeArrayOp && n.len == 4) { array_elem[w[1]] = w[2]; array_len_id[w[1]] = w[3]; }
        else if (n.op == OpConstantOp && n.len == 4) { const_value[w[2]] = w[3]; if (w[1] == int32) int_const[w[3]] = w[2]; }
        else if (n.op == OpDecorate && n.len >= 4 && w[2] == DecorationLocation) location[w[1]] = w[3];
        else if (n.op == OpVariable && n.len >= 4) var_type[w[2]] = w[1];
    }
    if (!input_triangles || !f32 || !v4 || !int32 || !first_function) return false;
    for (const Inst& n : insts) {
        const uint32_t* w = &tmp[n.at];
        if (n.op != OpVariable || n.len < 4 || !location.count(w[2])) continue;
        (w[3] == StorageInput ? input_var : output_var)[location[w[2]]] = w[2];
    }
    // every slot: an Input array [3] of vec4 and the Output variable spirv_add_outputs made
    std::vector<std::pair<uint32_t, uint32_t>> copies;   // (input var, output var)
    for (const SpvOutputSlot& s : slots) {
        if (!input_var.count(s.location) || !output_var.count(s.location)) return false;
        const uint32_t ptr = var_type[input_var[s.location]];
        const uint32_t arr = ptr_pointee.count(ptr) ? ptr_pointee[ptr] : 0;
        if (!arr || !array_elem.count(arr) || array_elem[arr] != v4 || !const_value.count(array_len_id[arr]) || const_value[array_len_id[arr]] != 3) return false;
        copies.emplace_back(input_var[s.location], output_var[s.location]);
    }
    uint32_t emits = 0;
    for (const Inst& n : insts) emits += n.op == OpEmitVertexOp;
    if (emits != 3) return false;

    std::vector<uint32_t> decls;
    if (!ptr_in_v4) { ptr_in_v4 = bound++; decls.insert(decls.end(), {(4u << 16) | OpTypePointer, ptr_in_v4, StorageInput, v4}); }
    for (uint32_t k = 0; k < 3; ++k)
        if (!int_const.count(k)) { const uint32_t id = bound++; int_const[k] = id; decls.insert(decls.end(), {(4u << 16) | OpConstantOp, int32, id, k}); }

    out.clear();
    out.reserve(tmp.size() + decls.size() + 12 * copies.size() + 8);
    out.insert(out.end(), tmp.begin(), tmp.begin() + 5);
    uint32_t emit_index = 0;
    for (const Inst& n : insts) {
        if (n.at == first_function) out.insert(out.end(), decls.begin(), decls.end());
        if (n.op == OpEmitVertexOp) {
            for (const auto& c : copies) {
                const uint32_t chain = bound++, val = bound++;
                out.insert(out.end(), {(5u << 16) | OpAccessChainOp, ptr_in_v4, chain, c.first, int_const[emit_index]});
                out.insert(out.end(), {(4u << 16) | OpLoadOp, v4, val, chain});
                out.insert(out.end(), {(3u << 16) | OpStoreOp, c.second, val});
            }
            ++emit_index;
        }
        out.insert(out.end(), tmp.begin() + static_cast<std::ptrdiff_t>(n.at), tmp.begin() + static_cast<std::ptrdiff_t>(n.at + n.len));
    }
    out[3] = bound;
    return true;
}

bool spirv_gs_input_topology(const std::vector<uint32_t>& in, uint32_t vk_topology, std::vector<uint32_t>& out) {
    enum : uint32_t {
        OpEntryPointOp = 15, OpExecutionModeOp = 16, OpTypeIntOp = 21, OpTypeArrayOp = 28, OpTypePointerOp = 32,
        OpConstantOp = 43, OpVariableOp = 59, ModelGeometry = 3, StorageInput = 1,
        ExecInputPoints = 19, ExecInputLines = 20, ExecInputLinesAdj = 21, ExecInputTriangles = 22, ExecInputTrianglesAdj = 23,
    };
    uint32_t want_mode = 0, want_len = 0;
    switch (vk_topology) {
        case 0: want_mode = ExecInputPoints; want_len = 1; break;           // VK_PRIMITIVE_TOPOLOGY_POINT_LIST
        case 1: case 2: want_mode = ExecInputLines; want_len = 2; break;    // LINE_LIST / LINE_STRIP
        case 6: case 7: want_mode = ExecInputLinesAdj; want_len = 4; break; // LINE_*_WITH_ADJACENCY
        case 3: case 4: case 5: want_mode = ExecInputTriangles; want_len = 3; break; // TRIANGLE_*
        case 8: case 9: want_mode = ExecInputTrianglesAdj; want_len = 6; break;      // TRIANGLE_*_WITH_ADJACENCY
        default: return false;
    }
    if (in.size() < 5) return false;

    uint32_t entry_id = 0, cur_mode = 0, cur_len = 0;
    size_t exec_at = 0, want_const_at = 0;
    uint32_t u32_id = 0, s32_id = 0, want_const_id = 0;
    std::map<uint32_t, uint32_t> const_val;
    std::map<uint32_t, uint32_t> arr_len;
    std::map<uint32_t, size_t> arr_at;
    std::map<uint32_t, uint32_t> ptr_sc, ptr_pointee;
    std::vector<uint32_t> input_var_ptrs;

    for (size_t i = 5; i < in.size();) {
        const uint32_t len = in[i] >> 16, op = in[i] & 0xffff;
        if (!len || i + len > in.size()) return false;
        const uint32_t* w = &in[i];
        if (op == OpEntryPointOp && len >= 3 && w[1] == ModelGeometry) {
            entry_id = w[2];
        } else if (op == OpExecutionModeOp && len >= 3 && entry_id && w[1] == entry_id &&
                   w[2] >= ExecInputPoints && w[2] <= ExecInputTrianglesAdj) {
            cur_mode = w[2];
            exec_at = i;
        } else if (op == OpTypeIntOp && len == 4 && w[2] == 32) {
            if (w[3] == 0) u32_id = w[1];
            else if (w[3] == 1) s32_id = w[1];
        } else if (op == OpConstantOp && len == 4 && ((u32_id && w[1] == u32_id) || (s32_id && w[1] == s32_id))) {
            const_val[w[2]] = w[3];
            if (w[3] == want_len && !want_const_id) { want_const_id = w[2]; want_const_at = i; }
        } else if (op == OpTypeArrayOp && len == 4) {
            arr_len[w[1]] = w[3];
            arr_at[w[1]] = i;
        } else if (op == OpTypePointerOp && len == 4) {
            ptr_sc[w[1]] = w[2];
            ptr_pointee[w[1]] = w[3];
        } else if (op == OpVariableOp && len >= 4 && w[3] == StorageInput) {
            input_var_ptrs.push_back(w[1]);
        }
        i += len;
    }

    if (!entry_id || !exec_at || cur_mode == want_mode) return false;
    switch (cur_mode) {
        case ExecInputPoints: cur_len = 1; break;
        case ExecInputLines: cur_len = 2; break;
        case ExecInputLinesAdj: cur_len = 4; break;
        case ExecInputTriangles: cur_len = 3; break;
        case ExecInputTrianglesAdj: cur_len = 6; break;
        default: return false;
    }

    uint32_t bound = in[3];
    std::map<uint32_t, uint32_t> ptr_to_old_arr;
    std::map<uint32_t, uint32_t> old_to_new_arr;
    size_t min_arr_at = in.size();
    for (uint32_t ptr_ty : input_var_ptrs) {
        auto sc_it = ptr_sc.find(ptr_ty);
        auto pt_it = ptr_pointee.find(ptr_ty);
        if (sc_it == ptr_sc.end() || sc_it->second != StorageInput || pt_it == ptr_pointee.end()) continue;
        const uint32_t arr_ty = pt_it->second;
        auto al_it = arr_len.find(arr_ty);
        if (al_it == arr_len.end()) continue;
        auto cv_it = const_val.find(al_it->second);
        if (cv_it == const_val.end() || cv_it->second != cur_len) continue;
        ptr_to_old_arr[ptr_ty] = arr_ty;
        if (!old_to_new_arr.count(arr_ty)) old_to_new_arr[arr_ty] = bound++;
        if (arr_at[arr_ty] < min_arr_at) min_arr_at = arr_at[arr_ty];
    }

    const uint32_t const_ty = u32_id ? u32_id : s32_id;
    const bool need_const = !old_to_new_arr.empty() && (!want_const_id || want_const_at > min_arr_at);
    if (need_const) {
        if (!const_ty) return false;
        want_const_id = bound++;
    }

    out.clear();
    out.reserve(in.size() + old_to_new_arr.size() * 4 + (need_const ? 4 : 0));
    out.insert(out.end(), in.begin(), in.begin() + 5);
    for (size_t i = 5; i < in.size();) {
        const uint32_t len = in[i] >> 16, op = in[i] & 0xffff;
        if (i == exec_at) {
            out.insert(out.end(), {(3u << 16) | OpExecutionModeOp, entry_id, want_mode});
        } else if (op == OpTypePointerOp && len == 4 && ptr_to_old_arr.count(in[i + 1])) {
            out.insert(out.end(), {in[i], in[i + 1], in[i + 2], old_to_new_arr[ptr_to_old_arr[in[i + 1]]]});
        } else {
            out.insert(out.end(), in.begin() + static_cast<std::ptrdiff_t>(i), in.begin() + static_cast<std::ptrdiff_t>(i + len));
        }
        if (need_const && op == OpTypeIntOp && len == 4 && in[i + 1] == const_ty) {
            out.insert(out.end(), {(4u << 16) | OpConstantOp, const_ty, want_const_id, want_len});
        }
        if (op == OpTypeArrayOp && len == 4) {
            auto it = old_to_new_arr.find(in[i + 1]);
            if (it != old_to_new_arr.end()) out.insert(out.end(), {(4u << 16) | OpTypeArrayOp, it->second, in[i + 2], want_const_id});
        }
        i += len;
    }
    out[3] = bound;
    return true;
}

}  // namespace nxrt::nvn::be::gpu

namespace nxrt::nvn::be::gpu {

bool spirv_int_attributes(const std::vector<uint32_t>& in, const std::vector<SpvIntSlot>& slots, std::vector<uint32_t>& out) {
    enum : uint32_t { OpTypeIntOp = 21, OpTypeFloatOp = 22, OpTypeVectorOp = 23, OpTypeArrayOp = 28, OpTypePointerOp = 32, OpConstantOp = 43, OpVariableOp = 59,
                      OpLoadOp = 61, OpStoreOp = 62, OpAccessChainOp = 65, OpInBoundsAccessChainOp = 66, OpDecorateOp = 71, OpCompositeConstructOp = 80,
                      OpBitcastOp = 124, OpEntryPointOp = 15, OpFunctionOp = 54, DecComponent = 31, StorageInputC = 1, StoragePrivateC = 6 };
    if (slots.empty() || in.size() < 5) return false;
    std::vector<Inst> insts;
    for (size_t i = 5; i < in.size();) {
        const uint32_t len = in[i] >> 16;
        if (!len || i + len > in.size()) return false;
        insts.push_back({i, in[i] & 0xffff, len});
        i += len;
    }
    uint32_t bound = in[3];
    uint32_t f32 = 0, v4 = 0, u32 = 0, s32 = 0, entry_id = 0;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> ptrs;      // (storage, pointee) -> pointer type id
    std::map<uint32_t, uint32_t> ptr_pointee, location, var_type;
    std::map<uint32_t, size_t> type_at;
    std::map<uint32_t, uint32_t> vec4_of;                         // scalar type id -> its vec4 id
    std::map<uint32_t, uint32_t> array_elem, array_len_id, const_value;   // array type -> element type / length constant; constant -> value
    for (const Inst& n : insts) {
        const uint32_t* w = &in[n.at];
        if (n.op == OpTypeFloatOp && n.len >= 3 && w[2] == 32) { f32 = w[1]; type_at[w[1]] = n.at; }
        else if (n.op == OpTypeIntOp && n.len == 4 && w[2] == 32) { (w[3] ? s32 : u32) = w[1]; type_at[w[1]] = n.at; }
        else if (n.op == OpTypeVectorOp && n.len == 4 && w[3] == 4) { vec4_of[w[2]] = w[1]; type_at[w[1]] = n.at; }
        else if (n.op == OpTypeArrayOp && n.len == 4) { array_elem[w[1]] = w[2]; array_len_id[w[1]] = w[3]; type_at[w[1]] = n.at; }
        else if (n.op == OpConstantOp && n.len == 4) const_value[w[2]] = w[3];
        else if (n.op == OpTypePointerOp && n.len == 4) { ptrs[{w[2], w[3]}] = w[1]; ptr_pointee[w[1]] = w[3]; type_at[w[1]] = n.at; }
        else if (n.op == OpDecorateOp && n.len >= 4 && w[2] == DecorationLocation) location[w[1]] = w[3];
        else if (n.op == OpVariableOp && n.len >= 4) var_type[w[2]] = w[1];
        else if (n.op == OpEntryPointOp && !entry_id) entry_id = w[2];
    }
    if (!f32 || !vec4_of.count(f32) || !entry_id) return false;
    v4 = vec4_of[f32];
    // One piece of work per input variable that changes. A plain variable is one vec4 at its Location; the translator also declares an Input
    // array of vec4 (a shader that indexes its attributes with a computed index), which covers Locations base .. base + length - 1.
    struct Elem { uint32_t location; int kind; uint32_t new_var; };   // kind 0 float, 1 uint, 2 sint
    struct Work { uint32_t old_var, old_ptr, array_type; size_t var_at; std::vector<Elem> elems; };
    std::vector<Work> work;
    size_t first_var_at = SIZE_MAX;
    std::map<uint32_t, int> slot_kind;                            // Location -> 1 uint / 2 sint
    for (const SpvIntSlot& sl : slots) slot_kind[sl.location] = sl.is_signed ? 2 : 1;
    std::map<uint32_t, size_t> work_of_var;
    for (const SpvIntSlot& sl : slots) {
        uint32_t var = 0;
        size_t at = 0;
        uint32_t array_type = 0, base = 0, len = 0;
        for (const Inst& n : insts) {
            const uint32_t* w = &in[n.at];
            if (n.op != OpVariableOp || n.len < 4 || w[3] != StorageInputC || !location.count(w[2])) continue;
            const uint32_t loc = location[w[2]], ptr = w[1];
            if (loc == sl.location && ptr_pointee.count(ptr) && ptr_pointee[ptr] == v4) { var = w[2]; at = n.at; array_type = 0; }
            else if (ptr_pointee.count(ptr) && array_elem.count(ptr_pointee[ptr]) && array_elem[ptr_pointee[ptr]] == v4 &&
                     const_value.count(array_len_id[ptr_pointee[ptr]])) {
                const uint32_t n_elems = const_value[array_len_id[ptr_pointee[ptr]]];
                if (sl.location >= loc && sl.location < loc + n_elems) { var = w[2]; at = n.at; array_type = ptr_pointee[ptr]; base = loc; len = n_elems; }
            }
        }
        if (!var) return false;
        if (work_of_var.count(var)) continue;                      // the array of an earlier slot
        Work wk{var, var_type[var], array_type, at, {}};
        if (array_type) {
            for (uint32_t e = 0; e < len; ++e) wk.elems.push_back({base + e, slot_kind.count(base + e) ? slot_kind[base + e] : 0, 0});
        } else {
            wk.elems.push_back({sl.location, sl.is_signed ? 2 : 1, 0});
        }
        work_of_var[var] = work.size();
        work.push_back(std::move(wk));
        first_var_at = std::min(first_var_at, at);
    }
    // the types the new variables need have to be declared before the first variable that changes
    std::vector<uint32_t> types;
    auto need_scalar = [&](bool is_signed) -> uint32_t {
        const uint32_t t = is_signed ? s32 : u32;
        return t && type_at.count(t) && type_at[t] < first_var_at ? t : 0;
    };
    auto vec4_for = [&](uint32_t scalar) -> uint32_t {
        auto it = vec4_of.find(scalar);
        if (it != vec4_of.end() && type_at[it->second] < first_var_at) return it->second;
        const uint32_t id = bound++;
        types.insert(types.end(), {(4u << 16) | OpTypeVectorOp, id, scalar, 4u});
        vec4_of[scalar] = id;
        type_at[id] = 0;
        return id;
    };
    auto ptr_for = [&](uint32_t storage, uint32_t pointee) -> uint32_t {
        auto it = ptrs.find({storage, pointee});
        if (it != ptrs.end() && type_at[it->second] < first_var_at) return it->second;
        const uint32_t id = bound++;
        types.insert(types.end(), {(4u << 16) | OpTypePointerOp, id, storage, pointee});
        ptrs[{storage, pointee}] = id;
        ptr_pointee[id] = pointee;
        type_at[id] = 0;
        return id;
    };
    std::map<uint32_t, uint32_t> priv_ptr_of;    // old Input pointer type -> the Private pointer type to the same pointee
    auto private_of = [&](uint32_t old_ptr) -> uint32_t {
        auto it = priv_ptr_of.find(old_ptr);
        if (it != priv_ptr_of.end()) return it->second;
        return priv_ptr_of[old_ptr] = ptr_for(StoragePrivateC, ptr_pointee[old_ptr]);
    };
    std::map<uint32_t, uint32_t> old_to_work;    // old var -> index in work
    std::vector<uint32_t> decls_after_types, decos;
    uint32_t v4u = 0, v4i = 0;
    for (size_t i = 0; i < work.size(); ++i) {
        Work& wk = work[i];
        for (Elem& el : wk.elems) {
            uint32_t vec = v4;
            if (el.kind) {
                const uint32_t sc = need_scalar(el.kind == 2);
                if (!sc) return false;
                uint32_t& vv = el.kind == 2 ? v4i : v4u;
                if (!vv) vv = vec4_for(sc);
                vec = vv;
            }
            const uint32_t in_ptr = ptr_for(StorageInputC, vec);
            el.new_var = bound++;
            decls_after_types.insert(decls_after_types.end(), {(4u << 16) | OpVariableOp, in_ptr, el.new_var, StorageInputC});
            decos.insert(decos.end(), {(4u << 16) | OpDecorateOp, el.new_var, DecorationLocation, el.location});
        }
        private_of(wk.old_ptr);
        old_to_work[wk.old_var] = static_cast<uint32_t>(i);
    }

    // the pointer types the access chains into the old inputs get (declared before the first changed variable)
    for (const Inst& n : insts) {
        const uint32_t* w = &in[n.at];
        if ((n.op == OpAccessChainOp || n.op == OpInBoundsAccessChainOp) && n.len >= 5 && old_to_work.count(w[3])) {
            if (!ptr_pointee.count(w[1])) return false;
            private_of(w[1]);
        }
    }

    // the entry function: the new instructions go behind its leading OpVariable run
    uint32_t entry_func = 0;
    for (const Inst& n : insts)
        if (n.op == OpEntryPointOp) { entry_func = in[n.at + 2]; break; }
    size_t insert_at = 0;
    for (size_t k = 0; k < insts.size() && !insert_at; ++k) {
        if (insts[k].op == OpFunctionOp && in[insts[k].at + 2] == entry_func) {
            size_t j = k + 1;
            while (j < insts.size() && insts[j].op != 248 /* OpLabel */) ++j;
            ++j;
            while (j < insts.size() && insts[j].op == OpVariableOp) ++j;
            if (j < insts.size()) insert_at = insts[j].at;
        }
    }
    if (!insert_at) return false;
    std::vector<uint32_t> init;
    for (const Work& wk : work) {
        std::vector<uint32_t> values;
        for (const Elem& el : wk.elems) {
            if (!el.kind) {                                        // a float element: the new input holds the same value
                const uint32_t loaded = bound++;
                init.insert(init.end(), {(4u << 16) | OpLoadOp, v4, loaded, el.new_var});
                values.push_back(loaded);
                continue;
            }
            const uint32_t vec = el.kind == 2 ? v4i : v4u;
            const uint32_t loaded = bound++, cast = bound++;
            init.insert(init.end(), {(4u << 16) | OpLoadOp, vec, loaded, el.new_var});
            init.insert(init.end(), {(4u << 16) | OpBitcastOp, v4, cast, loaded});
            values.push_back(cast);
        }
        if (!wk.array_type) {
            init.insert(init.end(), {(3u << 16) | OpStoreOp, wk.old_var, values[0]});
        } else {
            const uint32_t whole = bound++;
            init.push_back(((3u + static_cast<uint32_t>(values.size())) << 16) | OpCompositeConstructOp);
            init.push_back(wk.array_type);
            init.push_back(whole);
            init.insert(init.end(), values.begin(), values.end());
            init.insert(init.end(), {(3u << 16) | OpStoreOp, wk.old_var, whole});
        }
    }

    out.clear();
    out.reserve(in.size() + types.size() + decls_after_types.size() + decos.size() + init.size() + 16);
    out.insert(out.end(), in.begin(), in.begin() + 5);
    size_t first_function = 0;
    for (const Inst& n : insts)
        if (n.op == OpFunctionOp) { first_function = n.at; break; }
    bool deco_done = false;
    for (const Inst& n : insts) {
        const uint32_t* w = &in[n.at];
        if (!deco_done && is_type_or_constant(n.op)) { out.insert(out.end(), decos.begin(), decos.end()); deco_done = true; }
        if (n.at == first_var_at) out.insert(out.end(), types.begin(), types.end());
        if (n.at == first_function) out.insert(out.end(), decls_after_types.begin(), decls_after_types.end());
        if (n.at == insert_at) out.insert(out.end(), init.begin(), init.end());
        std::vector<uint32_t> c(in.begin() + static_cast<std::ptrdiff_t>(n.at), in.begin() + static_cast<std::ptrdiff_t>(n.at + n.len));
        if (n.op == OpVariableOp && n.len >= 4 && old_to_work.count(w[2])) {            // the old input becomes a private variable
            c[1] = private_of(work[old_to_work[w[2]]].old_ptr);
            c[3] = StoragePrivateC;
        } else if (n.op == OpDecorateOp && n.len >= 4 && old_to_work.count(w[1]) && (w[2] == DecorationLocation || w[2] == DecComponent)) {
            continue;
        } else if ((n.op == OpAccessChainOp || n.op == OpInBoundsAccessChainOp) && n.len >= 5 && old_to_work.count(w[3])) {
            c[1] = private_of(w[1]);                                                       // a pointer into the old input is a Private pointer now
        } else if (n.op == OpEntryPointOp) {
            // OpEntryPoint model id name... interface ids: the name is a literal string. The old input leaves the interface and its new
            // inputs join it.
            const char* nm = reinterpret_cast<const char*>(&c[3]);
            size_t chars = 0;
            while (chars < (c.size() - 3) * 4 && nm[chars]) ++chars;
            const size_t iface = 3 + chars / 4 + 1;
            std::vector<uint32_t> nc(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(iface));
            for (size_t k = iface; k < c.size(); ++k) {
                if (old_to_work.count(c[k])) {
                    for (const Elem& el : work[old_to_work[c[k]]].elems) nc.push_back(el.new_var);
                } else {
                    nc.push_back(c[k]);
                }
            }
            nc[0] = (static_cast<uint32_t>(nc.size()) << 16) | (nc[0] & 0xffff);
            c = std::move(nc);
        }
        out.insert(out.end(), c.begin(), c.end());
    }
    out[3] = bound;
    return true;
}


std::vector<SpvIntSlot> spirv_predict_int_inputs(const std::vector<uint32_t>& in) {
    enum : uint32_t { OpTypeIntOp = 21, OpTypeFloatOp = 22, OpTypeVectorOp = 23, OpTypeArrayOp = 28, OpTypePointerOp = 32, OpConstantOp = 43, OpVariableOp = 59,
                      OpLoadOp = 61, OpStoreOp = 62, OpAccessChainOp = 65, OpInBoundsAccessChainOp = 66, OpDecorateOp = 71, OpBitcastOp = 124, StorageInputC = 1 };
    std::vector<SpvIntSlot> result;
    if (in.size() < 5) return result;
    uint32_t f32 = 0, v4 = 0;
    std::map<uint32_t, bool> is_int32;                               // type id -> 32-bit integer
    std::map<uint32_t, uint32_t> ptr_pointee, array_elem, array_len, const_value, location, var_ptr;
    std::map<uint32_t, uint32_t> var_elems;                          // candidate Input variable -> number of vec4 elements (1 for a plain vec4)
    std::vector<Inst> insts;
    for (size_t i = 5; i < in.size();) {
        const uint32_t len = in[i] >> 16;
        if (!len || i + len > in.size()) return result;
        insts.push_back({i, in[i] & 0xffff, len});
        i += len;
    }
    for (const Inst& n : insts) {
        const uint32_t* w = &in[n.at];
        if (n.op == OpTypeFloatOp && n.len >= 3 && w[2] == 32) f32 = w[1];
        else if (n.op == OpTypeIntOp && n.len == 4 && w[2] == 32) is_int32[w[1]] = true;
        else if (n.op == OpTypeVectorOp && n.len == 4 && w[3] == 4 && w[2] == f32 && f32) v4 = w[1];
        else if (n.op == OpTypeArrayOp && n.len == 4) { array_elem[w[1]] = w[2]; array_len[w[1]] = w[3]; }
        else if (n.op == OpConstantOp && n.len == 4) const_value[w[2]] = w[3];
        else if (n.op == OpTypePointerOp && n.len == 4 && w[2] == StorageInputC) ptr_pointee[w[1]] = w[3];
        else if (n.op == OpDecorateOp && n.len >= 4 && w[2] == DecorationLocation) location[w[1]] = w[3];
        else if (n.op == OpVariableOp && n.len >= 4 && w[3] == StorageInputC) var_ptr[w[2]] = w[1];
    }
    if (!f32 || !v4) return result;
    for (const auto& [var, ptr] : var_ptr) {
        if (!location.count(var) || !ptr_pointee.count(ptr)) continue;
        const uint32_t t = ptr_pointee[ptr];
        if (t == v4) var_elems[var] = 1;
        else if (array_elem.count(t) && array_elem[t] == v4 && const_value.count(array_len[t]) && const_value[array_len[t]] <= 16) var_elems[var] = const_value[array_len[t]];
    }
    if (var_elems.empty()) return result;
    // use counts per location: [0] bitcast to a 32-bit integer, [1] anything else
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> uses;
    std::map<uint32_t, uint32_t> chain_loc;                          // access chain result -> location
    std::map<uint32_t, uint32_t> load_loc;                           // load result -> location
    std::map<uint32_t, uint32_t> pvar_loc;                           // local variable that holds a loaded attribute -> location
    auto spoil_var = [&](uint32_t var) { for (uint32_t e = 0; e < var_elems[var]; ++e) ++uses[location[var] + e].second; };
    for (const Inst& n : insts) {
        const uint32_t* w = &in[n.at];
        if ((n.op == OpAccessChainOp || n.op == OpInBoundsAccessChainOp) && n.len >= 5 && var_elems.count(w[3])) {
            const uint32_t var = w[3];
            const uint32_t indices = n.len - 4;
            const bool array = var_elems[var] > 1 || indices == 2;
            const uint32_t want = array ? 2 : 1;
            // a computed index reads whichever attribute the shader picks at run time: not counted either way
            if (indices != want || !const_value.count(w[4]) || (want == 2 && !const_value.count(w[5]))) continue;
            const uint32_t elem = array ? const_value[w[4]] : 0;
            if (elem >= var_elems[var]) continue;
            chain_loc[w[2]] = location[var] + elem;
        } else if (n.op == OpLoadOp && n.len >= 4) {
            if (chain_loc.count(w[3])) load_loc[w[2]] = chain_loc[w[3]];
            else if (pvar_loc.count(w[3])) load_loc[w[2]] = pvar_loc[w[3]];          // the value came back out of a local variable
            else if (var_elems.count(w[3])) spoil_var(w[3]);
        } else if (n.op == OpStoreOp && n.len >= 3 && load_loc.count(w[2]) && !chain_loc.count(w[1]) && !var_elems.count(w[1])) {
            auto it = pvar_loc.find(w[1]);                                            // stored into a local variable: later loads of it stand for the attribute
            if (it != pvar_loc.end() && it->second != load_loc[w[2]]) { ++uses[it->second].second; ++uses[load_loc[w[2]]].second; }
            else pvar_loc[w[1]] = load_loc[w[2]];
        }
    }
    for (const Inst& n : insts) {
        const uint32_t* w = &in[n.at];
        if (n.op == OpLoadOp || n.op == OpAccessChainOp || n.op == OpInBoundsAccessChainOp) continue;
        if (n.op == OpStoreOp && n.len >= 3 && pvar_loc.count(w[1])) continue;          // the store into a local variable was followed above
        for (uint32_t k = 1; k < n.len; ++k) {
            auto it = load_loc.find(w[k]);
            if (it == load_loc.end()) continue;
            if (n.op == OpBitcastOp && n.len == 4 && k == 3 && is_int32.count(w[1])) ++uses[it->second].first;
            else ++uses[it->second].second;
        }
    }
    for (const auto& [loc, u] : uses)
        if (u.first > 0 && u.second == 0) {
            SpvIntSlot sl;
            sl.location = loc;
            sl.is_signed = false;
            result.push_back(sl);
        }
    return result;
}

}  // namespace nxrt::nvn::be::gpu

namespace nxrt::nvn::be::gpu {

bool spirv_init_outputs(const std::vector<uint32_t>& in, std::vector<uint32_t>& out) {
    enum : uint32_t { OpLineOp = 8, OpTypeArrayOp = 28, OpConstantOp = 43, OpConstantCompositeOp = 44, OpStoreOp = 62, OpLabelOp = 248,
                      OpNoLineOp = 317, OpEmitVertexOp = 218, OpEmitStreamVertexOp = 220, DecBuiltIn = 11, DecComponent = 31,
                      ModelVertex = 0, ModelTessEval = 2, ModelGeometry = 3 };
    if (in.size() < 5) return false;
    std::vector<Inst> insts;
    for (size_t i = 5; i < in.size();) {
        const uint32_t len = in[i] >> 16;
        if (!len || i + len > in.size()) return false;
        insts.push_back({i, in[i] & 0xffff, len});
        i += len;
    }
    uint32_t bound = in[3];
    size_t entry_at = 0, first_function = 0;
    uint32_t entries = 0;
    struct Scalar { bool is_float; uint32_t width; };
    std::map<uint32_t, Scalar> scalars;                                   // OpTypeFloat / OpTypeInt id -> kind, width
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> vec_of;             // vector type -> (component type, count)
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> array_of;           // array type -> (element type, length constant id)
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> ptr_of;             // pointer type -> (storage class, pointee)
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> constants;          // (type, value) -> OpConstant id (one-word ones)
    std::map<uint32_t, uint32_t> const_value, location, component;
    std::map<uint32_t, bool> builtin;
    std::vector<std::pair<uint32_t, uint32_t>> output_vars;               // (variable id, pointer type) in declaration order
    for (const Inst& n : insts) {
        const uint32_t* w = &in[n.at];
        if (n.op == OpEntryPoint && n.len >= 4) { if (!entries++) entry_at = n.at; }
        else if (n.op == OpFunction) { if (!first_function) first_function = n.at; }
        else if (n.op == OpTypeFloat && n.len >= 3) scalars[w[1]] = {true, w[2]};
        else if (n.op == OpTypeInt && n.len == 4) scalars[w[1]] = {false, w[2]};
        else if (n.op == OpTypeVector && n.len == 4) vec_of[w[1]] = {w[2], w[3]};
        else if (n.op == OpTypeArrayOp && n.len == 4) array_of[w[1]] = {w[2], w[3]};
        else if (n.op == OpTypePointer && n.len == 4) ptr_of[w[1]] = {w[2], w[3]};
        else if (n.op == OpConstantOp && n.len == 4) { const_value[w[2]] = w[3]; constants.emplace(std::make_pair(w[1], w[3]), w[2]); }
        else if (n.op == OpDecorate && n.len >= 4 && w[2] == DecorationLocation) location[w[1]] = w[3];
        else if (n.op == OpDecorate && n.len >= 4 && w[2] == DecComponent) component[w[1]] = w[3];
        else if (n.op == OpDecorate && n.len >= 3 && w[2] == DecBuiltIn) builtin[w[1]] = true;
        else if (n.op == OpVariable && n.len == 4 && w[3] == StorageOutput && !first_function) output_vars.emplace_back(w[2], w[1]);   // no initializer
    }
    if (entries != 1 || !first_function) return false;
    const uint32_t* ep = &in[entry_at];
    const uint32_t ep_len = in[entry_at] >> 16, model = ep[1], entry_func = ep[2];
    if (model != ModelVertex && model != ModelTessEval && model != ModelGeometry) return false;
    // OpEntryPoint model id name... interface ids: the name is a literal string. Only a variable the entry point lists is stored to (before
    // SPIR-V 1.4 the list must hold every Output variable the entry point uses).
    std::map<uint32_t, bool> in_interface;
    {
        const char* nm = reinterpret_cast<const char*>(&ep[3]);
        size_t chars = 0;
        while (chars < (ep_len - 3) * 4 && nm[chars]) ++chars;
        for (size_t k = 3 + chars / 4 + 1; k < ep_len; ++k) in_interface[ep[k]] = true;
    }

    // The value each output starts with: component 3 is 1, the others 0 (1.0f for a float, 1 for an integer), counted from the variable's
    // Component decoration. An Output array of such vectors gets it in every element. New constants get ids from the bound.
    std::vector<uint32_t> decls;
    auto scalar_const = [&](uint32_t type, uint32_t value) -> uint32_t {
        auto it = constants.find({type, value});
        if (it != constants.end()) return it->second;
        const uint32_t id = bound++;
        decls.insert(decls.end(), {(4u << 16) | OpConstantOp, type, id, value});
        constants[{type, value}] = id;
        return id;
    };
    std::map<std::vector<uint32_t>, uint32_t> composites;                 // (type, constituents...) -> OpConstantComposite id
    auto composite_const = [&](uint32_t type, const std::vector<uint32_t>& parts) -> uint32_t {
        std::vector<uint32_t> key{type};
        key.insert(key.end(), parts.begin(), parts.end());
        auto it = composites.find(key);
        if (it != composites.end()) return it->second;
        const uint32_t id = bound++;
        decls.push_back(((3u + static_cast<uint32_t>(parts.size())) << 16) | OpConstantCompositeOp);
        decls.push_back(type);
        decls.push_back(id);
        decls.insert(decls.end(), parts.begin(), parts.end());
        composites[key] = id;
        return id;
    };
    // the start value of a scalar or vector of 32-bit float / int whose first component sits in slot `first`; 0 for any other type
    auto start_value = [&](uint32_t type, uint32_t first) -> uint32_t {
        uint32_t comp = type, count = 1;
        const bool vector = vec_of.count(type) != 0;
        if (vector) { comp = vec_of[type].first; count = vec_of[type].second; }
        auto sc = scalars.find(comp);
        if (sc == scalars.end() || sc->second.width != 32 || count < 1 || count > 4) return 0;
        const uint32_t one = sc->second.is_float ? 0x3f800000u : 1u;
        std::vector<uint32_t> parts;
        for (uint32_t c = 0; c < count; ++c) parts.push_back(scalar_const(comp, first + c == 3 ? one : 0u));
        return vector ? composite_const(type, parts) : parts[0];
    };
    std::vector<uint32_t> stores;
    for (const auto& [var, ptr] : output_vars) {
        if (!location.count(var) || builtin.count(var) || !in_interface.count(var)) continue;
        auto p = ptr_of.find(ptr);
        if (p == ptr_of.end() || p->second.first != StorageOutput) continue;
        const uint32_t type = p->second.second, first = component.count(var) ? component[var] : 0u;
        uint32_t value = 0;
        if (array_of.count(type)) {
            const auto [elem, len_id] = array_of[type];
            if (!const_value.count(len_id) || const_value[len_id] == 0 || const_value[len_id] > 64) continue;
            const uint32_t e = start_value(elem, first);
            if (!e) continue;
            value = composite_const(type, std::vector<uint32_t>(const_value[len_id], e));
        } else {
            value = start_value(type, first);
        }
        if (value) stores.insert(stores.end(), {(3u << 16) | OpStoreOp, var, value});
    }
    if (stores.empty()) return false;

    // the stores open the entry function, behind the leading OpVariable run of its first block
    size_t insert_at = 0;
    for (size_t k = 0; k < insts.size() && !insert_at; ++k) {
        if (insts[k].op != OpFunction || in[insts[k].at + 2] != entry_func) continue;
        size_t j = k + 1;
        while (j < insts.size() && insts[j].op != OpLabelOp && insts[j].op != OpFunction) ++j;
        if (j >= insts.size() || insts[j].op != OpLabelOp) return false;
        ++j;
        while (j < insts.size() && (insts[j].op == OpVariable || insts[j].op == OpLineOp || insts[j].op == OpNoLineOp)) ++j;
        if (j < insts.size()) insert_at = insts[j].at;
    }
    if (!insert_at) return false;

    out.clear();
    out.reserve(in.size() + decls.size() + stores.size() * 4 + 8);
    out.insert(out.end(), in.begin(), in.begin() + 5);
    for (const Inst& n : insts) {
        if (n.at == first_function) out.insert(out.end(), decls.begin(), decls.end());
        if (n.at == insert_at) out.insert(out.end(), stores.begin(), stores.end());
        out.insert(out.end(), in.begin() + static_cast<std::ptrdiff_t>(n.at), in.begin() + static_cast<std::ptrdiff_t>(n.at + n.len));
        // a geometry shader's outputs are undefined again after every emitted vertex (Vulkan). Maxwell keeps them instead, so a shader that
        // wrote an output once and emitted it twice would read (0,0,0,1) the second time; every GS this game ships rewrites each output
        // before each emit, or returns right after its only one. A shader that relies on it needs Private shadow copies flushed at each emit.
        if (model == ModelGeometry && (n.op == OpEmitVertexOp || n.op == OpEmitStreamVertexOp)) out.insert(out.end(), stores.begin(), stores.end());
    }
    out[3] = bound;
    return true;
}

// ---- integer outputs ----------------------------------------------------------------------------------------------------------
// A pass whose colour attachment is an integer format: Vulkan wants the fragment shader's output to be that integer type, while the
// console's ROP stores whatever bits the shader hands over (the game's shaders build them and write intBitsToFloat of them). Every
// Location output is redeclared as the unsigned integer vector of the same width and each stored value is bitcast, so the image gets
// exactly the bits the shader produced - which is what a logic op (OR, into the light-probe list) then combines.
static thread_local std::string g_out_rewrite_why;
const char* spirv_outputs_to_uint_reason() { return g_out_rewrite_why.c_str(); }

bool spirv_outputs_to_uint(const std::vector<uint32_t>& in, std::vector<uint32_t>& out) {
    // HANDOVER 90: a reason string, so a shader the rewrite refuses can be diagnosed without a debugger.
    g_out_rewrite_why.clear();
    auto fail = [](const char* w) { g_out_rewrite_why = w; return false; };
    enum : uint32_t {
        OpEntryPointOp = 15, OpTypeIntOp = 21, OpTypeFloatOp = 22, OpTypeVectorOp = 23, OpTypePointerOp = 32, OpFunctionOp = 54,
        OpVariableOp = 59, OpLoadOp = 61, OpStoreOp = 62, OpAccessChainOp = 65, OpInBoundsAccessChainOp = 66, OpDecorateOp = 71,
        OpConstantOp = 43, OpBitcastOp = 124, OpImageWriteOp = 99,
        StorageOutputOp = 3, DecorationLocationOp = 30, ModelFragment = 4,
    };
    if (in.size() < 5) return false;
    std::vector<Inst> insts;
    for (size_t i = 5; i < in.size();) {
        const uint32_t len = in[i] >> 16;
        if (!len || i + len > in.size()) return false;
        insts.push_back({i, in[i] & 0xffff, len});
        i += len;
    }
    uint32_t bound = in[3];
    size_t entry_at = 0, first_function = 0;
    uint32_t entries = 0;
    struct ScalarDecl { int kind = 0; uint32_t width = 0; };       // kind: 0 float, 1 uint, 2 sint
    std::map<uint32_t, ScalarDecl> scalars;                         // type -> kind, width
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> vecs;         // type -> (component type, count)
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> ptrs;         // type -> (storage class, pointee)
    std::map<uint32_t, size_t> type_at;                             // type -> instruction offset
    std::map<uint32_t, uint32_t> location;                          // variable -> Location
    std::map<uint32_t, uint32_t> var_ptr;                           // variable -> pointer type
    std::map<uint32_t, uint32_t> const_ids;                         // ids of OpConstant values (the component indices)
    std::vector<uint32_t> out_vars;
    for (const Inst& n : insts) {
        const uint32_t* w = &in[n.at];
        if (n.op == OpEntryPointOp) { ++entries; entry_at = n.at; }
        else if (n.op == OpFunctionOp) { if (!first_function) first_function = n.at; }
        else if (n.op == OpTypeIntOp && n.len == 4) { scalars[w[1]] = {w[3] ? 2 : 1, w[2]}; type_at[w[1]] = n.at; }
        else if (n.op == OpTypeFloatOp && n.len >= 3) { scalars[w[1]] = {0, w[2]}; type_at[w[1]] = n.at; }
        else if (n.op == OpTypeVectorOp && n.len == 4) { vecs[w[1]] = {w[2], w[3]}; type_at[w[1]] = n.at; }
        else if (n.op == OpTypePointerOp && n.len == 4) { ptrs[w[1]] = {w[2], w[3]}; type_at[w[1]] = n.at; }
        else if (n.op == OpDecorateOp && n.len >= 4 && w[2] == DecorationLocationOp) location[w[1]] = w[3];
        else if (n.op == OpConstantOp && n.len >= 4) const_ids[w[2]] = w[3];
        else if (n.op == OpVariableOp && n.len == 4 && w[3] == StorageOutputOp && !first_function) {
            out_vars.push_back(w[2]);
            var_ptr[w[2]] = w[1];
        }
    }
    if (entries != 1 || !first_function || out_vars.empty()) return false;
    if (in[entry_at + 1] != ModelFragment) return false;

    uint32_t uint_scalar = 0, float_scalar = 0;
    for (const auto& kv : scalars) {
        if (kv.second.width != 32) continue;
        if (kv.second.kind == 0) float_scalar = kv.first;
        else if (kv.second.kind == 1) uint_scalar = kv.first;
    }
    std::vector<uint32_t> decls;
    if (!uint_scalar) {
        uint_scalar = bound++;
        decls.insert(decls.end(), {(4u << 16) | OpTypeIntOp, uint_scalar, 32, 0});
        scalars[uint_scalar] = {1, 32};
        type_at[uint_scalar] = 0;
    }
    auto uint_vec = [&](uint32_t count) -> uint32_t {
        for (const auto& kv : vecs)
            if (kv.second.first == uint_scalar && kv.second.second == count) return kv.first;
        const uint32_t id = bound++;
        decls.insert(decls.end(), {(4u << 16) | OpTypeVectorOp, id, uint_scalar, count});
        vecs[id] = {uint_scalar, count};
        type_at[id] = 0;
        return id;
    };
    auto out_ptr = [&](uint32_t pointee) -> uint32_t {
        for (const auto& kv : ptrs)
            if (kv.second.first == StorageOutputOp && kv.second.second == pointee) return kv.first;
        const uint32_t id = bound++;
        decls.insert(decls.end(), {(4u << 16) | OpTypePointerOp, id, StorageOutputOp, pointee});
        ptrs[id] = {StorageOutputOp, pointee};
        type_at[id] = 0;
        return id;
    };

    std::map<uint32_t, uint32_t> new_var_type;                      // output variable -> new pointer type (vector of integers)
    std::map<uint32_t, uint32_t> new_value_type;                    // output variable -> new integer vector type
    for (uint32_t var : out_vars) {
        if (!location.count(var)) continue;
        auto p = ptrs.find(var_ptr[var]);
        if (p == ptrs.end() || p->second.first != StorageOutputOp) return fail("output variable without a pointer type");
        auto v = vecs.find(p->second.second);
        if (v == vecs.end()) return fail("scalar or array output");
        auto sc = scalars.find(v->second.first);
        if (sc == scalars.end() || sc->second.kind != 0 || sc->second.width != 32) return fail("output component is not f32");
        const uint32_t n = v->second.second;
        if (n < 1 || n > 4) return fail("output vector width");
        new_value_type[var] = uint_vec(n);
        new_var_type[var] = out_ptr(new_value_type[var]);
    }
    if (new_var_type.empty()) return fail("no output variable with a Location");

    // Component writes go through an access chain per component (spirv-cross prints them as out.x = ...). A chain this patch rewrites
    // must only ever be a store target: a load through it would hand integer bits to the shader's float math.
    std::map<uint32_t, uint32_t> chain_var;                         // chain result -> output variable
    std::map<size_t, uint32_t> chain_type;                          // chain instruction at -> its new pointer type
    for (const Inst& n : insts) {
        if (n.op != OpAccessChainOp && n.op != OpInBoundsAccessChainOp) continue;
        if (n.len != 5) continue;
        const uint32_t var = in[n.at + 3];
        if (!new_value_type.count(var)) continue;
        if (!const_ids.count(in[n.at + 4])) return fail("access chain index is not a constant");
        chain_var[in[n.at + 2]] = var;
        chain_type[n.at] = out_ptr(uint_scalar);
    }
    if (chain_var.empty() && new_var_type.empty()) return fail("no component chain");

    std::vector<std::pair<size_t, std::vector<uint32_t>>> inserts;  // words to put in front of the instruction at
    std::map<size_t, uint32_t> store_value;                         // store at -> the bitcast that replaces its object
    std::map<size_t, uint32_t> var_type_patch;                      // OpVariable at -> new pointer type
    // A shader that reads back what it wrote (read-modify-write, e.g. the light-probe list's writers) loads through the chain.
    // That is fine: the load then produces the target's integer type and gets a bitcast back to the shader's float type, so the
    // shader's own arithmetic sees the same bits. Anything else a chain is used for (a function argument, a copy) still refuses
    // the variant, because then the pointer would escape into code this patch cannot retype.
    std::map<uint32_t, uint32_t> load_bc;                           // load result id -> the bitcast that replaces it
    std::map<uint32_t, uint32_t> load_ty;                           // load result id -> its float type
    // Only instructions whose operands are all ids can carry the pointer, and only inside a function body: the type and
    // decoration sections hold literals that collide with chain ids (an OpMemberDecorate Offset of 192 read as id 192 is how
    // the first version of this scan refused 14 shaders that were fine).
    enum : uint32_t { OpFunctionCallOp = 57, OpCopyMemoryOp = 63, OpCopyMemorySizedOp = 64, OpSelectOp = 169, OpPhiOp = 245 };
    for (const Inst& n : insts) {
        if (n.at < first_function) continue;
        if (n.op != OpStoreOp && n.op != OpLoadOp && n.op != OpFunctionCallOp && n.op != OpCopyMemoryOp &&
            n.op != OpCopyMemorySizedOp && n.op != OpSelectOp && n.op != OpPhiOp) continue;
        const uint32_t* w = &in[n.at];
        for (uint32_t k = 1; k < n.len; ++k) {
            if (!chain_var.count(w[k])) continue;
            if (n.op == OpStoreOp && k == 1) continue;              // the store target
            if (n.op == OpLoadOp && k == 3 && n.len == 4) {         // a read through the chain
                load_bc[w[2]] = bound++;
                load_ty[w[2]] = w[1];
                continue;
            }
            return fail("an output access chain escapes into code this patch cannot retype");
        }
    }
    for (const Inst& n : insts) {
        const uint32_t* w = &in[n.at];
        if (n.op == OpVariableOp && n.len == 4 && new_var_type.count(w[2])) var_type_patch[n.at] = new_var_type[w[2]];
        if (n.op != OpStoreOp || n.len < 3) continue;
        uint32_t type = 0;
        auto c = chain_var.find(w[1]);
        if (c != chain_var.end()) type = uint_scalar;               // one component
        else if (new_value_type.count(w[1])) type = new_value_type[w[1]];
        else continue;
        const uint32_t tmp = bound++;
        inserts.emplace_back(n.at, std::vector<uint32_t>{(4u << 16) | OpBitcastOp, type, tmp, w[2]});
        store_value[n.at] = tmp;
    }
    if (store_value.empty()) return fail("no store to rewrite");
    std::sort(inserts.begin(), inserts.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<uint32_t> moved_vars;
    for (const Inst& n : insts) {
        auto vt = var_type_patch.find(n.at);
        if (vt == var_type_patch.end()) continue;
        moved_vars.push_back(in[n.at]);
        moved_vars.push_back(vt->second);
        moved_vars.push_back(in[n.at + 2]);
        moved_vars.push_back(in[n.at + 3]);
    }
    (void)type_at;
    out.clear();
    out.reserve(in.size() + decls.size() + inserts.size() * 4 + 8);
    out.insert(out.end(), in.begin(), in.begin() + 5);
    size_t next_insert = 0;
    for (const Inst& n : insts) {
        if (n.at == first_function) {
            out.insert(out.end(), decls.begin(), decls.end());
            out.insert(out.end(), moved_vars.begin(), moved_vars.end());
        }
        if (var_type_patch.count(n.at)) continue;
        while (next_insert < inserts.size() && inserts[next_insert].first == n.at) {
            out.insert(out.end(), inserts[next_insert].second.begin(), inserts[next_insert].second.end());
            ++next_insert;
        }
        const size_t at = out.size();
        out.insert(out.end(), in.begin() + static_cast<std::ptrdiff_t>(n.at), in.begin() + static_cast<std::ptrdiff_t>(n.at + n.len));
        auto ct = chain_type.find(n.at);
        if (ct != chain_type.end()) out[at + 1] = ct->second;
        auto sv = store_value.find(n.at);
        if (sv != store_value.end() && n.len > 2) out[at + 2] = sv->second;
        // a load through a rewritten chain yields the integer type; every user goes through the bitcast that follows it
        if (!load_bc.empty()) {
            const bool has_result = !(n.op == OpStoreOp || n.op == OpImageWriteOp || (n.op >= 246 && n.op <= 255));
            for (uint32_t k = 1; k < n.len; ++k) {
                if (has_result && k == 2) continue;                 // the instruction's own result id
                auto b = load_bc.find(out[at + k]);
                if (b != load_bc.end()) out[at + k] = b->second;
            }
            auto lb = load_ty.find(in[n.at + 2]);
            if (n.op == OpLoadOp && lb != load_ty.end()) {
                out.insert(out.end(), {(4u << 16) | OpBitcastOp, lb->second, load_bc[in[n.at + 2]], in[n.at + 2]});
                out[at + 1] = uint_scalar;                          // the load itself now yields uint
            }
        }
    }
    (void)float_scalar;
    out[3] = bound;
    return true;
}

}  // namespace nxrt::nvn::be::gpu

namespace nxrt::nvn::be::gpu {

// ---- early fragment tests -------------------------------------------------------------------------------------------------------
// A fragment shader that writes a storage image has a side effect, so a driver may run it for fragments that the depth or stencil test
// then rejects (late tests). The console does not: NVN's ROP tests first, so a rejected fragment leaves no trace. The light-probe list
// depends on that - its OR shaders register the tile in t51 and write the index in the same branch, and a registration that comes from a
// fragment the depth test rejects fills the per-tile slot with nothing (HANDOVER 42.21: the black rectangle). EarlyFragmentTests makes
// the tests run before the fragment shader, which is the console's order.
bool spirv_add_early_fragment_tests(const std::vector<uint32_t>& in, std::vector<uint32_t>& out, bool amd) {
    enum : uint32_t {
        OpExtensionOp = 10, OpEntryPointOp = 15, OpExecutionModeOp = 16, OpCapabilityOp = 17, OpDecorateOp = 71,
        OpImageTexelPointerOp = 60, OpImageWriteOp = 99,
        ModelFragment = 4, ExecutionModeEarlyFragmentTests = 9, ExecutionModeDepthReplacing = 10, DecorationBuiltIn = 11, BuiltInFragDepth = 22,
        ExecutionModeEarlyAndLateFragmentTestsAMD = 5017,
    };
    if (in.size() < 5) return false;
    bool fragment = false, image_write = false, writes_depth = false, already = false, has_amd_ext = false;
    uint32_t main_id = 0;
    size_t entry_at = 0, after_caps = 5;
    for (size_t i = 5; i < in.size();) {
        const uint32_t len = in[i] >> 16, op = in[i] & 0xffff;
        if (!len || i + len > in.size()) return false;
        if (op == OpCapabilityOp) after_caps = i + len;
        if (op == OpExtensionOp) {
            std::string s;
            for (uint32_t k = i + 1; k < i + len; ++k) {
                const char* c = reinterpret_cast<const char*>(&in[k]);
                for (int b = 0; b < 4; ++b) { if (c[b]) s.push_back(c[b]); else b = 4; }
            }
            if (s == "SPV_AMD_shader_early_and_late_fragment_tests") has_amd_ext = true;
        }
        if (op == OpEntryPointOp && in[i + 1] == ModelFragment) { fragment = true; main_id = in[i + 2]; entry_at = i; }
        if (op == OpImageWriteOp || op == OpImageTexelPointerOp) image_write = true;
        if (op == OpExecutionModeOp && main_id && in[i + 1] == main_id &&
            (in[i + 2] == ExecutionModeEarlyFragmentTests || in[i + 2] == ExecutionModeEarlyAndLateFragmentTestsAMD || in[i + 2] == ExecutionModeDepthReplacing))
            already = true;
        if (op == OpDecorateOp && len >= 4 && in[i + 2] == DecorationBuiltIn && in[i + 3] == BuiltInFragDepth) writes_depth = true;
        i += len;
    }
    if (!fragment || !image_write || writes_depth || already || !entry_at) return false;
    // AMD's mode also wants the extension declared: a null-terminated literal, padded to whole words, after the capabilities
    std::vector<uint32_t> ext;
    if (amd && !has_amd_ext) {
        static const char kNm[] = "SPV_AMD_shader_early_and_late_fragment_tests";
        const size_t bytes = sizeof kNm;                       // includes the terminator
        const size_t words = (bytes + 3) / 4;
        std::vector<char> lit(words * 4, 0);
        std::memcpy(lit.data(), kNm, sizeof kNm);
        ext.reserve(words + 1);
        ext.push_back(static_cast<uint32_t>(((words + 1) << 16) | OpExtensionOp));
        for (size_t k = 0; k < words; ++k) {
            uint32_t v;
            std::memcpy(&v, lit.data() + k * 4, 4);
            ext.push_back(v);
        }
    }
    const uint32_t mode = amd ? ExecutionModeEarlyAndLateFragmentTestsAMD : ExecutionModeEarlyFragmentTests;
    const size_t len = in[entry_at] >> 16;
    out.clear();
    out.reserve(in.size() + ext.size() + 3);
    if (!ext.empty()) {
        out.insert(out.end(), in.begin(), in.begin() + static_cast<std::ptrdiff_t>(after_caps));
        out.insert(out.end(), ext.begin(), ext.end());
        out.insert(out.end(), in.begin() + static_cast<std::ptrdiff_t>(after_caps), in.begin() + static_cast<std::ptrdiff_t>(entry_at + len));
    } else {
        out.insert(out.end(), in.begin(), in.begin() + static_cast<std::ptrdiff_t>(entry_at + len));
    }
    out.insert(out.end(), {static_cast<uint32_t>((3u << 16) | OpExecutionModeOp), main_id, mode});
    out.insert(out.end(), in.begin() + static_cast<std::ptrdiff_t>(entry_at + len), in.end());
    return true;
}

bool spirv_uint_samplers(const std::vector<uint32_t>& in, const std::vector<SpvSamplerSlot>& slots, std::vector<uint32_t>& out) {
    if (slots.empty() || in.size() < 5) return false;
    enum : uint32_t {
        OpTypeImageOp = 25, OpTypeSampledImageOp = 27, OpCopyObjectOp = 83, OpSampledImageOp = 86, OpImageOp = 100,
        OpLoadOp = 61, OpStoreOp = 62, OpCopyMemoryOp = 63, OpCopyMemorySizedOp = 64, OpImageWriteOp = 99,
        OpBitcastOp = 124, DecorationBindingOp = 33, DecorationDescriptorSetOp = 34, StorageUniformConstant = 0,
    };
    auto is_sample_op = [](uint32_t op) { return (op >= 87 && op <= 98) || (op >= 305 && op <= 314); };
    std::vector<Inst> insts;
    for (size_t i = 5; i < in.size();) {
        const uint32_t len = in[i] >> 16;
        if (!len || i + len > in.size()) return false;
        insts.push_back({i, in[i] & 0xffff, len});
        i += len;
    }

    uint32_t bound = in[3];
    size_t first_function = 0, first_img_at = 0, u32_at = 0;
    std::map<uint32_t, uint32_t> sets, bindings;
    std::map<uint32_t, size_t> img_at, simg_at, ptr_at, var_at;
    std::map<uint32_t, uint32_t> simg_to_img;
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> ptrs;
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> vecs;
    std::map<uint32_t, uint32_t> var_ptr;
    uint32_t f32 = 0, u32 = 0, v4u = 0;
    for (const Inst& n : insts) {
        const uint32_t* w = &in[n.at];
        if (!first_function && n.op == OpFunction) first_function = n.at;
        else if (n.op == OpDecorate && n.len >= 4) {
            if (w[2] == DecorationDescriptorSetOp) sets[w[1]] = w[3];
            else if (w[2] == DecorationBindingOp) bindings[w[1]] = w[3];
        } else if (n.op == OpTypeFloat && n.len >= 3 && w[2] == 32) f32 = w[1];
        else if (n.op == OpTypeInt && n.len == 4 && w[2] == 32 && w[3] == 0) {
            u32 = w[1];
            u32_at = n.at;
        } else if (n.op == OpTypeVector && n.len == 4) vecs[w[1]] = {w[2], w[3]};
        else if (n.op == OpTypeImageOp && n.len >= 9) {
            img_at[w[1]] = n.at;
            if (!first_img_at) first_img_at = n.at;
        } else if (n.op == OpTypeSampledImageOp && n.len == 3) {
            simg_to_img[w[1]] = w[2];
            simg_at[w[1]] = n.at;
        } else if (n.op == OpTypePointer && n.len == 4) {
            ptrs[w[1]] = {w[2], w[3]};
            ptr_at[w[1]] = n.at;
        } else if (n.op == OpVariable && n.len >= 4 && w[3] == StorageUniformConstant && !first_function) {
            var_ptr[w[2]] = w[1];
            var_at[w[2]] = n.at;
        }
    }
    if (!first_function || !first_img_at || !f32) return false;
    if (u32) {
        for (const auto& kv : vecs)
            if (kv.second.first == u32 && kv.second.second == 4) { v4u = kv.first; break; }
    }

    auto wanted_slot = [&](uint32_t vid) {
        auto bs = bindings.find(vid);
        if (bs == bindings.end()) return false;
        const uint32_t s = sets.count(vid) ? sets[vid] : 0;
        for (const SpvSamplerSlot& sl : slots)
            if (sl.set == s && sl.binding == bs->second) return true;
        return false;
    };

    std::map<size_t, std::vector<uint32_t>> after_type_decls;
    std::vector<uint32_t> pre_types;
    if (!u32) {
        u32 = bound++;
        pre_types.insert(pre_types.end(), {(4u << 16) | OpTypeInt, u32, 32u, 0u});
        u32_at = first_img_at;
    }
    if (!v4u) {
        v4u = bound++;
        if (u32_at > first_img_at) {
            auto& ins = after_type_decls[u32_at];
            ins.insert(ins.end(), {(4u << 16) | OpTypeVector, v4u, u32, 4u});
        } else {
            pre_types.insert(pre_types.end(), {(4u << 16) | OpTypeVector, v4u, u32, 4u});
        }
        vecs[v4u] = {u32, 4u};
    }

    std::map<std::vector<uint32_t>, std::pair<uint32_t, size_t>> created_images;
    std::map<uint32_t, std::pair<uint32_t, size_t>> created_simgs;
    std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, size_t>> created_ptrs;
    std::map<uint32_t, uint32_t> var_new_ptr, var_load_ty, var_img_ty;
    std::set<uint32_t> relocated_vars;

    for (const auto& kv : var_ptr) {
        const uint32_t vid = kv.first;
        if (!wanted_slot(vid)) continue;
        auto pit = ptrs.find(kv.second);
        if (pit == ptrs.end()) continue;
        const uint32_t storage = pit->second.first, pointee = pit->second.second;
        bool sampled = false;
        uint32_t old_img = 0;
        if (simg_to_img.count(pointee)) { sampled = true; old_img = simg_to_img[pointee]; }
        else if (img_at.count(pointee)) { old_img = pointee; }
        else continue;
        const size_t iat = img_at[old_img];
        const uint32_t ilen = in[iat] >> 16;
        if (in[iat + 2] != f32) continue;   // already integer or non-float

        const std::vector<uint32_t> tail(in.begin() + static_cast<std::ptrdiff_t>(iat + 3),
                                         in.begin() + static_cast<std::ptrdiff_t>(iat + ilen));
        uint32_t new_img = 0;
        size_t img_def_at = std::max(iat, u32_at);
        if (auto ci = created_images.find(tail); ci != created_images.end()) {
            new_img = ci->second.first;
            img_def_at = ci->second.second;
        } else {
            for (const auto& ex : img_at) {
                const uint32_t ex_len = in[ex.second] >> 16;
                if (in[ex.second + 2] == u32 && ex_len == ilen &&
                    std::equal(in.begin() + static_cast<std::ptrdiff_t>(ex.second + 3),
                               in.begin() + static_cast<std::ptrdiff_t>(ex.second + ex_len), tail.begin())) {
                    new_img = ex.first;
                    img_def_at = ex.second;
                    break;
                }
            }
            if (!new_img) {
                new_img = bound++;
                auto& ins = after_type_decls[img_def_at];
                ins.push_back((ilen << 16) | OpTypeImageOp);
                ins.push_back(new_img);
                ins.push_back(u32);
                ins.insert(ins.end(), tail.begin(), tail.end());
                created_images[tail] = {new_img, img_def_at};
            }
        }

        uint32_t new_pointee = new_img;
        size_t pointee_def_at = img_def_at;
        if (sampled) {
            uint32_t new_simg = 0;
            if (auto cs = created_simgs.find(new_img); cs != created_simgs.end()) {
                new_simg = cs->second.first;
                pointee_def_at = cs->second.second;
            } else {
                for (const auto& ex : simg_to_img) {
                    if (ex.second == new_img) {
                        new_simg = ex.first;
                        pointee_def_at = std::max(pointee_def_at, simg_at[ex.first]);
                        break;
                    }
                }
                if (!new_simg) {
                    new_simg = bound++;
                    auto& ins = after_type_decls[pointee_def_at];
                    ins.insert(ins.end(), {(3u << 16) | OpTypeSampledImageOp, new_simg, new_img});
                    created_simgs[new_img] = {new_simg, pointee_def_at};
                }
            }
            new_pointee = new_simg;
        }

        uint32_t new_ptr = 0;
        size_t ptr_def_at = pointee_def_at;
        const std::pair<uint32_t, uint32_t> pk = {storage, new_pointee};
        if (auto cp = created_ptrs.find(pk); cp != created_ptrs.end()) {
            new_ptr = cp->second.first;
            ptr_def_at = cp->second.second;
        } else {
            for (const auto& ex : ptrs) {
                if (ex.second == pk) {
                    new_ptr = ex.first;
                    ptr_def_at = std::max(ptr_def_at, ptr_at[ex.first]);
                    break;
                }
            }
            if (!new_ptr) {
                new_ptr = bound++;
                auto& ins = after_type_decls[ptr_def_at];
                ins.insert(ins.end(), {(4u << 16) | OpTypePointer, new_ptr, storage, new_pointee});
                created_ptrs[pk] = {new_ptr, ptr_def_at};
            }
        }
        var_new_ptr[vid] = new_ptr;
        var_load_ty[vid] = new_pointee;
        var_img_ty[vid] = new_img;
        const size_t vat = var_at[vid];
        if (vat <= ptr_def_at) {
            relocated_vars.insert(vid);
            const uint32_t vlen = in[vat] >> 16;
            auto& ins = after_type_decls[ptr_def_at];
            ins.push_back(in[vat]);
            ins.push_back(new_ptr);
            ins.insert(ins.end(),
                       in.begin() + static_cast<std::ptrdiff_t>(vat + 2),
                       in.begin() + static_cast<std::ptrdiff_t>(vat + vlen));
        }
    }
    if (var_new_ptr.empty()) return false;

    std::map<uint32_t, uint32_t> target_ssa;   // SSA value id -> variable id
    std::map<size_t, uint32_t> op_retype;      // inst.at -> new result type id
    struct BitcastAfter { uint32_t back_type, old_res, new_res; };
    std::map<size_t, BitcastAfter> bc_after;
    std::map<uint32_t, uint32_t> remap;

    for (const Inst& n : insts) {
        if (n.at < first_function || n.len < 4) continue;
        const uint32_t* w = &in[n.at];
        if (n.op == OpLoadOp && var_load_ty.count(w[3])) {
            target_ssa[w[2]] = w[3];
            op_retype[n.at] = var_load_ty[w[3]];
        } else if (n.op == OpCopyObjectOp && target_ssa.count(w[3])) {
            const uint32_t vid = target_ssa[w[3]];
            target_ssa[w[2]] = vid;
            op_retype[n.at] = var_load_ty[vid];
        } else if (n.op == OpSampledImageOp && target_ssa.count(w[3])) {
            const uint32_t vid = target_ssa[w[3]];
            target_ssa[w[2]] = vid;
            op_retype[n.at] = var_load_ty[vid];
        } else if (n.op == OpImageOp && target_ssa.count(w[3])) {
            const uint32_t vid = target_ssa[w[3]];
            target_ssa[w[2]] = vid;
            op_retype[n.at] = var_img_ty[vid];
        } else if (is_sample_op(n.op) && target_ssa.count(w[3])) {
            const uint32_t res_ty = w[1];
            uint32_t u_res_ty = 0;
            if (res_ty == f32) u_res_ty = u32;
            else if (auto vit = vecs.find(res_ty); vit != vecs.end() && vit->second.first == f32 && vit->second.second == 4) u_res_ty = v4u;
            if (!u_res_ty) continue;
            op_retype[n.at] = u_res_ty;
            const uint32_t bc_id = bound++;
            bc_after[n.at] = {res_ty, w[2], bc_id};
            remap[w[2]] = bc_id;
        }
    }
    if (bc_after.empty()) return false;

    out.clear();
    out.reserve(in.size() + pre_types.size() + bc_after.size() * 4 + 32);
    out.insert(out.end(), in.begin(), in.begin() + 5);
    out[3] = bound;

    for (const Inst& n : insts) {
        if (n.at == first_img_at && !pre_types.empty()) out.insert(out.end(), pre_types.begin(), pre_types.end());
        if (n.op == OpVariable && n.len >= 4 && relocated_vars.count(in[n.at + 2])) {
            if (auto dt = after_type_decls.find(n.at); dt != after_type_decls.end()) {
                out.insert(out.end(), dt->second.begin(), dt->second.end());
            }
            continue;
        }
        const size_t at = out.size();
        out.insert(out.end(), in.begin() + static_cast<std::ptrdiff_t>(n.at), in.begin() + static_cast<std::ptrdiff_t>(n.at + n.len));
        if (n.op == OpVariable && n.len >= 4) {
            if (auto vt = var_new_ptr.find(in[n.at + 2]); vt != var_new_ptr.end()) out[at + 1] = vt->second;
        }
        if (auto rt = op_retype.find(n.at); rt != op_retype.end()) out[at + 1] = rt->second;
        if (n.at >= first_function && !remap.empty()) {
            const bool no_res = (n.op == OpStoreOp || n.op == OpCopyMemoryOp || n.op == OpCopyMemorySizedOp ||
                                 n.op == OpImageWriteOp || (n.op >= 218 && n.op <= 225) || (n.op >= 246 && n.op <= 255) || n.op == 4416);
            const uint32_t first_op = no_res ? 1u : 3u;
            for (uint32_t k = first_op; k < n.len; ++k) {
                if (auto r = remap.find(out[at + k]); r != remap.end()) out[at + k] = r->second;
            }
        }
        if (auto dt = after_type_decls.find(n.at); dt != after_type_decls.end()) {
            out.insert(out.end(), dt->second.begin(), dt->second.end());
        }
        if (auto ba = bc_after.find(n.at); ba != bc_after.end()) {
            out.insert(out.end(), {(4u << 16) | OpBitcastOp, ba->second.back_type, ba->second.new_res, ba->second.old_res});
        }
    }
    return true;
}

// ---- fragment storage-image atomic coordinate scaling (issue #94) ---------------------------------------------------------------
// Ryujinx divides gl_FragCoord.xy by support_buffer.render_scale[0] so guest pixel math stays in 1x coordinates, which leaves
// OpImageTexelPointer coordinates unscaled (1x) and forces the runtime to drop the storage image + every render target in the
// same pass (and every pass sharing its depth buffer) back to 1x. When a fragment shader only indexes a 2D non-arrayed non-MS
// storage image via OpImageTexelPointer with an ivec2/uvec2 constructed from scalars (e.g. XC2's light-list mask shaders),
// replacing the OpImageTexelPointer coordinate with raw ivec2(gl_FragCoord.xy) keeps the storage image 1:1 with the scaled
// render targets at every scale (and is bit-identical at scale 1.0).
namespace {

struct FragAtomicPlan {
    bool ok = false;
    uint32_t frag_var = 0;
    uint32_t f32_ty = 0;
    uint32_t v4f_ty = 0;
    size_t v4f_end = 0;
    uint32_t v2f_ty = 0;
    struct Site {
        size_t at = 0;
        uint32_t coord_ty = 0;
        bool is_unsigned = false;
    };
    std::vector<Site> sites;
};

FragAtomicPlan analyze_frag_image_atomic(const std::vector<uint32_t>& in) {
    FragAtomicPlan plan;
    enum : uint32_t {
        OpEntryPointOp = 15, OpTypeIntOp = 21, OpTypeFloatOp = 22, OpTypeVectorOp = 23, OpTypeImageOp = 25,
        OpTypePointerOp = 32, OpVariableOp = 59, OpImageTexelPointerOp = 60, OpDecorateOp = 71,
        OpCompositeConstructOp = 80, OpImageWriteOp = 99, OpFDivOp = 136,
        ModelFragment = 4, StorageClassInput = 1, DecorationBuiltIn = 11, BuiltInFragCoord = 15,
    };
    if (in.size() < 5) return plan;
    bool is_fragment = false, has_fdiv = false;
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> int_ty;          // id -> (width, signedness)
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> vec_ty;          // id -> (comp_ty, count)
    std::map<uint32_t, bool> img2d_ty;                                 // id -> 2D non-arrayed non-MS
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> ptr_ty;          // id -> (storage_class, pointee)
    std::map<uint32_t, uint32_t> var_ty;                               // var_id -> ptr_type
    std::map<uint32_t, uint32_t> composite2_ty;                        // res_id -> vec2 type
    uint32_t frag_var = 0;
    size_t v4f_end = 0;

    for (size_t i = 5; i < in.size();) {
        const uint32_t len = in[i] >> 16, op = in[i] & 0xffff;
        if (!len || i + len > in.size()) return plan;
        if (op == OpEntryPointOp && len >= 3 && in[i + 1] == ModelFragment) is_fragment = true;
        else if (op == OpImageWriteOp) return plan;   // do not touch shaders with explicit OpImageWrite
        else if (op == OpFDivOp) has_fdiv = true;
        else if (op == OpDecorateOp && len >= 4 && in[i + 2] == DecorationBuiltIn && in[i + 3] == BuiltInFragCoord)
            frag_var = in[i + 1];
        else if (op == OpTypeIntOp && len >= 4) int_ty[in[i + 1]] = {in[i + 2], in[i + 3]};
        else if (op == OpTypeFloatOp && len >= 3 && in[i + 2] == 32) plan.f32_ty = in[i + 1];
        else if (op == OpTypeVectorOp && len >= 4) {
            vec_ty[in[i + 1]] = {in[i + 2], in[i + 3]};
            if (plan.f32_ty && in[i + 2] == plan.f32_ty) {
                if (in[i + 3] == 4) { plan.v4f_ty = in[i + 1]; v4f_end = i + len; }
                else if (in[i + 3] == 2) plan.v2f_ty = in[i + 1];
            }
        } else if (op == OpTypeImageOp && len >= 9) {
            img2d_ty[in[i + 1]] = (in[i + 3] == 1 && in[i + 5] == 0 && in[i + 6] == 0);   // Dim2D, !Arrayed, !MS
        } else if (op == OpTypePointerOp && len >= 4) {
            ptr_ty[in[i + 1]] = {in[i + 2], in[i + 3]};
        } else if (op == OpVariableOp && len >= 4) {
            var_ty[in[i + 2]] = in[i + 1];
        } else if (op == OpCompositeConstructOp && len == 5) {
            composite2_ty[in[i + 2]] = in[i + 1];
        } else if (op == OpImageTexelPointerOp && len == 6) {
            const uint32_t img_var = in[i + 3], coord = in[i + 4];
            auto iv = var_ty.find(img_var);
            if (iv == var_ty.end()) return plan;
            auto ip = ptr_ty.find(iv->second);
            if (ip == ptr_ty.end()) return plan;
            auto it = img2d_ty.find(ip->second.second);
            if (it == img2d_ty.end() || !it->second) return plan;
            auto ct = composite2_ty.find(coord);
            if (ct == composite2_ty.end()) return plan;
            auto vt = vec_ty.find(ct->second);
            if (vt == vec_ty.end() || vt->second.second != 2) return plan;
            auto nt = int_ty.find(vt->second.first);
            if (nt == int_ty.end() || nt->second.first != 32) return plan;
            plan.sites.push_back({i, ct->second, nt->second.second == 0});
        }
        i += len;
    }
    if (!is_fragment || !has_fdiv || !frag_var || !plan.f32_ty || !plan.v4f_ty || plan.sites.empty()) return plan;
    auto fv = var_ty.find(frag_var);
    if (fv == var_ty.end()) return plan;
    auto fp = ptr_ty.find(fv->second);
    if (fp == ptr_ty.end() || fp->second.first != StorageClassInput || fp->second.second != plan.v4f_ty) return plan;
    plan.frag_var = frag_var;
    plan.v4f_end = v4f_end;
    plan.ok = true;
    return plan;
}

}  // namespace

bool spirv_can_scale_frag_image_atomic(const std::vector<uint32_t>& in) {
    return analyze_frag_image_atomic(in).ok;
}

bool spirv_scale_frag_image_atomic(const std::vector<uint32_t>& in, std::vector<uint32_t>& out) {
    const FragAtomicPlan plan = analyze_frag_image_atomic(in);
    if (!plan.ok) return false;
    enum : uint32_t {
        OpTypeVectorOp = 23, OpImageTexelPointerOp = 60, OpLoadOp = 61, OpVectorShuffleOp = 79,
        OpConvertFToUOp = 109, OpConvertFToSOp = 110,
    };
    uint32_t bound = in[3];
    uint32_t v2f_ty = plan.v2f_ty;
    const bool need_v2f = (v2f_ty == 0);
    if (need_v2f) v2f_ty = bound++;

    std::map<size_t, FragAtomicPlan::Site> site_map;
    for (const FragAtomicPlan::Site& s : plan.sites) site_map[s.at] = s;

    out.clear();
    out.reserve(in.size() + (need_v2f ? 4 : 0) + plan.sites.size() * 15);
    out.insert(out.end(), in.begin(), in.begin() + 5);

    for (size_t i = 5; i < in.size();) {
        const uint32_t len = in[i] >> 16;
        if (!len || i + len > in.size()) { out.clear(); return false; }
        auto sit = site_map.find(i);
        if (sit != site_map.end()) {
            const uint32_t id_fc = bound++;
            const uint32_t id_xy = bound++;
            const uint32_t id_ixy = bound++;
            const uint32_t cvt_op = sit->second.is_unsigned ? OpConvertFToUOp : OpConvertFToSOp;
            out.insert(out.end(), {(4u << 16) | OpLoadOp, plan.v4f_ty, id_fc, plan.frag_var});
            out.insert(out.end(), {(7u << 16) | OpVectorShuffleOp, v2f_ty, id_xy, id_fc, id_fc, 0u, 1u});
            out.insert(out.end(), {(4u << 16) | cvt_op, sit->second.coord_ty, id_ixy, id_xy});
            out.insert(out.end(), {(6u << 16) | OpImageTexelPointerOp, in[i + 1], in[i + 2], in[i + 3], id_ixy, in[i + 5]});
        } else {
            out.insert(out.end(), in.begin() + static_cast<std::ptrdiff_t>(i),
                       in.begin() + static_cast<std::ptrdiff_t>(i + len));
            if (need_v2f && i + len == plan.v4f_end) {
                out.insert(out.end(), {(4u << 16) | OpTypeVectorOp, v2f_ty, plan.f32_ty, 2u});
            }
        }
        i += len;
    }
    out[3] = bound;
    return true;
}

}  // namespace nxrt::nvn::be::gpu
