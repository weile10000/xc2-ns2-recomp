// SPDX-License-Identifier: GPL-3.0-or-later
// SPIR-V rewrites the back end applies to the translated shaders when it builds a pipeline (render thread). The offline translator
// knows one shader at a time; the console's rules are looser than Vulkan's about what the stages of a pipeline have to agree on.
//
//   spirv_add_outputs   a fragment shader reads varyings (or an indexed array of them) that the stage before it never declared. The
//                       console hands it undefined values; Vulkan wants an output in every slot (VUID-RuntimeSpirv-OpEntryPoint-08743).
//                       The producer gets unwritten outputs there.
//   spirv_dual_source_output
//                       a fragment shader of a draw with dual-source blending: the console takes the second blend source from render
//                       target 1's output; Vulkan from the output at Location 0, Index 1 (and allows a single such attachment).
//
//   spirv_add_xfb      a vertex shader that runs with transform feedback: the Location outputs are captured into buffer 0 in Location
//                       order, each narrowed to the components the shader stores (a vec4 whose xyz are written is captured as 12
//                       bytes), tightly packed. The console's layout lives in the driver's compile data; this is the layout the
//                       game's stream-out shaders use (docs/S4e.md).
//
//   spirv_init_outputs  a vertex, tessellation evaluation or geometry shader that writes only some components of a Location output (the
//                       sky dome's vertex shader stores .w of one vec4 and .xyz of others) while the next stage reads all four. The
//                       console (and Ryujinx) start every user output at (0,0,0,1); in Vulkan an unwritten component is undefined: AMD's
//                       Windows driver happens to hand 0, Mesa RADV hands garbage (a NaN that blacks out the HDR frame). The entry point
//                       now stores (0,0,0,1) to every Location output first, and a geometry shader again after each emitted vertex.
//
// The idea (pattern-matching on the module's declarations, new ids from the bound) follows SuperMarioWonderRecomp's
// nvn_spirv_patch.cpp (MIT); the code is new.
#pragma once
#include <cstdint>
#include <vector>

namespace nxrt::nvn::be::gpu {

enum class SpvScalar : uint8_t { Float, Sint, Uint };

struct SpvOutputSlot {
    uint32_t location = 0;
    SpvScalar type = SpvScalar::Float;
};

//   spirv_outputs_to_uint
//                       a fragment shader of a pass whose colour attachment is an integer format: the console's ROP stores the register
//                       bits unchanged and the game's shaders hand them over as floats (intBitsToFloat). Vulkan requires an integer
//                       output for an integer attachment, so every Location output is redeclared as the same-width unsigned integer
//                       vector and the stored value is bitcast instead of converted. Needed for a render target that takes a logic op:
//                       logic operations are only defined for integer formats (the light-probe list t50, HANDOVER 35).
//
// Adds one never-written vec4 Output variable per slot (a 4-component vector of the slot's scalar type) to the entry point of `in`.
// Returns false (out untouched) when the module cannot be rewritten (malformed, no entry point, a slot that already has an output).
bool spirv_add_outputs(const std::vector<uint32_t>& in, const std::vector<SpvOutputSlot>& slots, std::vector<uint32_t>& out);

// Vertex inputs that carry integers. The console hands a vertex shader the raw bits of an integer vertex attribute (RGBA8UI bone
// indices, say) in the float registers the translation reads, and the translated shader declares every input as a float and
// bitcasts it where it wants an integer. A Vulkan integer format needs an integer input. For each slot the Input variable becomes a
// Private variable and a new Input variable of uvec4 (ivec4) takes its Location; the entry point copies it over with an OpBitcast, so
// the shader reads the same bits as on the console. The slot's input is a vec4 of floats, or an element of an Input array of vec4 (the
// translation of a shader that indexes its attributes by a computed value): then the array becomes Private as a whole, every Location it
// covers gets its own new Input (float, uvec4 or ivec4 by what the draw feeds) and the entry point builds the array from them. False (out
// untouched) otherwise.
struct SpvIntSlot {
    uint32_t location = 0;
    bool is_signed = false;
};
bool spirv_int_attributes(const std::vector<uint32_t>& in, const std::vector<SpvIntSlot>& slots, std::vector<uint32_t>& out);
// The vertex input locations (is_signed = false) a game would have to feed with integer attributes: the vector inputs whose every use, in any
// component, is a bitcast to a 32-bit integer (the translator reads raw integer attribute bits that way). A guess from the code alone, made
// so the patched variant can be compiled before the first draw; a draw whose real integer slots differ builds its own variant.
std::vector<SpvIntSlot> spirv_predict_int_inputs(const std::vector<uint32_t>& in);

// A geometry shader the translator made for a console "passthrough" geometry shader (the hardware copies every attribute of the input
// vertices to the output vertices; the shader code only sets the position and gl_Layer, so its translation writes no Location output at
// all). Adds the outputs for `slots` like spirv_add_outputs and makes the k-th OpEmitVertex copy the input at the same Location of vertex k.
// Needs an Input array [3] of vec4 at each slot's Location and exactly three OpEmitVertex in a module with triangle input. False (out
// untouched) otherwise; the caller then uses spirv_add_outputs.
bool spirv_gs_passthrough(const std::vector<uint32_t>& in, const std::vector<SpvOutputSlot>& slots, std::vector<uint32_t>& out);

// Rewrites a geometry shader's input primitive ExecutionMode (InputPoints / InputLines / Triangles / ...) and the length of its
// Input interface arrays (gl_in[], per-vertex Location inputs[]) to match `vk_topology` (VkPrimitiveTopology). Necessary because
// console geometry shaders take their input primitive from the draw call, so the offline translator defaults to Triangles even for
// POINT_LIST draws; passing POINT_LIST into a Triangles GS triggers AMD RDNA2 NGG hangs and screen corruption. Returns false (out
// untouched) if the module is not a geometry shader or already matches `vk_topology`.
bool spirv_gs_input_topology(const std::vector<uint32_t>& in, uint32_t vk_topology, std::vector<uint32_t>& out);

// Moves the fragment output at Location 1 to Location 0, Index 1. False (out untouched) when the module has no such output.
bool spirv_dual_source_output(const std::vector<uint32_t>& in, std::vector<uint32_t>& out);

// A fragment shader of a pass without colour attachments (keep_mask == 0), or with fewer bound colour attachments than declared:
// every Location output whose bit is not in `keep_mask` becomes a private variable (the Location/Index/Component decorations go, and
// the entry point's interface loses it before SPIR-V 1.4). Built-in outputs (FragDepth, SampleMask) are preserved.
bool spirv_outputs_to_private(const std::vector<uint32_t>& in, std::vector<uint32_t>& out, uint32_t keep_mask = 0);

// spirv_add_early_fragment_tests: a fragment shader that writes a storage image gets EarlyFragmentTests, so the depth / stencil test
// runs before it - the console's order. Without it a driver may run the shader for fragments the test rejects, and their image writes
// land while their colour does not (HANDOVER 42.21).
// amd: use EarlyAndLateFragmentTestsAMD (mode 5017, extension SPV_AMD_shader_early_and_late_fragment_tests) instead of
// EarlyFragmentTests - the mode amdvlk takes without faulting while it builds the pipeline.
bool spirv_add_early_fragment_tests(const std::vector<uint32_t>& in, std::vector<uint32_t>& out, bool amd = false);

// Stores the console's start value (component 3 = 1, the others 0, counted from the Component decoration; 1.0f for floats, 1 for
// integers) to every Location output a Vertex / TessellationEvaluation / Geometry entry point lists, right after the leading OpVariables
// of its first block, and in a geometry shader again after every OpEmitVertex / OpEmitStreamVertex. Covers scalars and vectors of 32-bit
// float / int and Output arrays of them; built-ins are left alone. False (out untouched) for another stage, an unexpected module or
// nothing to initialise.
bool spirv_init_outputs(const std::vector<uint32_t>& in, std::vector<uint32_t>& out);
bool spirv_outputs_to_uint(const std::vector<uint32_t>& in, std::vector<uint32_t>& out);
// Why the last call refused, or "" (diagnostic only, thread local).
const char* spirv_outputs_to_uint_reason();

// A fragment sampler binding that the shader declared as float (sampler2D) but the draw feeds with an unsigned integer
// texture (R8_UINT / R16_UINT / R32_UINT). Rewrites those (set, binding) variables to usampler and bitcasts the sampled/fetched
// uint bits back to the shader's float type so floatBitsToInt(...) reads the console's raw bits legally in Vulkan.
struct SpvSamplerSlot {
    uint32_t set = 0;
    uint32_t binding = 0;
};
bool spirv_uint_samplers(const std::vector<uint32_t>& in, const std::vector<SpvSamplerSlot>& slots, std::vector<uint32_t>& out);

struct SpvXfbOutput {
    uint32_t location, components, offset;   // offset in bytes inside a captured vertex
};
// Adds XfbBuffer / XfbStride / Offset decorations, the TransformFeedback capability and the Xfb execution mode. layout receives the
// captured outputs, stride the bytes of one captured vertex. False (out untouched): no Location outputs or an unexpected module.
bool spirv_add_xfb(const std::vector<uint32_t>& in, std::vector<uint32_t>& out, std::vector<SpvXfbOutput>* layout, uint32_t* stride);

// spirv_scale_frag_image_atomic: a fragment shader that performs image atomic ops (OpImageTexelPointer) on a 2D storage image
// at ivec2(trunc(gl_FragCoord.xy / render_scale[0])) (e.g. XC2 NS2 light-list registration mask R32UI 640x360, issue #94).
// Rewrites the OpImageTexelPointer coordinate to raw ivec2(gl_FragCoord.xy) so the storage image, its paired light-list RT, and
// the shared half-res depth/effect targets can all stay at the attachment scale instead of cascading back to 1x.
// Bit-identical when render_scale == 1.0.
bool spirv_can_scale_frag_image_atomic(const std::vector<uint32_t>& in);
bool spirv_scale_frag_image_atomic(const std::vector<uint32_t>& in, std::vector<uint32_t>& out);

}  // namespace nxrt::nvn::be::gpu
