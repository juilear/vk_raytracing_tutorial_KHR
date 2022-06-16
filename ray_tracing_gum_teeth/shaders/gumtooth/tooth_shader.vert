#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common.glsl"

layout(binding = 0) uniform _GlobalUniforms
{
    GlobalUniforms gUnifroms;
};

layout(push_constant) uniform _PushConstantRaster
{
    PushConstantRaster pcRaster;
};

layout(location = 0) in vec3 i_position;
layout(location = 1) in vec3 i_normal;

layout(location = 1) out vec3 o_ec_position;
layout(location = 2) out vec3 o_ec_normal;
layout(location = 3) out vec3 o_local_position;

void main()
{
    o_ec_normal = normalize(gUnifroms.normal_matrix * i_normal);
    o_ec_position = vec3(gUnifroms.view_matrix * pcRaster.model_matrix * vec4(i_position, 1.0));
    o_local_position = i_position;
	
	gl_Position = gUnifroms.projection_matrix * gUnifroms.view_matrix * pcRaster.model_matrix * vec4(i_position, 1.0);
}
