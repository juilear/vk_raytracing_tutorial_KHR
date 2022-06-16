#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "common.glsl"

layout(push_constant) uniform _PushConstantRaster
{
    PushConstantRaster pcRaster;
}; 

layout(location = 1) in vec3 i_ec_position;
layout(location = 2) in vec3 i_ec_normal;
layout(location = 3) in vec3 i_local_position;

layout(location = 0) out vec4 o_color;

layout(binding = eObjDescs) buffer TeethDesc_ { ObjDesc i[];} teethDescs;
layout(binding = eTextures) uniform sampler2D[] textureSamplers;

vec3 HighAmbientLight = vec3(0.56471f, 0.56471f, 0.56471f);
float RoughnessFactor = 50.23772;
float RoughnessRoot = 12.0;
float SelfIllumination = 0.0;
float AmbientReflectionFactor = 0.94333;
float AmbientReflection = 0.11013;
float HighLightFactor = 0.67769;
vec3 ReflectionColor = vec3(1.0, 1.0, 1.0);
LightInfo light[3];

 
vec3 HighLight(vec3 position, vec3 normal, vec3 specNormal, WaveFrontMaterial material, mat3 toObjectLocal)
{
    ObjDesc res = teethDescs.i[pcRaster.objIndex];
    vec3 eyeDir = toObjectLocal * normalize(-position.xyz);

    vec3 lighting = HighAmbientLight;
    for (int i = 0; i < 3; i++)
    {
        vec3 lightDir = toObjectLocal * normalize(light[i].position - position);
        lighting += max(dot(normal, lightDir), 0.0) *light[i].ambient;
    }

    vec3 highLight = vec3(0.0, 0.0, 0.0);
    for (int i = 0; i < 3; i++)
    {
        vec3 lightDir = toObjectLocal * normalize(light[i].position - position);
        float highLightValue = max(dot(normalize(reflect(-lightDir, specNormal)), eyeDir), 0.001);
        float tempV = 1.0 / max(pow(highLightValue, RoughnessFactor), 0.001);
        highLight += light[i].ambient * pow(RoughnessRoot, 1.0 - tempV) * tempV;
    }
    highLight = highLight*material.specular;

    vec3 fColor = mix(material.ambient.rgb * lighting, material.ambient.rgb, SelfIllumination);
    fColor *= vec3(AmbientReflectionFactor, AmbientReflectionFactor, AmbientReflectionFactor) + ReflectionColor * AmbientReflection;
    vec3 resultColor = fColor + highLight * HighLightFactor;
    return highLight;
    return material.ambient;
    return vec3(1.0, 0.0, 0.0);
}

void main()
{
    light[0].position = vec3(0.0, 0.0, 4.0);
    light[0].ambient  = vec3(0.40784, 0.40784, 0.40784);
    light[1].position = vec3(2.0, 0.0, -4.0);
    light[1].ambient  = vec3(0.01961, 0.0, 0.09804);
    light[2].position = vec3(-2.0, 0.0, -4.0);
    light[2].ambient  = vec3(0.01961, 0.0, 0.09804);

    ObjDesc toothResource = teethDescs.i[pcRaster.objIndex];
	
    float u = (toothResource.max_x - i_local_position.x) / toothResource.range_x;
    float v = 1.0 - (toothResource.max_z - i_local_position.z) / toothResource.range_z;
    vec2 coord = vec2(u, v);
	
	WaveFrontMaterial material;
    material.ambient = texture(textureSamplers[0], coord).xyz;
    material.diffuse = material.ambient;
    material.specular = material.ambient;
	
	mat3 toObjectLocal = mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
	vec3 FrontColor = vec3(0.0);
    FrontColor += HighLight(i_ec_position, i_ec_normal, i_ec_normal, material, toObjectLocal);
    o_color = vec4(FrontColor, 1.0);
}
