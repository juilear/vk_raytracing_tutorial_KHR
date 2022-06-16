#define STANDARD [object Object]
#define USE_NORMALMAP 
#define USE_UV 
#define USE_MAP 
#define USE_SPECULARMAP 
#define USE_REFLECTMAP 
#define MORPH_BINORMAL 
#define USE_MORPHTARGETS 
#define CLIP_PLANES_COUNT 0


precision highp float;
precision highp int;
#define GAMMA_FACTOR 2
#define MAX_BONES 0

uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat3 normalMatrix;
uniform vec3 cameraPosition;

attribute vec3 position;
attribute vec3 normal;
attribute vec2 uv;

#ifdef USE_MORPHTARGETS

	attribute vec3 morphTarget0;
	attribute vec3 morphTarget1;
	attribute vec3 morphTarget2;
	attribute vec3 morphTarget3;

	attribute vec3 morphTarget4;
	attribute vec3 morphTarget5;
	attribute vec3 morphTarget6;
	attribute vec3 morphTarget7;

#endif

#ifdef USE_NORMALMAP
attribute vec3 binormals;
varying vec3 binormal;
#endif

#ifdef MORPH_BINORMAL
attribute vec3 morphBinormals;
#endif

// varying vec3 vViewPosition;
varying vec4 modelPosition;
varying vec3 mPos; 

#ifdef USE_UV

	varying vec2 vUv;

	uniform mat3 uvTransform;
#endif

#ifdef USE_MORPHTARGETS

	uniform float morphTargetBaseInfluence;

	uniform float morphTargetInfluences[ 8 ];

#endif

attribute vec2 syntheticUV;
#if defined( USE_MAP )
varying vec2 sUv;
#endif
varying float depthValue;

void main(){
  #ifdef USE_UV

	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;

#endif
  #if defined( USE_MAP )
    sUv = syntheticUV;
  #endif

  vec3 transformed = vec3( position );

  vec3 objectNormal = vec3( normal );

#ifdef USE_TANGENT

	vec3 objectTangent = vec3( tangent.xyz );

#endif
  #ifdef USE_MORPHTARGETS

	// morphTargetBaseInfluence is set based on BufferGeometry.morphTargetsRelative value:
	// When morphTargetsRelative is false, this is set to 1 - sum(influences); this results in position = sum((target - base) * influence)
	// When morphTargetsRelative is true, this is set to 1; as a result, all morph targets are simply added to the base after weighting
	transformed *= morphTargetBaseInfluence;
	transformed += morphTarget0 * morphTargetInfluences[ 0 ];
	transformed += morphTarget1 * morphTargetInfluences[ 1 ];
	transformed += morphTarget2 * morphTargetInfluences[ 2 ];
	transformed += morphTarget3 * morphTargetInfluences[ 3 ];


	transformed += morphTarget4 * morphTargetInfluences[ 4 ];
	transformed += morphTarget5 * morphTargetInfluences[ 5 ];
	transformed += morphTarget6 * morphTargetInfluences[ 6 ];
	transformed += morphTarget7 * morphTargetInfluences[ 7 ];

#endif


  #ifdef USE_NORMALMAP
    binormal = normalMatrix * normalize(binormals);
  #endif

  #ifdef USE_MORPHTARGETS
    #ifdef MORPH_BINORMAL
      binormal *= morphTargetBaseInfluence;
      binormal += normalMatrix * normalize(morphBinormals) * morphTargetInfluences[0];
    #endif
  #endif

  
    #ifdef USE_MORPHTARGETS
      vec4 mvPosition = modelViewMatrix * vec4( transformed, 1.0);
    #endif

  vec4 projectedVertex  = projectionMatrix * mvPosition;
  gl_Position = projectedVertex;
  depthValue = projectedVertex.w;
// vViewPosition = - mvPosition.xyz;
  modelPosition = mvPosition;
  mPos = position;
}