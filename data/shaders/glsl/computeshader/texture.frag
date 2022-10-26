#version 450

layout (binding = 1) uniform sampler2D samplerColor;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	vec4 currentColor = texture(samplerColor, inUV);
	outFragColor = vec4(currentColor.rgb, currentColor.a - 0.1f);
}